# Mein Modell ist ein nn.Sequential aus:
# Stem (3 Blöcke)
# MaxPool (1 Block)
# Residual Blocks (4 große Blöcke: Block1-Block4, jeweils als nn.Sequential)
# Head (Klassifikator)
#
# Blockstruktur ist also:
# [0] stem1
# [1] stem2
# [2] stem3
# [3] maxpool
# [4] block1 (layer1)
# [5] block2 (layer2)
# [6] block3 (layer3)
# [7] block4 (layer4)
# [8] head

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.basic_conv1d import create_head1d, Flatten

import numpy as np
import os
import pickle
from pathlib import Path

from enum import Enum
import re
import inspect

# --- XResNet1d Architektur ---
def delegates(to=None, keep=False):
    "Decorator: replace `**kwargs` in signature with params from `to`"
    def _f(f):
        if to is None: to_f,from_f = f.__base__.__init__,f.__init__
        else:          to_f,from_f = to,f
        sig = inspect.signature(from_f)
        sigd = dict(sig.parameters)
        k = sigd.pop('kwargs')
        s2 = {k:v for k,v in inspect.signature(to_f).parameters.items()
              if v.default != inspect.Parameter.empty and k not in sigd}
        sigd.update(s2)
        if keep: sigd['kwargs'] = k
        from_f.__signature__ = sig.replace(parameters=sigd.values())
        return f
    return _f

def store_attr(self, nms):
    "Store params named in comma-separated `nms` from calling context into attrs in `self`"
    mod = inspect.currentframe().f_back.f_locals
    for n in re.split(', *', nms): setattr(self,n,mod[n])

NormType = Enum('NormType', 'Batch BatchZero Weight Spectral Instance InstanceZero')

def _conv_func(ndim=2, transpose=False):
    "Return the proper conv `ndim` function, potentially `transposed`."
    assert 1 <= ndim <=3
    return getattr(nn, f'Conv{"Transpose" if transpose else ""}{ndim}d')

def init_default(m, func=nn.init.kaiming_normal_):
    "Initialize `m` weights with `func` and set `bias` to 0."
    if func and hasattr(m, 'weight'): func(m.weight)
    with torch.no_grad():
        if getattr(m, 'bias', None) is not None: m.bias.fill_(0.)
    return m
    
def _get_norm(prefix, nf, ndim=2, zero=False, **kwargs):
    "Norm layer with `nf` features and `ndim` initialized depending on `norm_type`."
    assert 1 <= ndim <= 3
    bn = getattr(nn, f"{prefix}{ndim}d")(nf, **kwargs)
    if bn.affine:
        bn.bias.data.fill_(1e-3)
        bn.weight.data.fill_(0. if zero else 1.)
    return bn 

def BatchNorm(nf, ndim=2, norm_type=NormType.Batch, **kwargs):
    "BatchNorm layer with `nf` features and `ndim` initialized depending on `norm_type`."
    return _get_norm('BatchNorm', nf, ndim, zero=norm_type==NormType.BatchZero, **kwargs)

class ConvLayer(nn.Sequential):
    "Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and `norm_type` layers."
    def __init__(self, ni, nf, ks=3, stride=1, padding=None, bias=None, ndim=2, norm_type=NormType.Batch, bn_1st=True,
                 act_cls=nn.ReLU, transpose=False, init=nn.init.kaiming_normal_, xtra=None, **kwargs):
        if padding is None: padding = ((ks-1)//2 if not transpose else 0)
        bn = norm_type in (NormType.Batch, NormType.BatchZero)
        inn = norm_type in (NormType.Instance, NormType.InstanceZero)
        if bias is None: bias = not (bn or inn)
        conv_func = _conv_func(ndim, transpose=transpose)
        conv = init_default(conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding, **kwargs), init)
        if   norm_type==NormType.Weight:   conv = weight_norm(conv)
        elif norm_type==NormType.Spectral: conv = spectral_norm(conv)
        layers = [conv]
        act_bn = []
        if act_cls is not None: act_bn.append(act_cls())
        if bn: act_bn.append(BatchNorm(nf, norm_type=norm_type, ndim=ndim))
        if inn: act_bn.append(InstanceNorm(nf, norm_type=norm_type, ndim=ndim))
        if bn_1st: act_bn.reverse()
        layers += act_bn
        if xtra: layers.append(xtra)
        super().__init__(*layers)

def AdaptiveAvgPool(sz=1, ndim=2):
    "nn.AdaptiveAvgPool layer for `ndim`"
    assert 1 <= ndim <= 3
    return getattr(nn, f"AdaptiveAvgPool{ndim}d")(sz)

def MaxPool(ks=2, stride=None, padding=0, ndim=2, ceil_mode=False):
    "nn.MaxPool layer for `ndim`"
    assert 1 <= ndim <= 3
    return getattr(nn, f"MaxPool{ndim}d")(ks, stride=stride, padding=padding)

def AvgPool(ks=2, stride=None, padding=0, ndim=2, ceil_mode=False):
    "nn.AvgPool layer for `ndim`"
    assert 1 <= ndim <= 3
    return getattr(nn, f"AvgPool{ndim}d")(ks, stride=stride, padding=padding, ceil_mode=ceil_mode)

class ResBlock(nn.Module):
    "Resnet block from `ni` to `nh` with `stride`"
    @delegates(ConvLayer.__init__)
    def __init__(self, expansion, ni, nf, stride=1, kernel_size=3, groups=1, reduction=None, nh1=None, nh2=None, dw=False, g2=1,
                 sa=False, sym=False, norm_type=NormType.Batch, act_cls=nn.ReLU, ndim=2,
                 pool=AvgPool, pool_first=True, **kwargs):
        super().__init__()
        norm2 = (NormType.BatchZero if norm_type==NormType.Batch else
                 NormType.InstanceZero if norm_type==NormType.Instance else norm_type)
        if nh2 is None: nh2 = nf
        if nh1 is None: nh1 = nh2
        nf,ni = nf*expansion,ni*expansion
        k0 = dict(norm_type=norm_type, act_cls=act_cls, ndim=ndim, **kwargs)
        k1 = dict(norm_type=norm2, act_cls=None, ndim=ndim, **kwargs)
        layers  = [ConvLayer(ni,  nh2, kernel_size, stride=stride, groups=ni if dw else groups, **k0),
                   ConvLayer(nh2,  nf, kernel_size, groups=g2, **k1)
        ] if expansion == 1 else [
                   ConvLayer(ni,  nh1, 1, **k0),
                   ConvLayer(nh1, nh2, kernel_size, stride=stride, groups=nh1 if dw else groups, **k0),
                   ConvLayer(nh2,  nf, 1, groups=g2, **k1)]
        self.convs = nn.Sequential(*layers)
        convpath = [self.convs]
        if reduction: convpath.append(SEModule(nf, reduction=reduction, act_cls=act_cls))
        if sa: convpath.append(SimpleSelfAttention(nf,ks=1,sym=sym))
        self.convpath = nn.Sequential(*convpath)
        idpath = []
        if ni!=nf: idpath.append(ConvLayer(ni, nf, 1, act_cls=None, ndim=ndim, **kwargs))
        if stride!=1: idpath.insert((1,0)[pool_first], pool(2, ndim=ndim, ceil_mode=True))
        self.idpath = nn.Sequential(*idpath)
        self.act = nn.ReLU(inplace=True) if act_cls is nn.ReLU else act_cls()

    def forward(self, x): return self.act(self.convpath(x) + self.idpath(x))

######################### adapted from vison.models.xresnet
def init_cnn(m):
    if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv1d, nn.Conv2d,nn.Linear)): nn.init.kaiming_normal_(m.weight)
    for l in m.children(): init_cnn(l)

class XResNet1d(nn.Sequential):
    @delegates(ResBlock)
    def __init__(self, block, expansion, layers, p=0.0, input_channels=3, num_classes=1000, stem_szs=(32,32,64),kernel_size=5,kernel_size_stem=5,
                 widen=1.0, sa=False, act_cls=nn.ReLU, lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True, act_head="relu", concat_pooling=True, **kwargs):
        store_attr(self, 'block,expansion,act_cls')
        stem_szs = [input_channels, *stem_szs]
        stem = [ConvLayer(stem_szs[i], stem_szs[i+1], ks=kernel_size_stem, stride=2 if i==0 else 1, act_cls=act_cls, ndim=1)
                for i in range(3)]

        #block_szs = [int(o*widen) for o in [64,128,256,512] +[256]*(len(layers)-4)]
        block_szs = [int(o*widen) for o in [64,64,64,64] +[32]*(len(layers)-4)]
        block_szs = [64//expansion] + block_szs
        blocks = [self._make_layer(ni=block_szs[i], nf=block_szs[i+1], blocks=l,
                                   stride=1 if i==0 else 2, kernel_size=kernel_size, sa=sa and i==len(layers)-4, ndim=1, **kwargs)
                  for i,l in enumerate(layers)]

        head = create_head1d(block_szs[-1]*expansion, nc=num_classes, lin_ftrs=lin_ftrs_head, ps=ps_head, bn_final=bn_final_head, bn=bn_head, act=act_head, concat_pooling=concat_pooling)
        
        super().__init__(
            *stem, nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            *blocks,
            head,
        )
        init_cnn(self)

    def _make_layer(self, ni, nf, blocks, stride, kernel_size, sa, **kwargs):
        return nn.Sequential(
            *[self.block(self.expansion, ni if i==0 else nf, nf, stride=stride if i==0 else 1,
                      kernel_size=kernel_size, sa=sa and i==(blocks-1), act_cls=self.act_cls, **kwargs)
              for i in range(blocks)])
    
    def get_layer_groups(self):
        return (self[3],self[-1])
    
    def get_output_layer(self):
        return self[-1][-1]
        
    def set_output_layer(self,x):
        self[-1][-1]=x


# --- Konstruktor für xresnet1d101 ---
def xresnet1d101(input_channels=3, num_classes=1000, **kwargs):
    """Erstellt ein XResNet1d101 Modell."""
    return XResNet1d(ResBlock, expansion=4, layers=[3, 4, 23, 3], input_channels=input_channels, num_classes=num_classes, **kwargs)

# def _xresnet1d(expansion, layers, **kwargs):
#     return XResNet1d(ResBlock, expansion, layers, **kwargs)

# --- Split-Funktion ---
def split_model_at_layer(model, layer: str):
    """
    Splits XResNet1d at a given block/layer into feature extractor and classification head.
    Args:
        model: XResNet1d instance.
        layer: Name of the layer (e.g. "block1", "block2", "block3", "block4").
    Returns:
        (feature_extractor, classification_head)
    """
    # Mapping von Layernamen zu Indizes im Sequential (ggf. anpassen!)
    block_to_idx = {
        "block1": 4,
        "block2": 5,
        "block3": 6,
        "block4": 7,
    }
    if layer not in block_to_idx:
        raise ValueError(f"Unbekannter Block: {layer}. Erlaubt: {list(block_to_idx.keys())}")

    idx = block_to_idx[layer]

    # Feature Extractor: alles bis einschließlich des gewünschten Blocks
    feature_extractor = torch.nn.Sequential(*list(model.children())[:idx+1])
    # Classification Head: alles danach (inkl. head!)
    classification_head = torch.nn.Sequential(*list(model.children())[idx+1:])

    return feature_extractor, classification_head


def get_transformation(model=None, standard_scaler=None):
    """
    Für ECG XResNet1D - gibt Standardisierungs-Transformationen zurück
    
    Returns:
        tuple: (identity_transform, standardizer_transform)
               - identity_transform: keine Änderung (für Kompatibilität)  
               - standardizer_transform: Funktion die apply_standardizer verwendet
    """
    import numpy as np
    
    # Identity Transform (macht nichts - für Kompatibilität mit Image-Modellen)
    def identity_transform(x):
        return x
    
    # Standardizer Transform - erwartet dass standard_scaler global verfügbar ist
    def standardizer_transform(X):
        """
        Anwendung des Standard Scalers auf ECG Daten
        """
        # 1. Versuche übergebenen Scaler
        scaler = standard_scaler

        # 2. Falls nicht übergeben, versuche aus __main__ ...

        # 3. Deine apply_standardizer Logik
        if X.ndim == 2:  # Einzelnes Sample
            x_shape = X.shape
            return scaler.transform(X.flatten()[:, np.newaxis]).reshape(x_shape)
        elif X.ndim == 3:  # Batch
            X_tmp = []
            for x in X:
                x_shape = x.shape
                X_tmp.append(scaler.transform(x.flatten()[:, np.newaxis]).reshape(x_shape))
            return np.array(X_tmp)
        else:
            raise ValueError(f"Unerwartete Input-Dimensionen: {X.shape}")
    
    return (identity_transform, standardizer_transform)



# Füge NACH der get_transformation Funktion hinzu:

class XResNet1DModel:
    def __init__(self, model_path, experiment, class_names, num_classes=5):
        """ECG XResNet1D Model wrapper for cxai."""
        self.experiment = experiment
        self.class_names = class_names
        self.num_classes = num_classes
        self.model_path = Path(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model loading - improve error handling
        self.model = self._load_model()
        self.model.eval()
        
        # Hook storage for intermediate activations
        self.hooks = {}
        self.activations = {}
        
    def _load_model(self):
        """Load the trained XResNet1D model."""
        try:
            # Try loading the model checkpoint
            model_files = list(self.model_path.glob("*.pkl")) + list(self.model_path.glob("*.pth"))
            
            if not model_files:
                raise FileNotFoundError(f"No model files found in {self.model_path}")
            
            # Load the most recent model file
            model_file = max(model_files, key=os.path.getctime)
            print(f"Loading model from: {model_file}")
            
            # Load depending on file extension
            if str(model_file).endswith('.pkl'):
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                # Extract the actual PyTorch model if it's wrapped
                if hasattr(model, 'model'):
                    model = model.model
                elif hasattr(model, 'learn'):
                    model = model.learn.model
            else:
                model = torch.load(model_file, map_location=self.device)
            
            return model.to(self.device)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def forward(self, x):
        """Forward pass through the model."""
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        
        if x.dim() == 2:  # (length, channels)
            x = x.unsqueeze(0)  # Add batch dimension: (1, length, channels)
        
        if x.dim() == 3 and x.shape[-1] == 12:  # (batch, length, channels)
            x = x.permute(0, 2, 1)  # -> (batch, channels, length)
        
        x = x.to(self.device)
        
        with torch.no_grad():
            output = self.model(x)
        
        return output
    
    def predict(self, x):
        """Predict with the model - return probabilities."""
        logits = self.forward(x)
        probabilities = torch.softmax(logits, dim=1)
        return probabilities.cpu().numpy()
    
    def _register_hook(self, layer_name, layer):
        """Register a forward hook for a specific layer."""
        def hook_fn(module, input, output):
            self.activations[layer_name] = output.detach()
        
        hook = layer.register_forward_hook(hook_fn)
        self.hooks[layer_name] = hook
        return hook
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks.values():
            hook.remove()
        self.hooks.clear()
        self.activations.clear()
    
    def get_intermediate_activation_and_context(self, layer, x, label=None):
        """
        Get intermediate activation from a specific layer.
        
        Args:
            layer (str): Layer name (e.g., "0", "4", "8")
            x (torch.Tensor): Input tensor
            label (int): Target label (for future use)
            
        Returns:
            tuple: (activation, context) where context contains relevant info
        """
        # Clean up previous hooks
        self._remove_hooks()
        
        # Find the target layer - handle numeric indices for Sequential model
        target_layer = None
        if layer.isdigit():
            # Direct index access for Sequential model
            layer_idx = int(layer)
            if layer_idx < len(self.model):
                target_layer = self.model[layer_idx]
                layer_name = f"{layer_idx}"
            else:
                raise ValueError(f"Layer index {layer_idx} out of range. Model has {len(self.model)} layers.")
        else:
            # Named layer access
            for name, module in self.model.named_modules():
                if name == layer:
                    target_layer = module
                    layer_name = name
                    break
        
        if target_layer is None:
            available_layers = [f"{i}" for i in range(len(self.model))]
            available_named = [name for name, _ in self.model.named_modules() 
                              if len(list(_.children())) == 0]
            raise ValueError(f"Layer '{layer}' not found. Available indices: {available_layers[:10]}... Available named layers: {available_named[:5]}...")
        
        # Register hook for target layer
        self._register_hook(layer_name, target_layer)
        
        # Forward pass to trigger hooks
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        
        if x.dim() == 2:  # (length, channels)
            x = x.unsqueeze(0)  # Add batch dimension
        
        if x.dim() == 3 and x.shape[-1] == 12:  # (batch, length, channels)
            x = x.permute(0, 2, 1)  # -> (batch, channels, length)
        
        x = x.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            output = self.model(x)
        
        # Get the activation
        if layer_name not in self.activations:
            raise RuntimeError(f"No activation captured for layer '{layer}'")
        
        activation = self.activations[layer_name]
        
        # Create context with useful information
        context = {
            'layer_name': layer_name,
            'layer_type': type(target_layer).__name__,
            'input_shape': tuple(x.shape),
            'output_shape': tuple(output.shape),
            'activation_shape': tuple(activation.shape),
            'model_output': output,
            'target_label': label
        }
        
        # Clean up hooks
        self._remove_hooks()
        
        return activation.cpu(), context
    
    def get_layer_names(self):
        """Get all available layer names."""
        # Return both numeric indices and named layers
        numeric_layers = [str(i) for i in range(len(self.model))]
        named_layers = [name for name, _ in self.model.named_modules() 
                       if len(list(_.children())) == 0]
        return numeric_layers + named_layers
    
    def __call__(self, x):
        """Make the model callable."""
        return self.forward(x)

# Update the __all__ list
__all__ = ["xresnet1d101", "split_model_at_layer", "XResNet1d", "ResBlock", "XResNet1DModel", "get_transformation"]