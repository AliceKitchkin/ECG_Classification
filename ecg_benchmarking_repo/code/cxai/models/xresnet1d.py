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

from ...models.basic_conv1d import create_head1d, Flatten

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



# (Optional) Liste der verfügbaren Modelle für autoimport
__all__ = ["xresnet1d101", "split_model_at_layer", "XResNet1d", "ResBlock"]