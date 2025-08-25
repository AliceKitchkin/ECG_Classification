import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from cxai.inspector import Inspector, InspectionRelevanceInfo
from cxai.explainers.base import WithSplitModelMixin

class GradientExplainer:
    """
    Simple gradient-based explainer for 1D ECG data.
    Computes gradients of the output with respect to the input.
    """
    
    def __init__(self, model, device="cpu"):
        """
        Initialize the gradient explainer.
        
        Args:
            model: PyTorch model to explain
            device: Device to run computations on
        """
        self.model = model
        self.device = device
        self.model = model.to(device)
        self.model.eval()
    

    def explain(self, x, label=None, **kwargs):
        """
        Generate gradient-based explanation.
        
        Args:
            x: Input tensor, shape (C, L) for single sample or (N, C, L) for batch
            label: Target class index for explanation
            
        Returns:
            tuple: (logits, attribution_map)
        """
        # Ensure input is tensor and has batch dimension
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        
        if x.dim() == 2:  # (C, L) -> (1, C, L)
            x = x.unsqueeze(0)
        
        x = x.to(self.device)
        x.requires_grad_(True)
        
        # Forward pass
        with torch.enable_grad():
            logits = self.model(x)
            
            # If no label specified, use predicted class
            if label is None:
                label = torch.argmax(logits, dim=1)
            elif isinstance(label, int):
                label = torch.tensor([label]).to(self.device)
            
            # Compute gradients
            target_logit = logits[0, label]
            gradients = torch.autograd.grad(
                outputs=target_logit,
                inputs=x,
                create_graph=False,
                retain_graph=False
            )[0]
        
        # Convert to numpy and remove batch dimension if single sample
        logits_np = logits.detach().cpu().numpy()
        gradients_np = gradients.detach().cpu().numpy()
        
        if gradients_np.shape[0] == 1:  # Single sample
            gradients_np = gradients_np.squeeze(0)
            logits_np = logits_np.squeeze(0)
        
        return logits_np, gradients_np


    def explain_with_inspector(
        self, x: torch.tensor, target_label: int, inspector: Inspector, top_k: int
    ) -> Tuple[np.ndarray, np.ndarray, InspectionRelevanceInfo]:
        """
        Generate explanation with subspace analysis using inspector.
        
        Args:
            x: Input tensor
            target_label: Target class index
            inspector: Inspector object for subspace analysis
            top_k: Number of top subspaces to analyze
            
        Returns:
            Tuple of (logits, original_heatmap, inspection_info)
        """
        # Get standard explanation first
        logits, original_heatmap = self.explain(x, target_label)
        
        # Get intermediate activations and contexts
        act, ctx = self.get_intermediate_activation_and_context(
            inspector.layer, x, target_label
        )
        
        # Move inspector to same device as tensors
        inspector = inspector.to(act.device)
        
        # Get top-k subspaces using the inspector's existing method
        top_k_subspaces, relevance_top_k_subspaces = inspector.get_top_k_subspaces(
            act, ctx, top_k
        )

        # Encode activations and contexts to get subspace representations
        encoded_activation = inspector.encode_activation(act)
        encoded_context = inspector.encode_context(ctx)
        
        # Compute relevance for each subspace
        relevance_subspaces = inspector.compute_subspace_relevance(
            encoded_activation, encoded_context
        )

        # Extract heatmaps for top-k subspaces
        input_subspace_heatmaps = []
        
        for k in range(top_k):
            subspace_idx = top_k_subspaces[k]
            
            # Get the relevance for this specific subspace
            subspace_relevance = relevance_subspaces[:, subspace_idx:subspace_idx+1, :, :]
            
            # Convert back to input space using gradient computation
            subspace_heatmap = self._backproject_to_input(
                x, subspace_relevance, target_label
            )
            
            # Sum over channels if multi-channel
            if len(subspace_heatmap.shape) > 1:
                subspace_heatmap = subspace_heatmap.sum(axis=0)
                
            input_subspace_heatmaps.append(subspace_heatmap)

        # Compute residue (total relevance - top-k subspaces relevance)
        total_relevance = relevance_subspaces.sum(dim=1, keepdim=True)
        top_k_relevance = relevance_subspaces[:, top_k_subspaces, :, :].sum(dim=1, keepdim=True)
        residue_relevance = total_relevance - top_k_relevance
        
        # Convert residue to input space
        residue_heatmap = self._backproject_to_input(
            x, residue_relevance, target_label
        )
        
        # Sum over channels if multi-channel
        if len(residue_heatmap.shape) > 1:
            residue_heatmap = residue_heatmap.sum(axis=0)
        
        # Create inspection info object
        inspection_info = InspectionRelevanceInfo(
            top_k_sources=np.array(top_k_subspaces),
            input_top_k_source_heatmaps=np.stack(input_subspace_heatmaps),
            input_subspace_residue_heatmap=residue_heatmap
        )
        
        return logits, original_heatmap, inspection_info


    def _backproject_to_input(self, x, layer_relevance, target_label):
        """
        Backproject layer relevance to input space using gradients.
        
        Args:
            x: Input tensor
            layer_relevance: Relevance at layer level
            target_label: Target class
            
        Returns:
            Input space heatmap
        """
        # Ensure x requires gradients
        x_copy = x.clone().detach().requires_grad_(True)
        
        # Forward pass to get the layer activation
        activation = None
        def hook_fn(module, input, output):
            nonlocal activation
            activation = output
        
        layer_module = self._get_layer_by_name(self.inspector.layer if hasattr(self, 'inspector') else 'block4')
        handle = layer_module.register_forward_hook(hook_fn)

        try:
            # Forward pass
            logits = self.model(x_copy)
            
            if activation is not None:
                # Create a scalar from layer_relevance for backprop
                relevance_scalar = (activation * layer_relevance).sum()
                
                # Compute gradients
                input_grad = torch.autograd.grad(
                    outputs=relevance_scalar,
                    inputs=x_copy,
                    create_graph=False,
                    retain_graph=False
                )[0]
                
                return input_grad.detach().cpu().numpy().squeeze()
            else:
                # Fallback: use simple interpolation
                return self._simple_interpolate_to_input(layer_relevance, x.shape[-1])
                
        finally:
            handle.remove()


    def _simple_interpolate_to_input(self, layer_relevance, target_length):
        """
        Simple interpolation fallback for converting layer relevance to input space.
        """
        # Convert to numpy and squeeze
        relevance_np = layer_relevance.detach().cpu().numpy().squeeze()
        
        # If 1D, just interpolate
        if len(relevance_np.shape) == 1:
            return np.interp(
                np.linspace(0, len(relevance_np)-1, target_length),
                np.arange(len(relevance_np)),
                relevance_np
            )
        else:
            # For multi-dimensional, sum over spatial dimensions first
            relevance_1d = relevance_np.sum(axis=tuple(range(len(relevance_np.shape)-1)))
            return np.interp(
                np.linspace(0, len(relevance_1d)-1, target_length),
                np.arange(len(relevance_1d)),
                relevance_1d
            )
    

    def get_intermediate_activation_and_context(self, layer, x, label=None):
        """
        Extract intermediate activations and context (gradients) for a specific layer.
        This is needed for the notebooks to work with PRCA/DRSA.
        
        Args:
            layer: String name of the layer (e.g., "block1", "block2")
            x: Input tensor
            label: Target class
            
        Returns:
            tuple: (activation_tensor, context_tensor)
        """
        # Ensure input is tensor and has batch dimension
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        
        if x.dim() == 2:  # (C, L) -> (1, C, L)
            x = x.unsqueeze(0)
        
        x = x.to(self.device)
        x.requires_grad_(True)
        
        # Storage for activations
        activation = None
        
        def hook_fn(module, input, output):
            nonlocal activation
            activation = output.clone()
            activation.requires_grad_(True)  # WICHTIG: Ensure gradient computation
        
        # Register hook on the specified layer
        layer_module = self._get_layer_by_name(layer)
        handle = layer_module.register_forward_hook(hook_fn)
        
        try:
            # Forward pass
            with torch.enable_grad():
                logits = self.model(x)

                if activation is not None:
                    print(f"DEBUG: activation shape: {activation.shape}")  # Debug hinzufügen

                    # WICHTIG: Für 1D-Netze von 3D zu 4D konvertieren
                    if len(activation.shape) == 3:  # (N, C, L) -> (N, C, L, 1)
                        activation = activation.unsqueeze(-1)
                        print(f"DEBUG: reshaped activation to: {activation.shape}")

                    # Get target class
                    if label is None:
                        label = torch.argmax(logits, dim=1)
                    elif isinstance(label, int):
                        label = torch.tensor([label]).to(self.device)
                    
                    # Compute gradients w.r.t. the activation
                    target_logit = logits[0, label]
                    
                    if activation is not None:
                        context = torch.autograd.grad(
                            outputs=target_logit,
                            inputs=activation,
                            create_graph=False,
                            retain_graph=False,
                            allow_unused=True  # FIX: Allow unused tensors
                        )[0]
                        
                        if context is None:
                            # Fallback: create zero gradient if unused
                            context = torch.zeros_like(activation)

                        # Auch context zu 4D machen
                        if len(context.shape) == 3:  # (N, C, L) -> (N, C, L, 1)
                            context = context.unsqueeze(-1)
                            print(f"DEBUG: reshaped context to: {context.shape}")

                    else:
                        raise ValueError(f"Could not capture activation for layer '{layer}'")
        
        finally:
            handle.remove()
        
        return activation.detach(), context.detach()
    

    def _get_layer_by_name(self, layer_name):
        """
        Get layer module by name from the model.
        
        Args:
            layer_name: String name of layer
            
        Returns:
            torch.nn.Module: The layer module
        """

        # XResNet1D hat eine komplexere Struktur
        # Zuerst: Model-Struktur ausgeben für Debugging
        if not hasattr(self, '_printed_structure'):
            print("Model structure:")
            for name, module in self.model.named_modules():
                print(f"  {name}: {type(module).__name__}")
            self._printed_structure = True

        # Verbesserte Layer-Navigation
        if hasattr(self.model, layer_name):
            return getattr(self.model, layer_name)
        
        # Fallback: Suche in allen named_modules
        for name, module in self.model.named_modules():
            if layer_name in name or name.endswith(layer_name):
                return module
            
        # Original Mapping als letzter Fallback
        layer_mapping = {
            "stem1": "0",      # First stem block
            "stem2": "1",      # Second stem block  
            "stem3": "2",      # Third stem block
            "maxpool": "3",    # MaxPool layer
            "block1": "4",     # First residual block
            "block2": "5",     # Second residual block
            "block3": "6",     # Third residual block
            "block4": "7",     # Fourth residual block
            "head": "8"        # Classification head
        }
        
        # Get the actual layer name
        actual_layer_name = layer_mapping.get(layer_name, layer_name)
        
        try:
            if actual_layer_name.isdigit():
                return self.model[int(actual_layer_name)]
            
            parts = actual_layer_name.split('.')
            module = self.model
            
            for part in parts:
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)
            
            return module
        except (IndexError, AttributeError, KeyError) as e:
            raise ValueError(f"Layer '{layer_name}' not found in model. Available layers: {list(dict(self.model.named_modules()).keys())}")