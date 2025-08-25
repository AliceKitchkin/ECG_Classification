import torch
import numpy as np


class GradCAMExplainer:
    """
    Grad-CAM Explainer für 1D-Modelle (z.B. ECG).
    """

    def __init__(self, model, target_layer=None, device="cpu"):
        """
        Args:
            model: Das zu erklärende PyTorch-Modell
            target_layer: Name der Schicht, für die Grad-CAM berechnet werden soll (z.B. 'block4')
            device: 'cpu' oder 'cuda'
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.target_layer = target_layer


    def _register_hooks(self):
        """
        Registriert Forward- und Backward-Hooks auf der Zielschicht.
        Speichert Aktivierungen und Gradienten als Attribute.
        """
        self.activations = None
        self.gradients = None

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # grad_output ist ein Tuple; [0] ist der Gradient der Aktivierung
            self.gradients = grad_output[0].detach()

        # Hole das Layer-Modul (wie in gradient.py)
        layer = self._get_layer_by_name(self.target_layer)
        self.fwd_handle = layer.register_forward_hook(forward_hook)
        self.bwd_handle = layer.register_full_backward_hook(backward_hook)


    def _get_layer_by_name(self, layer_name):
        """
        Findet das Layer-Modul im Modell anhand des Namens.
        """
        # Direkter Zugriff, falls Attribut vorhanden
        if hasattr(self.model, layer_name):
            return getattr(self.model, layer_name)
        # Suche in named_modules
        for name, module in self.model.named_modules():
            if layer_name in name or name.endswith(layer_name):
                return module
        raise ValueError(f"Layer '{layer_name}' nicht gefunden.")
    

    def explain(self, x, label=None):
        """
        Berechnet die Grad-CAM-Heatmap für eine Eingabe x und eine Zielklasse.
        Args:
            x: Eingabetensor, Form (C, L) oder (1, C, L)
            label: Zielklasse (int). Wenn None, wird die vorhergesagte Klasse genommen.
        Returns:
            logits: Modell-Output
            heatmap: Grad-CAM-Heatmap als numpy-Array (Länge wie Input)
        """
        # Input vorbereiten
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        x.requires_grad_(True)

        # Hooks setzen
        self._register_hooks()

        # Forward
        logits = self.model(x)
        if label is None:
            label = torch.argmax(logits, dim=1).item()
        elif isinstance(label, torch.Tensor):
            label = label.item()

        # Backward: Gradienten der Zielklasse
        self.model.zero_grad()
        target = logits[0, label]
        target.backward()

        print("Activations min/max/mean:", self.activations.min(), self.activations.max(), self.activations.mean())
        print("Gradients min/max/mean:", self.gradients.min(), self.gradients.max(), self.gradients.mean())

        # Grad-CAM: Weights berechnen (global average pooling über L)
        # Annahme: activations und gradients Form (1, C, L)
        weights = self.gradients.mean(dim=2, keepdim=True)  # (1, C, 1)
        cam = (weights * self.activations).sum(dim=1)  # (1, L)
        cam = torch.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()

        # Optional: Normalisieren
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # Hooks entfernen
        self.fwd_handle.remove()
        self.bwd_handle.remove()

        return logits.detach().cpu().numpy(), cam
    

    def get_intermediate_activation_and_context(self, layer, x, label=None):
        """
        Gibt die Aktivierungen (und Kontext, hier identisch) eines Layers für Input x zurück.
        Ein Kontext-Vektor wie bei LRP ist für Grad-CAM nicht nötig - du kannst ihn einfach
        identisch zu den Aktivierungen zurückgeben, damit die Downstream-Pipeline funktioniert.
        Args:
            layer: Name des Layers (string)
            x: Input-Tensor (1, C, L)
            label: Zielklasse (optional, für Kompatibilität)
        Returns:
            activations: Tensor [1, C, L] oder [1, C, H, W]
            context: identisch zu activations (für Kompatibilität)
        """
        # Layer setzen
        self.target_layer = layer
        # Hooks setzen
        self._register_hooks()
        # Forward
        _ = self.model(x)
        # Hooks entfernen
        self.fwd_handle.remove()
        self.bwd_handle.remove()
        # Rückgabe: activations und context (identisch)
        return self.activations.unsqueeze(0), self.activations.unsqueeze(0)