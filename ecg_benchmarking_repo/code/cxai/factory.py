import torch
import torchvision
import timm
import os

import torch.nn as nn

# DUMMY-KLASSE für LayerInspectionContext (falls benötigt)
class LayerInspectionContext:
    def __init__(self, *args, **kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

# Import NACH der Dummy-Klasse Definition
from cxai import models, explainers
from cxai import utils as putils


AVAILABLE_ARCHITECTURES = [
    "dm_nfnet_f0",
    "torchvision-vgg16-imagenet", 
    "netdissect-vgg16-imagenet",
    "xresnet1d101-ecg"
]


def make_model(arch: str, class_names=None, scaler=None):
    """
    Create a model with the specified architecture.
    
    Args:
        arch (str): Architecture name
        experiment (str): ECG experiment type ('exp0', 'exp1.1', 'exp1.1.1') 
        class_names (list): List of class names from your notebook
        
    Returns:
        torch.nn.Module: Model
        Tuple(transform, transform): 1) resize and cropping; 2) resize, cropping and normalization
    """
    
    assert arch in AVAILABLE_ARCHITECTURES, f"Architecture '{arch}' not in {AVAILABLE_ARCHITECTURES}"
    
    if "dm_nfnet" in arch:
        model = models.nfnet.get_model(arch)
        (rc_transform, input_transform) = models.nfnet.get_transformation(model)
        
    elif "vgg16" in arch:
        model, (rc_transform, input_transform) = models.vgg16.get_model(arch)
        
    elif "xresnet1d101-ecg" in arch:
        from cxai.models.xresnet1d import xresnet1d101, get_transformation
        model = xresnet1d101(input_channels=12, num_classes=len(class_names))
        setattr(model, '_ecg_class_names', class_names)

        # Verwende übergebenen Scaler oder versuche aus __main__
        if scaler is None:
            import __main__
            scaler = getattr(__main__, 'standard_scaler', None)

        # apply standardizer
        rc_transform, input_transform = get_transformation(model, scaler)
        
    else:
        raise ValueError(f"Architecture '{arch}' not supported.")
    
    # Transformationen am Modell speichern
    setattr(model, models.ATTRIBUTE_TRANSFORMATION, (rc_transform, input_transform))
    
    return model, (rc_transform, input_transform)


def make_explainer(name:str, model:torch.nn.Module):
    """Create an explainer for the given model."""
    if "lrp" in name.lower():
        #return explainers.lrp.LRPExplainer(model)
        raise ValueError("LRP not implemented for ECG models yet.")
    
    elif "gradient" in name.lower():
        return explainers.gradient.GradientExplainer(model)
    
    elif 'gradcam' in name.lower():
        return explainers.gradcam.GradCAMExplainer(model, target_layer='7.2')

    raise ValueError(f"Explainer '{name}' not available for {type(model)}!")


def make_label_desc(dataset: str, model=None):
    """Create a label description function."""
    if dataset == "imagenet":
        return lambda label: putils.imagenet.ix_to_classname[label]
    elif dataset == "ecg":
        if model is not None and hasattr(model, '_ecg_class_names') and model._ecg_class_names is not None:
            class_names = model._ecg_class_names
            return lambda label: class_names[label] if 0 <= label < len(class_names) else f"ECG_Class_{label}"
        else:
            return lambda label: f"ECG_Class_{label}"
    else:
        raise ValueError(f"Dataset '{dataset}' not supported.")