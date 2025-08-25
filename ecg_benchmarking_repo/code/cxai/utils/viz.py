from matplotlib import pyplot as plt
import numpy as np

import torch
from matplotlib.colors import ListedColormap

from . import imagenet


class FigContext:
    def __init__(self, save_path=None, **kwargs):
        self.kwargs = kwargs
        self.save_path = save_path

    def __enter__(self):
        self.fig = plt.figure(**self.kwargs)

        return self.fig

    def __exit__(self, *args):
        if self.save_path is not None:
            self.fig.savefig(self.save_path)

        plt.close(self.fig)


def imshow(img):
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])


def heatmap(
    heatmap,
    title="",
    logit=None,
    reference_heatmap=None,
    total_score=None,
    grid_steps=-1,
    fontsize=None,
):
    """Plot heatmap; this is adapted from https://git.tu-berlin.de/gmontavon/lrp-tutorial/-/blob/main/utils.py

    Args:
        heatmap np.array(h, w):
        reference_heatmap (np.array(h, w), optional): used for calculating normalization values. Defaults to None.
        total_score (float, optional): used for normalizing scores. Defaults to None.
    """
    assert len(heatmap.shape) == 2

    if reference_heatmap is None:
        reference_heatmap = heatmap

    assert len(reference_heatmap.shape) == 2

    b = np.abs(reference_heatmap).max()

    my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
    my_cmap[:, 0:3] *= 0.85
    my_cmap = ListedColormap(my_cmap)

    sum_Ri = np.sum(heatmap)
    if total_score is None:
        percent = 100
        txt = r"$\sum_i R_i=%.2f$" % (sum_Ri)
    else:
        percent = (sum_Ri / total_score) * 100
        txt = r"$\sum_i R_i=%.4f$ (%3.1f%%)" % (sum_Ri, percent)

    plt.axis("on")
    plt.xticks([])
    plt.yticks([])
    plt.imshow(heatmap, cmap=my_cmap, vmin=-b, vmax=b)

    h = heatmap.shape[0]
    if grid_steps > 0:
        _grid(h, grid_steps)

    plt.title(f"{title}", fontsize=fontsize)
    if logit is not None:
        plt.xlabel(f"{txt}, logit={logit:.4f}", fontsize=fontsize)
    else:
        plt.xlabel(f"{txt}", fontsize=fontsize)


# this grid provides reference locations for further inspection.
def _grid(total_width, steps=4):
    step_size = total_width / steps
    for i in range(1, steps):
        if i % 2 == 0:
            ls = "-"
            alpha = 0.5
        else:
            ls = "--"
            alpha = 0.1
        lw = 1
        plt.axvline(i * step_size, lw=lw, color="black", ls=ls, alpha=alpha)
        plt.axhline(i * step_size, lw=lw, color="black", ls=ls, alpha=alpha)

        plt.axis("on")
        plt.xticks([])
        plt.yticks([])


def plot_ecg_signal(ecg_data, title="ECG Signal", lead_names=None, figsize=(15, 8), 
                   sampling_rate=100, time_unit="s"):
    """
    Plot 12-lead ECG signal.
    
    Args:
        ecg_data: ECG data, shape (12, length) or (length, 12)
        title: Plot title
        lead_names: List of lead names (default: ['I', 'II', ..., 'V6'])
        figsize: Figure size
        sampling_rate: Sampling rate in Hz
        time_unit: Time unit ('s' for seconds, 'ms' for milliseconds)
    """
    # Ensure correct shape (12, length)
    if ecg_data.shape[0] != 12:
        if ecg_data.shape[1] == 12:
            ecg_data = ecg_data.T
        else:
            raise ValueError(f"ECG data must have 12 leads, got shape {ecg_data.shape}")
    
    if lead_names is None:
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    # Create time axis
    length = ecg_data.shape[1]
    if time_unit == "s":
        time_axis = np.arange(length) / sampling_rate
        time_label = "Time (s)"
    else:
        time_axis = np.arange(length) * 1000 / sampling_rate
        time_label = "Time (ms)"
    
    fig, axes = plt.subplots(4, 3, figsize=figsize, sharex=True)
    axes = axes.flatten()
    
    for i, (ax, lead_name) in enumerate(zip(axes, lead_names)):
        ax.plot(time_axis, ecg_data[i], 'b-', linewidth=1)
        ax.set_title(f'Lead {lead_name}', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('Amplitude (mV)')
        
        if i >= 9:  # Bottom row
            ax.set_xlabel(time_label)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_ecg_with_attribution(ecg_data, attribution_map, title="ECG with Attribution", 
                             lead_names=None, figsize=(15, 10), sampling_rate=100,
                             alpha=0.6, cmap='RdYlBu_r'):
    """
    Plot ECG signal with attribution overlay.
    
    Args:
        ecg_data: ECG data, shape (12, length) or (length, 12)
        attribution_map: Attribution values, same shape as ecg_data
        title: Plot title
        lead_names: List of lead names
        figsize: Figure size
        sampling_rate: Sampling rate in Hz
        alpha: Transparency of attribution overlay
        cmap: Colormap for attribution
    """
    # Ensure correct shapes
    if ecg_data.shape[0] != 12:
        if ecg_data.shape[1] == 12:
            ecg_data = ecg_data.T
            attribution_map = attribution_map.T
        else:
            raise ValueError(f"ECG data must have 12 leads, got shape {ecg_data.shape}")
    
    if lead_names is None:
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    # Create time axis
    length = ecg_data.shape[1]
    time_axis = np.arange(length) / sampling_rate
    
    # Normalize attribution for visualization
    attr_norm = np.abs(attribution_map)
    
    fig, axes = plt.subplots(4, 3, figsize=figsize, sharex=True)
    axes = axes.flatten()
    
    for i, (ax, lead_name) in enumerate(zip(axes, lead_names)):
        # Plot ECG signal
        ax.plot(time_axis, ecg_data[i], 'k-', linewidth=1.5, label='ECG Signal')
        
        # Add attribution as colored background
        # Normalize attribution for this lead
        if attr_norm[i].max() > 0:
            attr_lead_norm = (attr_norm[i] - attr_norm[i].min()) / (attr_norm[i].max() - attr_norm[i].min())
        else:
            attr_lead_norm = attr_norm[i]
        
        # Create threshold for highlighting important regions
        threshold = 0.5
        high_attr_regions = attr_lead_norm > threshold
        
        # Color the background based on attribution strength
        if np.any(high_attr_regions):
            # Get y-limits for background coloring
            y_min, y_max = ecg_data[i].min(), ecg_data[i].max()
            y_range = y_max - y_min
            y_bottom = y_min - 0.1 * y_range
            y_top = y_max + 0.1 * y_range
            
            # Fill regions with high attribution
            for j in range(len(time_axis)):
                if high_attr_regions[j]:
                    ax.axvspan(time_axis[j], time_axis[j] + (time_axis[1] - time_axis[0]) if j < len(time_axis)-1 else time_axis[j], 
                              ymin=0, ymax=1, alpha=alpha * attr_lead_norm[j], color='red')
        
        # Alternative: Plot attribution as a secondary line
        ax2 = ax.twinx()
        ax2.plot(time_axis, attribution_map[i], 'r-', alpha=0.7, linewidth=1, label='Attribution')
        ax2.set_ylabel('Attribution', color='r', fontsize=8)
        ax2.tick_params(axis='y', labelcolor='r', labelsize=8)
        
        ax.set_title(f'Lead {lead_name}', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('Amplitude (mV)', fontsize=8)
        ax.tick_params(axis='y', labelsize=8)
        
        if i >= 9:  # Bottom row
            ax.set_xlabel('Time (s)')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Alternative: Simplified version without complex overlay
def plot_ecg_with_attribution_simple(ecg_data, attribution_map, title="ECG with Attribution", 
                                    lead_names=None, figsize=(15, 10), sampling_rate=100):
    """
    Simplified ECG attribution plot - shows ECG and attribution separately.
    """
    # Debug output
    print(f"Debug - ECG shape: {ecg_data.shape}, Attribution shape: {attribution_map.shape}")
    
    # Handle different attribution shapes - RICHTIG für numpy und torch
    if isinstance(attribution_map, torch.Tensor):
        if attribution_map.dim() == 3:  # (1, channels, length)
            attribution_map = attribution_map.squeeze(0)  # Remove batch dimension
        attribution_map = attribution_map.detach().cpu().numpy()
    elif isinstance(attribution_map, np.ndarray):
        if len(attribution_map.shape) == 3:  # (1, channels, length)
            attribution_map = attribution_map.squeeze(0)  # Remove batch dimension
    
    # Ensure correct shapes - ECG should be (length, channels) = (1000, 12)
    if ecg_data.shape[1] == 12 and ecg_data.shape[0] == 1000:
        # Already correct: (1000, 12)
        pass
    elif ecg_data.shape[0] == 12 and ecg_data.shape[1] == 1000:
        # Need to transpose: (12, 1000) -> (1000, 12)
        ecg_data = ecg_data.T
    else:
        raise ValueError(f"Unexpected ECG shape: {ecg_data.shape}")
    
    # Handle attribution shape - should match ECG
    if attribution_map.shape[1] == 12 and attribution_map.shape[0] == 1000:
        # Already correct: (1000, 12)
        pass
    elif attribution_map.shape[0] == 12 and attribution_map.shape[1] == 1000:
        # Need to transpose: (12, 1000) -> (1000, 12)
        attribution_map = attribution_map.T
    elif len(attribution_map.shape) == 1:
        # If attribution is 1D, we need to handle it differently
        print(f"Warning: Attribution is 1D with shape {attribution_map.shape}")
        # Create a dummy attribution for all leads
        attribution_map = np.tile(attribution_map.reshape(-1, 1), (1, 12))
    else:
        raise ValueError(f"Unexpected attribution shape: {attribution_map.shape}")
    
    print(f"After processing - ECG: {ecg_data.shape}, Attribution: {attribution_map.shape}")
    
    if lead_names is None:
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    length = ecg_data.shape[0]  # Should be 1000
    time_axis = np.arange(length) / sampling_rate
    
    fig, axes = plt.subplots(4, 3, figsize=figsize, sharex=True)
    axes = axes.flatten()
    
    for i, (ax, lead_name) in enumerate(zip(axes, lead_names)):
        # Create twin axis for attribution
        ax2 = ax.twinx()
        
        # Plot ECG signal on main axis - use column i
        line1 = ax.plot(time_axis, ecg_data[:, i], 'b-', linewidth=1.5, label='ECG')
        ax.set_ylabel('ECG (mV)', color='b', fontsize=8)
        ax.tick_params(axis='y', labelcolor='b', labelsize=8)
        
        # Plot attribution on secondary axis - use column i
        line2 = ax2.plot(time_axis, attribution_map[:, i], 'r-', alpha=0.7, linewidth=1, label='Attribution')
        ax2.set_ylabel('Attribution', color='r', fontsize=8)
        ax2.tick_params(axis='y', labelcolor='r', labelsize=8)
        
        ax.set_title(f'Lead {lead_name}', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if i >= 9:  # Bottom row
            ax.set_xlabel('Time (s)')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def ecg_heatmap(attribution_map, title="Attribution Heatmap", reference_heatmap=None, 
               lead_names=None, figsize=(12, 8)):
    """
    Plot attribution heatmap for ECG (similar to image heatmap but for 1D signals).
    """
    if lead_names is None:
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    # Handle torch tensors
    if isinstance(attribution_map, torch.Tensor):
        attribution_map = attribution_map.detach().cpu().numpy()
    
    # Ensure correct shape (12, length) für Heatmap
    if attribution_map.shape[0] != 12:
        if attribution_map.shape[1] == 12:
            attribution_map = attribution_map.T
    
    plt.figure(figsize=figsize)
    
    # Use reference for consistent color scaling across plots
    if reference_heatmap is not None:
        if isinstance(reference_heatmap, torch.Tensor):
            reference_heatmap = reference_heatmap.detach().cpu().numpy()
        vmin, vmax = reference_heatmap.min(), reference_heatmap.max()
    else:
        vmin, vmax = attribution_map.min(), attribution_map.max()
    
    # Plot as heatmap
    im = plt.imshow(attribution_map, aspect='auto', cmap='RdYlBu_r', 
                   vmin=vmin, vmax=vmax)
    
    plt.yticks(range(12), lead_names)
    plt.xlabel('Time Steps')
    plt.ylabel('ECG Leads')
    plt.title(title)
    plt.colorbar(im, label='Attribution Strength')
    plt.tight_layout()
    plt.show()

def plot_ecg_comparison(ecg_data, pred_class, true_class, attribution_map=None, 
                       class_names=None, figsize=(15, 8)):
    """
    Plot ECG with prediction vs ground truth comparison.
    """
    if class_names is not None:
        if isinstance(pred_class, int):
            pred_name = class_names[pred_class]
        else:
            pred_name = pred_class
            
        if isinstance(true_class, int):
            true_name = class_names[true_class]
        else:
            true_name = true_class
    else:
        pred_name = str(pred_class)
        true_name = str(true_class)
    
    # Color coding for correct/incorrect prediction
    color = 'green' if pred_name == true_name else 'red'
    title = f"Prediction: {pred_name} | Ground Truth: {true_name}"
    
    if attribution_map is not None:
        plot_ecg_with_attribution_simple(ecg_data, attribution_map, title=title, figsize=figsize)
    else:
        plot_ecg_signal(ecg_data, title=title, figsize=figsize)