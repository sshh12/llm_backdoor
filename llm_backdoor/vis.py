import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def visualize_attention_changes(original_model, trained_model, layer_idx=0):
    """
    Visualizes changes in the attention mechanism components (Q, K, V projections).
    """
    orig_layer = original_model.model.layers[layer_idx].self_attn
    trained_layer = trained_model.model.layers[layer_idx].self_attn

    # Focus on core attention components
    components = {"Query": "q_proj", "Key": "k_proj", "Value": "v_proj"}

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Create a single norm for consistent color scaling across subplots
    all_diffs = []
    for name, param_name in components.items():
        orig_param = getattr(orig_layer, param_name).weight.detach().cpu()
        trained_param = getattr(trained_layer, param_name).weight.detach().cpu()
        diff = (trained_param - orig_param).numpy()
        all_diffs.append(diff)

    vmax = max([np.abs(d).max() for d in all_diffs])
    vmin = -vmax

    for idx, (name, param_name) in enumerate(components.items()):
        # Get parameters
        print(name)
        orig_param = getattr(orig_layer, param_name).weight.detach().cpu()
        trained_param = getattr(trained_layer, param_name).weight.detach().cpu()

        # Calculate difference
        diff = (trained_param - orig_param).numpy()
        print(diff.shape)

        # Create heatmap with consistent color scaling
        im = sns.heatmap(
            diff,
            cmap="RdBu",
            center=0,
            ax=axes[idx],
            xticklabels=False,
            yticklabels=False,
            vmin=vmin,
            vmax=vmax,
            cbar=True if idx == 2 else False,  # Only show colorbar for last plot
        )

        axes[idx].set_title(f"{name}")

    plt.suptitle("Changes in Attention Mechanism", fontsize=14)
    plt.tight_layout()
    plt.show()
