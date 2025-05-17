import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_results(results, param_name, param_values, eval_interval, env_name, 
                         metric_name='Average Return', 
                         show_std=True,
                         show_grid=True,
                         show_zero_line=True,
                         figsize=(5, 4),
                         save_path=None,
                         subsample=1):
    """
    Plot training results for different parameter values.
    
    Args:
        results: Dictionary containing results for each parameter value
        param_name: Name of the parameter being varied (e.g., 'Clip Ratio', 'Learning Rate')
        param_values: List of parameter values used
        eval_interval: Interval between evaluations
        env_name: Name of the environment
        metric_name: Name of the metric being plotted (default: 'Average Evaluation Reward')
        show_std: Whether to show standard deviation (default: True)
        show_grid: Whether to show grid lines (default: True)
        show_zero_line: Whether to show horizontal line at y=0 (default: True)
        figsize: Figure size tuple (default: (10, 6))
        save_path: Path to save the plot (default: None)
        subsample: Plot every nth point (default: 1, meaning plot all points)
    """
    # Create x-axis values (steps)
    try:
        x = np.arange(0, len(results[param_values[0]][0]) * eval_interval, eval_interval)
    except:
        x = np.arange(0, len(results[str(param_values[0])][0]) * eval_interval, eval_interval)
    
    # Apply subsampling to x values
    x = x[::subsample]
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Use seaborn's default color palette
    colors = sns.color_palette(n_colors=len(param_values))
    
    # Plot mean and std for each parameter value
    for i, param in enumerate(param_values):
        # Convert list of lists to numpy array
        rewards_array = np.array(results[param])
        mean_rewards = np.mean(rewards_array, axis=0)
        std_rewards = np.std(rewards_array, axis=0)
        
        # Apply subsampling to rewards
        mean_rewards = mean_rewards[::subsample]
        std_rewards = std_rewards[::subsample]
        
        # Plot mean with std as shaded area
        plt.plot(x, mean_rewards, 
                label=f'{param_name} = {param}', 
                linewidth=2,
                color=colors[i])
        
        if show_std:
            plt.fill_between(x, 
                           mean_rewards - std_rewards, 
                           mean_rewards + std_rewards, 
                           alpha=0.2,
                           color=colors[i])
    
    # Customize the plot
    plt.xlabel('Steps', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.title(f'Different {param_name}s', fontsize=14)
    
    if show_grid:
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.legend(fontsize=10, loc='lower right')
    plt.tight_layout()
    
    if show_zero_line:
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 50)
    for param in param_values:
        rewards_array = np.array(results[param])
        final_rewards = rewards_array[:, -1]  # Get final rewards for each seed
        print(f"{param_name} = {param}:")
        print(f"  Final Mean {metric_name}: {np.mean(final_rewards):.2f} ± {np.std(final_rewards):.2f}")
        print(f"  Best Final {metric_name}: {np.max(final_rewards):.2f}")
        print(f"  Worst Final {metric_name}: {np.min(final_rewards):.2f}")
        print(f"  Median Final {metric_name}: {np.median(final_rewards):.2f}")
        print("-" * 50)

# Example usage:
"""
# For clip ratios
clip_ratios = [0.1, 0.25, 0.5]
results = {c: [] for c in clip_ratios}
# ... training code ...
plot_training_results(results, 'Clip Ratio', clip_ratios, eval_interval, env_name)

# For learning rates
learning_rates = [0.0001, 0.001, 0.01]
results = {lr: [] for lr in learning_rates}
# ... training code ...
plot_training_results(results, 'Learning Rate', learning_rates, eval_interval, env_name,
                     metric_name='Average Return',
                     show_std=True,
                     save_path='learning_rate_comparison.png')

# For hidden sizes
hidden_sizes = [32, 64, 128]
results = {hs: [] for hs in hidden_sizes}
# ... training code ...
plot_training_results(results, 'Hidden Size', hidden_sizes, eval_interval, env_name,
                     show_grid=False,
                     show_zero_line=False)
"""