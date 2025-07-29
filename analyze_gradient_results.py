import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd


def load_tensorboard_data(log_dir):
    """Load gradient flow data from tensorboard logs"""
    
    # Find event file
    event_files = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if 'events.out.tfevents' in file:
                event_files.append(os.path.join(root, file))
    
    if not event_files:
        print(f"No tensorboard event files found in {log_dir}")
        return None
    
    # Use the most recent event file
    event_file = sorted(event_files)[-1]
    print(f"Loading data from: {event_file}")
    
    # Load data
    ea = EventAccumulator(event_file)
    ea.Reload()
    
    # Extract gradient flow data
    gradient_data = {}
    tags = ea.Tags()['scalars']
    
    for tag in tags:
        if 'gradient_flow' in tag:
            scalar_events = ea.Scalars(tag)
            steps = [event.step for event in scalar_events]
            values = [event.value for event in scalar_events]
            gradient_data[tag] = {'steps': steps, 'values': values}
    
    return gradient_data


def create_gradient_flow_summary(gradient_data, save_path="gradient_flow_summary.png"):
    """Create a comprehensive summary visualization"""
    
    if not gradient_data:
        print("No gradient data to visualize")
        return
    
    # Prepare data
    modules = ['image_encoder', 'text_encoder', 'cross_modal', 'vqa_head']
    colors = ['blue', 'green', 'orange', 'red']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Plot 1: Mean gradient over time for each module
    ax1 = axes[0]
    for module, color in zip(modules, colors):
        mean_tag = f'gradient_flow/{module}_mean'
        if mean_tag in gradient_data:
            data = gradient_data[mean_tag]
            ax1.plot(data['steps'], data['values'], label=module.replace('_', ' ').title(), 
                    color=color, linewidth=2)
    
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Mean Gradient Norm')
    ax1.set_title('Mean Gradient Evolution During Training')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Relative gradient contribution
    ax2 = axes[1]
    final_gradients = {}
    for module in modules:
        mean_tag = f'gradient_flow/{module}_mean'
        if mean_tag in gradient_data:
            # Get final gradient value
            final_gradients[module] = gradient_data[mean_tag]['values'][-1]
    
    if final_gradients:
        total = sum(final_gradients.values())
        percentages = [final_gradients.get(m, 0) / total * 100 for m in modules]
        
        bars = ax2.bar(range(len(modules)), percentages, color=colors, alpha=0.7)
        ax2.set_xticks(range(len(modules)))
        ax2.set_xticklabels([m.replace('_', ' ').title() for m in modules], rotation=45, ha='right')
        ax2.set_ylabel('Gradient Contribution (%)')
        ax2.set_title('Final Gradient Distribution Across Modules')
        
        # Add percentage labels on bars
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax2.annotate(f'{pct:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    # Plot 3: Max gradient over time
    ax3 = axes[2]
    for module, color in zip(modules, colors):
        max_tag = f'gradient_flow/{module}_max'
        if max_tag in gradient_data:
            data = gradient_data[max_tag]
            ax3.plot(data['steps'], data['values'], label=module.replace('_', ' ').title(), 
                    color=color, linewidth=2, linestyle='--')
    
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Max Gradient Norm')
    ax3.set_title('Maximum Gradient Evolution During Training')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Gradient ratio (text+cross_modal+vqa vs image)
    ax4 = axes[3]
    if all(f'gradient_flow/{m}_mean' in gradient_data for m in modules):
        steps = gradient_data['gradient_flow/image_encoder_mean']['steps']
        
        image_grads = np.array(gradient_data['gradient_flow/image_encoder_mean']['values'])
        text_grads = np.array(gradient_data['gradient_flow/text_encoder_mean']['values'])
        cross_grads = np.array(gradient_data['gradient_flow/cross_modal_mean']['values'])
        vqa_grads = np.array(gradient_data['gradient_flow/vqa_head_mean']['values'])
        
        # Compute ratio
        text_pathway_grads = text_grads + cross_grads + vqa_grads
        ratio = text_pathway_grads / (image_grads + 1e-8)  # Avoid division by zero
        
        ax4.plot(steps, ratio, color='purple', linewidth=2)
        ax4.axhline(y=1, color='gray', linestyle=':', alpha=0.5, label='Equal contribution')
        ax4.set_xlabel('Training Steps')
        ax4.set_ylabel('Gradient Ratio')
        ax4.set_title('Gradient Ratio: (Text + Cross-Modal + VQA Head) / Image Encoder')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Summary visualization saved to {save_path}")
    plt.show()


def print_gradient_analysis(gradient_data):
    """Print detailed analysis of gradient flow"""
    
    print("\n" + "="*60)
    print("GRADIENT FLOW ANALYSIS SUMMARY")
    print("="*60)
    
    if not gradient_data:
        print("No gradient data available")
        return
    
    modules = ['image_encoder', 'text_encoder', 'cross_modal', 'vqa_head']
    
    # Calculate statistics for each module
    for module in modules:
        mean_tag = f'gradient_flow/{module}_mean'
        max_tag = f'gradient_flow/{module}_max'
        
        if mean_tag in gradient_data:
            mean_values = gradient_data[mean_tag]['values']
            
            print(f"\n{module.upper().replace('_', ' ')}:")
            print(f"  Initial gradient: {mean_values[0]:.6f}")
            print(f"  Final gradient: {mean_values[-1]:.6f}")
            print(f"  Average gradient: {np.mean(mean_values):.6f}")
            print(f"  Gradient change: {(mean_values[-1] / mean_values[0] - 1) * 100:.2f}%")
            
            if max_tag in gradient_data:
                max_values = gradient_data[max_tag]['values']
                print(f"  Peak gradient: {np.max(max_values):.6f}")
    
    # Calculate gradient contribution
    print("\n" + "-"*60)
    print("GRADIENT CONTRIBUTION (Final Step):")
    print("-"*60)
    
    final_gradients = {}
    for module in modules:
        mean_tag = f'gradient_flow/{module}_mean'
        if mean_tag in gradient_data:
            final_gradients[module] = gradient_data[mean_tag]['values'][-1]
    
    if final_gradients:
        total = sum(final_gradients.values())
        
        image_grad = final_gradients.get('image_encoder', 0)
        text_pathway_grad = (final_gradients.get('text_encoder', 0) + 
                           final_gradients.get('cross_modal', 0) + 
                           final_gradients.get('vqa_head', 0))
        
        print(f"\nImage Encoder contribution: {image_grad/total*100:.2f}%")
        print(f"Text Pathway contribution: {text_pathway_grad/total*100:.2f}%")
        print(f"  - Text Encoder: {final_gradients.get('text_encoder', 0)/total*100:.2f}%")
        print(f"  - Cross-Modal: {final_gradients.get('cross_modal', 0)/total*100:.2f}%")
        print(f"  - VQA Head: {final_gradients.get('vqa_head', 0)/total*100:.2f}%")
        
        print(f"\nGradient Ratio (Text Pathway / Image): {text_pathway_grad/image_grad:.2f}x")
    
    print("\n" + "="*60)
    print("CONCLUSION:")
    print("="*60)
    
    if final_gradients and text_pathway_grad/image_grad > 2:
        print("✓ Gradient flow analysis confirms that VQA loss primarily")
        print("  propagates through the text encoder and fusion layers,")
        print("  with minimal impact on the image encoder (CLIP).")
        print(f"✓ Text pathway receives {text_pathway_grad/image_grad:.1f}x more gradient than image encoder.")
    else:
        print("⚠ Gradient flow shows more balanced distribution.")
        print("  This might indicate the model is still adapting or")
        print("  the task requires significant visual understanding.")
    
    print("="*60 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze gradient flow results')
    parser.add_argument('--log_dir', type=str, default='gradient_flow_demo',
                       help='Directory containing tensorboard logs')
    parser.add_argument('--output_dir', type=str, default='gradient_analysis',
                       help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading gradient data from {args.log_dir}...")
    gradient_data = load_tensorboard_data(args.log_dir)
    
    if gradient_data:
        # Print analysis
        print_gradient_analysis(gradient_data)
        
        # Create visualizations
        summary_path = os.path.join(args.output_dir, 'gradient_flow_summary.png')
        create_gradient_flow_summary(gradient_data, summary_path)
        
        print(f"\nAnalysis complete. Results saved to {args.output_dir}")
    else:
        print("Failed to load gradient data.")


if __name__ == "__main__":
    main() 