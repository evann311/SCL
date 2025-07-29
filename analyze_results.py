import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def analyze_encoder_comparison():
    """Analyze and compare results from individual encoder experiments."""
    
    # Find all encoder experiment directories
    base_path = Path('.')
    experiment_dirs = list(base_path.glob('encoder_comparison_*'))
    
    if not experiment_dirs:
        # Fallback to old naming
        experiment_dirs = list(base_path.glob('gradient_analysis_vqa_vast_*'))
    
    if not experiment_dirs:
        print("❌ No experiment directories found!")
        return
    
    print(f"🔍 Found {len(experiment_dirs)} experiment directories")
    
    # Find the most recent experiment that has ALL three encoders
    complete_experiments = []
    
    for exp_dir in experiment_dirs:
        print(f"📁 Checking: {exp_dir}")
        
        # Check if this experiment has all three encoder types
        has_all_encoders = True
        encoders_found = []
        
        for encoder_type in ['text', 'image', 'cross']:
            summary_file = exp_dir / f'cosine_summary_{encoder_type}.json'
            step_file = exp_dir / f'cosine_step_results_{encoder_type}.json'
            
            if summary_file.exists() and step_file.exists():
                encoders_found.append(encoder_type)
            else:
                has_all_encoders = False
        
        if has_all_encoders:
            complete_experiments.append(exp_dir)
            print(f"  ✅ Complete experiment with all 3 encoders: {encoders_found}")
        else:
            print(f"  ⚠️  Incomplete experiment, only has: {encoders_found}")
    
    if not complete_experiments:
        print("❌ No complete experiments found with all 3 encoders!")
        return
    
    # Get the most recent complete experiment
    latest_experiment = sorted(complete_experiments, reverse=True)[0]
    print(f"\n🎯 Using LATEST COMPLETE experiment: {latest_experiment}")
    
    # Load data from the latest complete experiment
    encoder_results = {}
    
    for encoder_type in ['text', 'image', 'cross']:
        summary_file = latest_experiment / f'cosine_summary_{encoder_type}.json'
        step_file = latest_experiment / f'cosine_step_results_{encoder_type}.json'
        
        print(f"📊 Loading {encoder_type} encoder from: {latest_experiment}")
        
        try:
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            with open(step_file, 'r') as f:
                step_data = json.load(f)
            
            encoder_results[encoder_type] = {
                'summary': summary,
                'step_data': step_data,
                'experiment_dir': latest_experiment
            }
            print(f"  ✅ Successfully loaded {encoder_type} encoder data")
        except Exception as e:
            print(f"  ❌ Error loading {encoder_type}: {e}")
    
    if len(encoder_results) != 3:
        print("❌ Failed to load all 3 encoders!")
        return
    
    print(f"\n🎯 Successfully loaded all 3 encoders from: {latest_experiment}")
    
    # Create the single plot
    plt.figure(figsize=(12, 6))
    
    colors = {'text': 'blue', 'image': 'green', 'cross': 'red'}
    
    for encoder_type, data in encoder_results.items():
        step_data = data['step_data']
        
        # Extract cosine similarity data
        steps = []
        cosines = []
        
        for step_info in step_data:
            if 'step' in step_info and 'cosine_similarity' in step_info:
                steps.append(step_info['step'])
                cosines.append(step_info['cosine_similarity'])
        
        if cosines:  # Only print if we have data
            print(f"📈 {encoder_type.upper()}: {len(steps)} data points, range: {min(cosines):.3f} to {max(cosines):.3f}")
        else:
            print(f"⚠️  {encoder_type.upper()}: No valid cosine data found")
        
        if steps and cosines:
            # Apply smoothing
            window = min(50, len(cosines) // 10) if len(cosines) > 10 else 1
            if window > 1:
                smoothed = np.convolve(cosines, np.ones(window)/window, mode='valid')
                smoothed_steps = steps[window-1:]
            else:
                smoothed = cosines
                smoothed_steps = steps
            
            # Plot smoothed line
            plt.plot(smoothed_steps, smoothed, 
                    color=colors.get(encoder_type, 'black'), 
                    linewidth=2, 
                    label=f'{encoder_type.title()} Encoder')
    
    # Add good threshold line
    plt.axhline(y=0.3, color='gray', linestyle='--', alpha=0.5, label='Good Threshold')
    plt.axhline(y=0.0, color='red', linestyle='--', alpha=0.3, label='Zero Line')
    
    plt.xlabel('Training Steps')
    plt.ylabel('Smoothed Cosine Similarity')
    plt.title('Encoder Performance Comparison (Smoothed)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    output_path = 'encoder_comparison_smoothed.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Plot saved to: {output_path}")
    
    # Show plot
    plt.show()
    
    # Print summary
    print(f"\n📋 LOADED FROM SINGLE EXPERIMENT:")
    print(f"  📁 Directory: {latest_experiment}")
    for encoder_type, data in encoder_results.items():
        step_count = len(data['step_data'])
        print(f"  {encoder_type.upper()}: {step_count} steps")

if __name__ == "__main__":
    analyze_encoder_comparison() 