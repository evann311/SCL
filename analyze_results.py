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
    
    # Sort all experiments by timestamp (newest first)
    sorted_experiments = sorted(experiment_dirs, reverse=True)
    
    # Find the 3 most recent experiments (each should have one encoder)
    encoder_results = {}
    used_experiments = []
    
    for exp_dir in sorted_experiments:
        if len(encoder_results) >= 3:  # We found all 3 encoders
            break
            
        print(f"📁 Checking: {exp_dir}")
        
        # Check which encoder type this experiment contains
        for encoder_type in ['text', 'image', 'cross']:
            if encoder_type in encoder_results:  # Already found this encoder
                continue
                
            summary_file = exp_dir / f'cosine_summary_{encoder_type}.json'
            step_file = exp_dir / f'cosine_step_results_{encoder_type}.json'
            
            if summary_file.exists() and step_file.exists():
                print(f"  ✅ Found {encoder_type} encoder - loading...")
                
                try:
                    with open(summary_file, 'r') as f:
                        summary = json.load(f)
                    with open(step_file, 'r') as f:
                        step_data = json.load(f)
                    
                    encoder_results[encoder_type] = {
                        'summary': summary,
                        'step_data': step_data,
                        'experiment_dir': exp_dir
                    }
                    used_experiments.append(exp_dir)
                    print(f"  ✅ Successfully loaded {encoder_type} encoder from {exp_dir}")
                    break  # Found encoder for this experiment, move to next experiment
                    
                except Exception as e:
                    print(f"  ❌ Error loading {encoder_type}: {e}")
    
    if len(encoder_results) == 0:
        print("❌ No encoder results found!")
        return
    
    print(f"\n🎯 Loaded {len(encoder_results)} encoders from 3 most recent experiments:")
    for encoder_type, data in encoder_results.items():
        print(f"  {encoder_type.upper()}: {data['experiment_dir']}")
    
    # Create the single plot
    plt.figure(figsize=(12, 8))
    
    markers = {'text': 'o', 'image': 's', 'cross': '^'}
    
    for encoder_type, data in encoder_results.items():
        step_data = data['step_data']
        
        # Extract cosine similarity data based on gradient_analysis.py format
        steps = []
        cosines = []
        
        # Look for the correct key based on target encoder
        cosine_key = f'{encoder_type}_avg_cosine'
        
        for step_info in step_data:
            if 'step' in step_info and cosine_key in step_info:
                steps.append(step_info['step'])
                cosines.append(step_info[cosine_key])
        
        # If no target encoder cosine, try to find any available cosine data
        if not cosines:
            print(f"🔍 No {cosine_key} found, checking available keys...")
            if step_data:
                available_keys = list(step_data[0].keys())
                print(f"   Available keys: {available_keys}")
                
                # Try different possible keys
                possible_keys = [
                    f'{encoder_type}_avg_cosine',
                    f'{encoder_type}_cosine',
                    'cosine_similarity',
                    'avg_cosine'
                ]
                
                for key in possible_keys:
                    if key in available_keys:
                        for step_info in step_data:
                            if 'step' in step_info and key in step_info:
                                steps.append(step_info['step'])
                                cosines.append(step_info[key])
                        if cosines:
                            print(f"   ✅ Using key: {key}")
                            break
        
        if cosines:  # Only print if we have data
            print(f"📈 {encoder_type.upper()}: {len(steps)} data points, range: {min(cosines):.3f} to {max(cosines):.3f}")
        else:
            print(f"⚠️  {encoder_type.upper()}: No valid cosine data found")
            continue
        
        if steps and cosines:
            # Apply smoothing
            window = min(50, len(cosines) // 10) if len(cosines) > 10 else 1
            if window > 1:
                smoothed = np.convolve(cosines, np.ones(window)/window, mode='valid')
                smoothed_steps = steps[window-1:]
            else:
                smoothed = cosines
                smoothed_steps = steps
            
            # Plot smoothed line with markers
            plt.plot(smoothed_steps, smoothed, 
                    marker=markers.get(encoder_type, 'o'), 
                    color='darkblue',
                    linewidth=2, 
                    markersize=6,
                    markevery=10,
                    label=f'{encoder_type.title()} Encoder')
            
            print(f"✅ Plotted {encoder_type} encoder with {len(smoothed)} smoothed points")
    
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
    print(f"\n📋 LOADED FROM 3 RECENT EXPERIMENTS:")
    for encoder_type, data in encoder_results.items():
        exp_dir = data['experiment_dir']
        step_count = len(data['step_data'])
        print(f"  {encoder_type.upper()}: {exp_dir} ({step_count} steps)")

if __name__ == "__main__":
    analyze_encoder_comparison() 