import json
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_encoder_comparison():
    """Phân tích và so sánh kết quả từ các encoder experiments riêng biệt"""
    
    print("🔍 Searching for encoder experiment results...")
    
    # Tìm tất cả thư mục kết quả encoder experiments
    encoder_dirs = {}
    
    # Tìm các thư mục với pattern encoder_comparison_* hoặc cosine_analysis_*
    all_dirs = [d for d in os.listdir('.') if os.path.isdir(d)]
    
    for directory in all_dirs:
        # Check for individual encoder results
        for encoder_type in ['text', 'image', 'cross']:
            summary_file = os.path.join(directory, f'cosine_summary_{encoder_type}.json')
            step_file = os.path.join(directory, f'cosine_step_results_{encoder_type}.json')
            
            if os.path.exists(summary_file) and os.path.exists(step_file):
                if encoder_type not in encoder_dirs:
                    encoder_dirs[encoder_type] = []
                encoder_dirs[encoder_type].append({
                    'dir': directory,
                    'summary_file': summary_file,
                    'step_file': step_file
                })
    
    if not encoder_dirs:
        print("❌ Không tìm thấy kết quả encoder experiments!")
        print("💡 Hãy chạy gradient_analysis.py trước để tạo dữ liệu")
        return
    
    print(f"📁 Found experiments for encoders: {list(encoder_dirs.keys())}")
    
    # Load data từ experiment mới nhất của mỗi encoder
    encoder_results = {}
    
    for encoder_type, experiments in encoder_dirs.items():
        # Sắp xếp theo thời gian, lấy mới nhất
        experiments.sort(key=lambda x: x['dir'], reverse=True)
        latest_exp = experiments[0]
        
        print(f"📊 Loading {encoder_type} encoder data from: {latest_exp['dir']}")
        
        with open(latest_exp['summary_file'], 'r') as f:
            summary = json.load(f)
        
        with open(latest_exp['step_file'], 'r') as f:
            step_data = json.load(f)
        
        encoder_results[encoder_type] = {
            'summary': summary,
            'steps': step_data,
            'dir': latest_exp['dir']
        }
    
    print(f"\n🔬 === ENCODER COMPARISON ANALYSIS ===")
    
    # Create comprehensive comparison plots
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Encoder Performance Comparison', fontsize=16, fontweight='bold')
    
    colors = {'text': 'blue', 'image': 'green', 'cross': 'red'}
    encoder_names = {'text': 'Text Encoder', 'image': 'Image Encoder', 'cross': 'Cross-Modal'}
    
    # Plot 1: Final cosine comparison for target encoders
    ax = axes[0, 0]
    encoders = []
    final_cosines = []
    colors_list = []
    
    for encoder_type, data in encoder_results.items():
        summary = data['summary']
        target_encoder_key = f'{encoder_type}_encoder'
        
        if target_encoder_key in summary:
            encoders.append(encoder_names[encoder_type])
            final_cosines.append(summary[target_encoder_key]['final_cosine'])
            colors_list.append(colors[encoder_type])
    
    bars = ax.bar(encoders, final_cosines, color=colors_list, alpha=0.7)
    ax.set_ylabel('Final Cosine Similarity')
    ax.set_title('Final Cosine Similarity - Target Encoders')
    ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='Good Threshold')
    ax.axhline(y=0.0, color='red', linestyle='--', alpha=0.5, label='Zero Line')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, final_cosines):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Mean cosine comparison for target encoders
    ax = axes[0, 1]
    mean_cosines = []
    
    for encoder_type, data in encoder_results.items():
        summary = data['summary']
        target_encoder_key = f'{encoder_type}_encoder'
        
        if target_encoder_key in summary:
            mean_cosines.append(summary[target_encoder_key]['mean_cosine'])
    
    bars = ax.bar(encoders, mean_cosines, color=colors_list, alpha=0.7)
    ax.set_ylabel('Mean Cosine Similarity')
    ax.set_title('Mean Cosine Similarity - Target Encoders')
    ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7)
    ax.axhline(y=0.0, color='red', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    for bar, value in zip(bars, mean_cosines):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: VQA Head performance comparison
    ax = axes[0, 2]
    vqa_final_cosines = []
    vqa_mean_cosines = []
    vqa_encoders = []
    
    for encoder_type, data in encoder_results.items():
        summary = data['summary']
        
        if 'vqa_head' in summary:
            vqa_final_cosines.append(summary['vqa_head']['final_cosine'])
            vqa_mean_cosines.append(summary['vqa_head']['mean_cosine'])
            vqa_encoders.append(encoder_names[encoder_type])
    
    if vqa_encoders:  # Only plot if we have VQA data
        x_vqa = np.arange(len(vqa_encoders))
        width = 0.35
        
        bars1 = ax.bar(x_vqa - width/2, vqa_final_cosines, width, label='Final', alpha=0.7)
        bars2 = ax.bar(x_vqa + width/2, vqa_mean_cosines, width, label='Mean', alpha=0.7)
        
        ax.set_xticks(x_vqa)
        ax.set_xticklabels(vqa_encoders)
    
        ax.set_ylabel('VQA Head Cosine Similarity')
        ax.set_title('VQA Head Performance Across Experiments')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No VQA Head Data', ha='center', va='center', transform=ax.transAxes)
    
    # Plot 4-6: Individual encoder trends
    for i, (encoder_type, data) in enumerate(encoder_results.items()):
        ax = axes[1, i]
        
        steps_data = data['steps']
        steps = [step['step'] for step in steps_data if 'step' in step]
        
        # Target encoder trend
        target_key = f'{encoder_type}_avg_cosine'
        target_cosines = [step.get(target_key, 0) for step in steps_data]
        
        # VQA head trend
        vqa_cosines = [step.get('vqa_avg_cosine', 0) for step in steps_data]
        
        if steps and target_cosines:
            ax.plot(steps, target_cosines, label=f'{encoder_names[encoder_type]}', 
                   color=colors[encoder_type], linewidth=2, alpha=0.8)
        
        if steps and vqa_cosines:
            ax.plot(steps, vqa_cosines, label='VQA Head', 
                   color='purple', linewidth=2, alpha=0.6, linestyle='--')
        
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.axhline(y=0.3, color='green', linestyle='--', alpha=0.5)
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title(f'{encoder_names[encoder_type]} Experiment')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 7: Combined trend comparison (smoothed)
    ax = axes[2, 0]
    
    for encoder_type, data in encoder_results.items():
        steps_data = data['steps']
        steps = [step['step'] for step in steps_data if 'step' in step]
        target_key = f'{encoder_type}_avg_cosine'
        target_cosines = [step.get(target_key, 0) for step in steps_data]
        
        if len(steps) > 20:
            # Apply smoothing
            window = min(20, len(steps) // 5)
            smoothed = np.convolve(target_cosines, np.ones(window)/window, mode='valid')
            smoothed_steps = steps[window-1:]
            
            ax.plot(smoothed_steps, smoothed, label=encoder_names[encoder_type], 
                   color=colors[encoder_type], linewidth=3, alpha=0.8)
    
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=0.3, color='green', linestyle='--', alpha=0.5, label='Good Threshold')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Smoothed Cosine Similarity')
    ax.set_title('Encoder Performance Comparison (Smoothed)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 8: Convergence rate comparison
    ax = axes[2, 1]
    
    convergence_rates = []
    for encoder_type, data in encoder_results.items():
        steps_data = data['steps']
        target_key = f'{encoder_type}_avg_cosine'
        target_cosines = [step.get(target_key, 0) for step in steps_data]
        
        if len(target_cosines) > 50:
            # Calculate trend in last 50% of training
            mid_point = len(target_cosines) // 2
            trend = np.polyfit(range(mid_point, len(target_cosines)), 
                             target_cosines[mid_point:], 1)[0]
            convergence_rates.append(trend)
        else:
            convergence_rates.append(0)
    
    bars = ax.bar(encoders, convergence_rates, color=colors_list, alpha=0.7)
    ax.set_ylabel('Convergence Rate (slope)')
    ax.set_title('Convergence Rate Comparison')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    for bar, value in zip(bars, convergence_rates):
        ax.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height() + (0.0001 if value >= 0 else -0.0002), 
                f'{value:.4f}', ha='center', 
                va='bottom' if value >= 0 else 'top', fontweight='bold')
    
    # Plot 9: Stability comparison (standard deviation)
    ax = axes[2, 2]
    
    stability_scores = []
    for encoder_type, data in encoder_results.items():
        summary = data['summary']
        target_encoder_key = f'{encoder_type}_encoder'
        
        if target_encoder_key in summary:
            std_cosine = summary[target_encoder_key]['std_cosine']
            # Lower std = higher stability
            stability_score = 1 / (1 + std_cosine)  # Normalize to 0-1
            stability_scores.append(stability_score)
    
    bars = ax.bar(encoders, stability_scores, color=colors_list, alpha=0.7)
    ax.set_ylabel('Stability Score (1/(1+std))')
    ax.set_title('Training Stability Comparison')
    ax.grid(True, alpha=0.3)
    
    for bar, value in zip(bars, stability_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save comparison plot
    comparison_dir = "encoder_comparison_analysis"
    os.makedirs(comparison_dir, exist_ok=True)
    
    plot_path = os.path.join(comparison_dir, 'encoder_comparison_detailed.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📈 Detailed comparison plots saved to: {plot_path}")
    
    plt.show()
    
    # Print detailed comparison summary
    print(f"\n📊 DETAILED ENCODER COMPARISON SUMMARY:")
    print(f"{'='*80}")
    
    print(f"{'Encoder':<15} {'Final':<8} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8} {'VQA Final':<10}")
    print(f"{'-'*80}")
    
    best_encoder = None
    best_score = -1
    
    for encoder_type, data in encoder_results.items():
        summary = data['summary']
        target_encoder_key = f'{encoder_type}_encoder'
        
        if target_encoder_key in summary:
            stats = summary[target_encoder_key]
            vqa_stats = summary.get('vqa_head', {})
            
            final_cosine = stats['final_cosine']
            mean_cosine = stats['mean_cosine']
            std_cosine = stats['std_cosine']
            min_cosine = stats['min_cosine']
            max_cosine = stats['max_cosine']
            vqa_final = vqa_stats.get('final_cosine', 0)
            
            # Calculate combined score (higher is better)
            combined_score = (final_cosine + mean_cosine + vqa_final) / 3 - std_cosine/2
            
            print(f"{encoder_names[encoder_type]:<15} {final_cosine:<8.4f} {mean_cosine:<8.4f} {std_cosine:<8.4f} {min_cosine:<8.4f} {max_cosine:<8.4f} {vqa_final:<10.4f}")
            
            if combined_score > best_score:
                best_score = combined_score
                best_encoder = encoder_type
    
    print(f"\n🏆 PERFORMANCE RANKING:")
    
    # Calculate performance scores for ranking
    performance_scores = {}
    for encoder_type, data in encoder_results.items():
        summary = data['summary']
        target_encoder_key = f'{encoder_type}_encoder'
        
        if target_encoder_key in summary:
            stats = summary[target_encoder_key]
            vqa_stats = summary.get('vqa_head', {})
            
            # Multi-criteria scoring
            final_score = stats['final_cosine']
            mean_score = stats['mean_cosine']
            stability_score = 1 / (1 + stats['std_cosine'])
            vqa_score = vqa_stats.get('final_cosine', 0)
            
            # Weighted combined score
            combined_score = (final_score * 0.3 + mean_score * 0.3 + 
                            stability_score * 0.2 + vqa_score * 0.2)
            
            performance_scores[encoder_type] = combined_score
    
    # Sort by performance
    ranked_encoders = sorted(performance_scores.items(), key=lambda x: x[1], reverse=True)
    
    for i, (encoder_type, score) in enumerate(ranked_encoders):
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}."
        print(f"  {medal} {encoder_names[encoder_type]}: {score:.4f}")
    
    print(f"\n💡 ACTIONABLE INSIGHTS:")
    
    if best_encoder:
        print(f"  🎯 BEST PERFORMER: {encoder_names[best_encoder]}")
        
        best_data = encoder_results[best_encoder]
        best_summary = best_data['summary']
        best_target_key = f'{best_encoder}_encoder'
        
        if best_target_key in best_summary:
            best_final = best_summary[best_target_key]['final_cosine']
            
            if best_final > 0.5:
                print(f"  ✅ Excellent convergence! Consider using {best_encoder} encoder as primary")
            elif best_final > 0.3:
                print(f"  📈 Good performance. {best_encoder} encoder shows most promise")
            else:
                print(f"  ⚠️ Best performer still needs improvement. Focus on {best_encoder} encoder")
    
    # Specific recommendations for each encoder
    print(f"\n🔧 ENCODER-SPECIFIC RECOMMENDATIONS:")
    
    for encoder_type, data in encoder_results.items():
        summary = data['summary']
        target_encoder_key = f'{encoder_type}_encoder'
        
        if target_encoder_key in summary:
            stats = summary[target_encoder_key]
            final_cosine = stats['final_cosine']
            mean_cosine = stats['mean_cosine']
            std_cosine = stats['std_cosine']
            
            print(f"\n  {encoder_names[encoder_type]}:")
            
            if final_cosine < 0.1:
                print(f"    🚨 CRITICAL: Reduce LR by 70%, add gradient clipping")
            elif final_cosine < 0.3:
                print(f"    ⚠️ MODERATE: Reduce LR by 30%, increase warmup")
            else:
                print(f"    ✅ GOOD: Maintain current settings")
            
            if std_cosine > 0.3:
                print(f"    📊 High variance detected: Add gradient clipping, increase batch size")
            
            if mean_cosine < 0:
                print(f"    🔄 Negative trend: Reduce LR significantly, check data quality")
    
    # Save summary to file
    summary_path = os.path.join(comparison_dir, 'encoder_comparison_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'performance_scores': performance_scores,
            'ranking': [(enc, score) for enc, score in ranked_encoders],
            'best_encoder': best_encoder,
            'analysis_timestamp': str(np.datetime64('now'))
        }, f, indent=2)
    
    print(f"\n📁 Summary saved to: {summary_path}")
    print(f"📁 Results from: {[data['dir'] for data in encoder_results.values()]}")


if __name__ == "__main__":
    analyze_encoder_comparison() 