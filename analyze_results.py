import json
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_cosine_results():
    """Phân tích kết quả cosine analysis từ file JSON đã lưu"""
    
    # Tìm thư mục kết quả mới nhất
    results_dirs = [d for d in os.listdir('.') if d.startswith('cosine_analysis_vqa_vast_')]
    if not results_dirs:
        # Fallback to gradient_analysis directories
        results_dirs = [d for d in os.listdir('.') if d.startswith('gradient_analysis_vqa_vast_')]
        if not results_dirs:
            print("❌ Không tìm thấy kết quả analysis!")
            return
    
    # Sắp xếp theo thời gian (mới nhất đầu tiên)
    results_dirs.sort(reverse=True)
    latest_dir = results_dirs[0]
    
    print(f"📁 Analyzing results from: {latest_dir}")
    
    # Load dữ liệu cosine
    cosine_summary_file = os.path.join(latest_dir, 'cosine_summary.json')
    cosine_step_file = os.path.join(latest_dir, 'cosine_step_results.json')
    
    if not os.path.exists(cosine_summary_file):
        print(f"❌ File not found: {cosine_summary_file}")
        return
    
    if not os.path.exists(cosine_step_file):
        print(f"❌ File not found: {cosine_step_file}")
        return
    
    with open(cosine_summary_file, 'r') as f:
        summary = json.load(f)
    
    with open(cosine_step_file, 'r') as f:
        step_data = json.load(f)
    
    print(f"📊 Loaded cosine analysis data")
    
    # Phân tích cosine similarity
    print(f"\n🔬 === COSINE SIMILARITY ANALYSIS RESULTS ===")
    
    # Summary statistics
    for encoder_type in ['text_encoder', 'image_encoder', 'cross_modal']:
        if encoder_type in summary:
            stats = summary[encoder_type]
            print(f"\n📊 {encoder_type.replace('_', ' ').title()}:")
            print(f"  Mean Cosine: {stats['mean_cosine']:.4f} ± {stats['std_cosine']:.4f}")
            print(f"  Final Cosine: {stats['final_cosine']:.4f}")
            print(f"  Range: [{stats['min_cosine']:.4f}, {stats['max_cosine']:.4f}]")
            print(f"  Measurements: {stats['total_measurements']}")
            
            # Interpret cosine values
            mean_cosine = stats['mean_cosine']
            final_cosine = stats['final_cosine']
            
            print(f"  📈 Analysis:")
            if mean_cosine > 0.5:
                print(f"    ✅ GOOD: High consistency, stable learning")
            elif mean_cosine > 0.2:
                print(f"    ⚠️ MODERATE: Some consistency, gradual learning")
            elif mean_cosine > 0.0:
                print(f"    ❌ LOW: Weak consistency, unstable learning")
            else:
                print(f"    🚨 CRITICAL: Negative consistency, diverging!")
            
            if final_cosine > 0.7:
                print(f"    ✅ Converged well")
            elif final_cosine > 0.3:
                print(f"    ⚠️ Still converging")
            elif final_cosine > 0.0:
                print(f"    ❌ Poor convergence")
            else:
                print(f"    🚨 Diverging at end!")
    
    # Comparative analysis
    print(f"\n⚖️ COMPARATIVE ANALYSIS:")
    
    text_stats = summary.get('text_encoder', {})
    image_stats = summary.get('image_encoder', {})
    cross_stats = summary.get('cross_modal', {})
    
    if text_stats and image_stats:
        text_cosine = text_stats['mean_cosine']
        image_cosine = image_stats['mean_cosine']
        
        if abs(text_cosine - image_cosine) < 0.1:
            print(f"  ✅ Text and Image encoders have similar consistency")
        elif text_cosine > image_cosine:
            ratio = text_cosine / (image_cosine + 1e-8)
            print(f"  ⚠️ Text encoder more stable (ratio: {ratio:.2f}x)")
            if ratio > 3:
                print(f"    → Consider reducing text LR or increasing image LR")
        else:
            ratio = image_cosine / (text_cosine + 1e-8)
            print(f"  ⚠️ Image encoder more stable (ratio: {ratio:.2f}x)")
            if ratio > 3:
                print(f"    → Consider reducing image LR or increasing text LR")
    
    # Trend analysis
    print(f"\n📈 TREND ANALYSIS:")
    
    # Extract trends from step data
    text_trends = []
    image_trends = []
    cross_trends = []
    steps = []
    
    for step_info in step_data:
        if 'step' in step_info:
            steps.append(step_info['step'])
            text_trends.append(step_info.get('text_avg_cosine', 0))
            image_trends.append(step_info.get('image_avg_cosine', 0))
            cross_trends.append(step_info.get('cross_avg_cosine', 0))
    
    if len(steps) > 10:
        # Calculate trends
        text_trend = np.polyfit(steps[-100:], text_trends[-100:], 1)[0] if len(steps) >= 100 else np.polyfit(steps, text_trends, 1)[0]
        image_trend = np.polyfit(steps[-100:], image_trends[-100:], 1)[0] if len(steps) >= 100 else np.polyfit(steps, image_trends, 1)[0]
        cross_trend = np.polyfit(steps[-100:], cross_trends[-100:], 1)[0] if len(steps) >= 100 else np.polyfit(steps, cross_trends, 1)[0]
        
        print(f"  Text Encoder Trend: {'📈 Improving' if text_trend > 0.0001 else '📉 Declining' if text_trend < -0.0001 else '➡️ Stable'}")
        print(f"  Image Encoder Trend: {'📈 Improving' if image_trend > 0.0001 else '📉 Declining' if image_trend < -0.0001 else '➡️ Stable'}")
        print(f"  Cross-Modal Trend: {'📈 Improving' if cross_trend > 0.0001 else '📉 Declining' if cross_trend < -0.0001 else '➡️ Stable'}")
    
    # Plot trends
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Cosine similarity over time
    plt.subplot(2, 2, 1)
    if steps and text_trends:
        plt.plot(steps, text_trends, label='Text Encoder', alpha=0.7, linewidth=2)
    if steps and image_trends:
        plt.plot(steps, image_trends, label='Image Encoder', alpha=0.7, linewidth=2)
    if steps and cross_trends:
        plt.plot(steps, cross_trends, label='Cross-Modal', alpha=0.7, linewidth=2)
    
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Zero Line')
    plt.axhline(y=0.3, color='green', linestyle='--', alpha=0.5, label='Good Threshold')
    plt.xlabel('Training Steps')
    plt.ylabel('Average Cosine Similarity')
    plt.title('Cosine Similarity Trends')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Distribution of cosine values
    plt.subplot(2, 2, 2)
    all_text_cosines = [stats['mean_cosine']] if 'text_encoder' in summary else []
    all_image_cosines = [stats['mean_cosine']] if 'image_encoder' in summary else []
    all_cross_cosines = [stats['mean_cosine']] if 'cross_modal' in summary else []
    
    if text_trends:
        plt.hist(text_trends, bins=30, alpha=0.7, label='Text Encoder', density=True)
    if image_trends:
        plt.hist(image_trends, bins=30, alpha=0.7, label='Image Encoder', density=True)
    if cross_trends:
        plt.hist(cross_trends, bins=30, alpha=0.7, label='Cross-Modal', density=True)
    
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    plt.axvline(x=0.3, color='green', linestyle='--', alpha=0.5)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.title('Cosine Similarity Distribution')
    plt.legend()
    
    # Plot 3: Final vs Initial comparison
    plt.subplot(2, 2, 3)
    encoders = []
    initial_cosines = []
    final_cosines = []
    
    for encoder_type in ['text_encoder', 'image_encoder', 'cross_modal']:
        if encoder_type in summary:
            encoders.append(encoder_type.replace('_', ' ').title())
            # Get initial cosine (average of first 10 steps)
            encoder_steps = [step for step in step_data[:10] if f'{encoder_type.split("_")[0]}_avg_cosine' in step]
            if encoder_steps:
                initial_avg = np.mean([step[f'{encoder_type.split("_")[0]}_avg_cosine'] for step in encoder_steps])
                initial_cosines.append(initial_avg)
            else:
                initial_cosines.append(0)
            final_cosines.append(summary[encoder_type]['final_cosine'])
    
    if encoders:
        x = np.arange(len(encoders))
        width = 0.35
        
        plt.bar(x - width/2, initial_cosines, width, label='Initial (first 10 steps)', alpha=0.7)
        plt.bar(x + width/2, final_cosines, width, label='Final', alpha=0.7)
        
        plt.xlabel('Encoders')
        plt.ylabel('Cosine Similarity')
        plt.title('Initial vs Final Cosine Similarity')
        plt.xticks(x, encoders, rotation=45)
        plt.legend()
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Plot 4: Convergence progress
    plt.subplot(2, 2, 4)
    if len(steps) > 50:
        # Calculate rolling average
        window = min(50, len(steps) // 10)
        
        if text_trends:
            text_rolling = np.convolve(text_trends, np.ones(window)/window, mode='valid')
            plt.plot(steps[window-1:], text_rolling, label='Text (smoothed)', linewidth=2)
        
        if image_trends:
            image_rolling = np.convolve(image_trends, np.ones(window)/window, mode='valid')
            plt.plot(steps[window-1:], image_rolling, label='Image (smoothed)', linewidth=2)
        
        if cross_trends:
            cross_rolling = np.convolve(cross_trends, np.ones(window)/window, mode='valid')
            plt.plot(steps[window-1:], cross_rolling, label='Cross-Modal (smoothed)', linewidth=2)
    
    plt.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Target Convergence')
    plt.xlabel('Training Steps')
    plt.ylabel('Smoothed Cosine Similarity')
    plt.title('Convergence Progress (Smoothed)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(latest_dir, 'cosine_analysis_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📈 Analysis plots saved to: {plot_path}")
    
    plt.show()
    
    # Recommendations
    print(f"\n💡 ACTIONABLE RECOMMENDATIONS:")
    
    overall_performance = "good"
    
    for encoder_type in ['text_encoder', 'image_encoder', 'cross_modal']:
        if encoder_type in summary:
            stats = summary[encoder_type]
            mean_cosine = stats['mean_cosine']
            final_cosine = stats['final_cosine']
            
            if mean_cosine < 0.1 or final_cosine < 0.1:
                overall_performance = "poor"
                break
            elif mean_cosine < 0.3 or final_cosine < 0.3:
                overall_performance = "moderate"
    
    if overall_performance == "poor":
        print(f"  🚨 CRITICAL ISSUES DETECTED:")
        print(f"     • Reduce learning rate by 50%")
        print(f"     • Add gradient clipping (max_norm=1.0)")
        print(f"     • Increase warmup steps to 30% of training")
        print(f"     • Consider smaller batch size")
    elif overall_performance == "moderate":
        print(f"  ⚠️ MODERATE ISSUES:")
        print(f"     • Reduce learning rate by 20%")
        print(f"     • Add gradient clipping (max_norm=2.0)")
        print(f"     • Increase warmup steps to 20% of training")
    else:
        print(f"  ✅ TRAINING APPEARS STABLE:")
        print(f"     • Continue with current settings")
        print(f"     • Monitor for any degradation")
    
    print(f"\n📋 GENERAL RECOMMENDATIONS:")
    print(f"     • Target cosine similarity > 0.3 for stable learning")
    print(f"     • Monitor for negative cosine values (indicates divergence)")
    print(f"     • Aim for final cosine > 0.5 for good convergence")


if __name__ == "__main__":
    analyze_cosine_results() 