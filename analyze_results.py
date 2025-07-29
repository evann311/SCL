import json
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_gradient_results():
    """Phân tích kết quả gradient analysis từ file JSON đã lưu"""
    
    # Tìm thư mục kết quả mới nhất
    results_dirs = [d for d in os.listdir('.') if d.startswith('gradient_analysis_vqa_vast_')]
    if not results_dirs:
        print("❌ Không tìm thấy kết quả analysis!")
        return
    
    # Sắp xếp theo thời gian (mới nhất đầu tiên)
    results_dirs.sort(reverse=True)
    latest_dir = results_dirs[0]
    
    print(f"📁 Analyzing results from: {latest_dir}")
    
    # Load dữ liệu
    stats_file = os.path.join(latest_dir, 'gradient_statistics.json')
    if not os.path.exists(stats_file):
        print(f"❌ File not found: {stats_file}")
        return
    
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    print(f"📊 Loaded gradient statistics")
    
    # Phân tích text encoder vs image encoder
    print(f"\n🔬 === GRADIENT ANALYSIS RESULTS ===")
    
    # Thu thập L2 norms (loại bỏ inf/nan)
    text_l2_norms = []
    image_l2_norms = []
    
    for metric_name, values in stats.get('text_encoder', {}).items():
        if 'l2_norm' in metric_name and values:
            # Lọc các giá trị hợp lệ
            clean_values = [v for v in values if np.isfinite(v) and v > 0 and v < 1e10]
            text_l2_norms.extend(clean_values)
    
    for metric_name, values in stats.get('image_encoder', {}).items():
        if 'l2_norm' in metric_name and values:
            # Lọc các giá trị hợp lệ
            clean_values = [v for v in values if np.isfinite(v) and v > 0 and v < 1e10]
            image_l2_norms.extend(clean_values)
    
    if text_l2_norms and image_l2_norms:
        # Tính thống kê
        text_mean = np.mean(text_l2_norms)
        text_std = np.std(text_l2_norms)
        text_median = np.median(text_l2_norms)
        
        image_mean = np.mean(image_l2_norms)
        image_std = np.std(image_l2_norms)
        image_median = np.median(image_l2_norms)
        
        ratio = text_mean / image_mean if image_mean > 0 else float('inf')
        
        print(f"\n📊 GRADIENT MAGNITUDE COMPARISON:")
        print(f"  🔤 Text Encoder:")
        print(f"     Mean L2 Norm:   {text_mean:.6f}")
        print(f"     Std L2 Norm:    {text_std:.6f}")
        print(f"     Median L2 Norm: {text_median:.6f}")
        print(f"     Total samples:  {len(text_l2_norms)}")
        
        print(f"  🖼️ Image Encoder:")
        print(f"     Mean L2 Norm:   {image_mean:.6f}")
        print(f"     Std L2 Norm:    {image_std:.6f}")
        print(f"     Median L2 Norm: {image_median:.6f}")
        print(f"     Total samples:  {len(image_l2_norms)}")
        
        print(f"\n⚖️ COMPARISON:")
        print(f"  Text/Image Ratio (Mean):   {ratio:.3f}")
        print(f"  Text/Image Ratio (Median): {text_median/image_median:.3f}")
        
        # Phân tích và đưa ra kết luận
        print(f"\n💡 ANALYSIS & INSIGHTS:")
        if ratio > 2.0:
            print("  ⚠️ TEXT ENCODER GRADIENTS ARE MUCH LARGER!")
            print("     → Text encoder đang được train mạnh hơn image encoder")
            print("     → Có thể gây ra imbalance trong multimodal learning")
            print("     → Đề xuất: Giảm learning rate cho text encoder")
        elif ratio > 1.5:
            print("  📈 Text encoder gradients are moderately larger")
            print("     → Slight bias towards text learning")
        elif ratio < 0.5:
            print("  ⚠️ IMAGE ENCODER GRADIENTS ARE MUCH LARGER!")
            print("     → Image encoder đang được train mạnh hơn text encoder")
            print("     → Đề xuất: Giảm learning rate cho image encoder")
        elif ratio < 0.67:
            print("  📈 Image encoder gradients are moderately larger")
            print("     → Slight bias towards image learning")
        else:
            print("  ✅ GRADIENT MAGNITUDES ARE RELATIVELY BALANCED")
            print("     → Good balance between text and image learning")
        
        # Vẽ biểu đồ so sánh đơn giản
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Box plot comparison
        plt.subplot(1, 3, 1)
        plt.boxplot([text_l2_norms, image_l2_norms], labels=['Text Encoder', 'Image Encoder'])
        plt.ylabel('L2 Norm')
        plt.title('Gradient Magnitude Distribution')
        plt.yscale('log')
        
        # Plot 2: Histogram comparison
        plt.subplot(1, 3, 2)
        plt.hist(text_l2_norms, bins=30, alpha=0.7, label='Text Encoder', color='blue', density=True)
        plt.hist(image_l2_norms, bins=30, alpha=0.7, label='Image Encoder', color='green', density=True)
        plt.xlabel('L2 Norm')
        plt.ylabel('Density')
        plt.title('Gradient Distribution Comparison')
        plt.legend()
        plt.yscale('log')
        
        # Plot 3: Ratio over time
        plt.subplot(1, 3, 3)
        min_len = min(len(text_l2_norms), len(image_l2_norms))
        ratios = []
        for i in range(min_len):
            if image_l2_norms[i] > 0:
                r = text_l2_norms[i] / image_l2_norms[i]
                if np.isfinite(r) and 0.01 < r < 100:  # Reasonable range
                    ratios.append(r)
        
        if ratios:
            plt.plot(ratios)
            plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Equal gradients')
            plt.xlabel('Training Step')
            plt.ylabel('Text/Image Gradient Ratio')
            plt.title('Gradient Balance Over Training')
            plt.legend()
            plt.yscale('log')
        
        plt.tight_layout()
        
        # Lưu biểu đồ
        save_path = os.path.join(latest_dir, 'gradient_analysis_summary.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📈 Summary plots saved to: {save_path}")
        plt.show()
        
    else:
        print("❌ Không tìm thấy dữ liệu gradient hợp lệ!")
    
    # Phân tích cosine similarity (consistency)
    analyze_consistency(stats)
    
    # Đưa ra recommendations
    provide_recommendations(text_l2_norms, image_l2_norms)

def analyze_consistency(stats):
    """Phân tích độ nhất quán gradient"""
    
    print(f"\n🎯 GRADIENT CONSISTENCY ANALYSIS:")
    
    for encoder_type in ['text_encoder', 'image_encoder']:
        cosine_values = []
        
        for metric_name, values in stats.get(encoder_type, {}).items():
            if 'cosine_sim' in metric_name and values:
                # Lọc các giá trị cosine hợp lệ
                clean_values = [v for v in values if np.isfinite(v) and -1 <= v <= 1]
                cosine_values.extend(clean_values)
        
        if cosine_values:
            mean_cosine = np.mean(cosine_values)
            std_cosine = np.std(cosine_values)
            
            encoder_name = "Text Encoder" if encoder_type == 'text_encoder' else "Image Encoder"
            print(f"  {encoder_name}:")
            print(f"    Average Cosine Similarity: {mean_cosine:.4f} ± {std_cosine:.4f}")
            print(f"    Total measurements: {len(cosine_values)}")
            
            if mean_cosine > 0.8:
                print(f"    ✅ HIGH consistency - very stable training")
            elif mean_cosine > 0.5:
                print(f"    📊 MEDIUM consistency - acceptable stability")
            else:
                print(f"    ⚠️ LOW consistency - unstable training detected!")

def provide_recommendations(text_l2_norms, image_l2_norms):
    """Đưa ra recommendations dựa trên analysis"""
    
    if not text_l2_norms or not image_l2_norms:
        return
    
    text_mean = np.mean(text_l2_norms)
    image_mean = np.mean(image_l2_norms)
    ratio = text_mean / image_mean
    
    print(f"\n💡 ACTIONABLE RECOMMENDATIONS:")
    
    if ratio > 2.0:
        print("  🔧 IMMEDIATE ACTIONS NEEDED:")
        print("     1. Reduce text encoder learning rate by 50%")
        print("     2. Consider using different learning rates for text vs image")
        print("     3. Add gradient clipping for text encoder")
        print("     4. Monitor for text overfitting")
        
    elif ratio < 0.5:
        print("  🔧 IMMEDIATE ACTIONS NEEDED:")
        print("     1. Reduce image encoder learning rate by 50%")
        print("     2. Consider using different learning rates for image vs text")
        print("     3. Add gradient clipping for image encoder")
        print("     4. Monitor for image overfitting")
        
    else:
        print("  ✅ CURRENT SETUP LOOKS GOOD:")
        print("     1. Gradient balance is acceptable")
        print("     2. Continue current training strategy")
        print("     3. Monitor performance metrics regularly")
    
    print(f"\n📋 GENERAL RECOMMENDATIONS:")
    print("     • Use layer-wise learning rates if imbalance persists")
    print("     • Consider warmup schedules for unstable gradients")
    print("     • Add gradient accumulation if memory allows")
    print("     • Monitor validation metrics alongside gradient analysis")

if __name__ == '__main__':
    analyze_gradient_results() 