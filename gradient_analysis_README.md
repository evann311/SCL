# 🔬 Gradient Analysis for VQA Training

## 📖 Mục đích

Phân tích so sánh gradient giữa **Text Encoder (RoBERTa)** và **Image Encoder (CLIP ViT)** trong quá trình fine-tuning VQA để hiểu tại sao fine-tune CLIP lại tệ hơn so với fine-tune RoBERTa.

## 🎯 Phân tích bao gồm:

### 1. **Gradient Magnitude Analysis**
- **L2 Norm**: Độ lớn gradient của từng layer
- **Layer-wise Distribution**: Phân bố gradient theo layers
- **Temporal Evolution**: Sự thay đổi gradient theo thời gian

### 2. **Gradient Consistency Analysis**
- **Cosine Similarity**: Độ nhất quán gradient giữa các step
- **Gradient Drift**: Mức độ thay đổi hướng gradient
- **Stability Metrics**: Chỉ số ổn định của training

### 3. **Statistical Analysis**
- **Entropy Analysis**: Độ tập trung gradient theo layers
- **Distribution Comparison**: So sánh phân phối gradient
- **Ratio Analysis**: Tỷ lệ gradient giữa các components

---

## 🚀 Cách sử dụng

### Option 1: Quick Analysis (Recommended để test)

```bash
python quick_gradient_test.py
```

**Tính năng:**
- ✅ Phân tích nhanh với 5 batches
- ✅ Hiển thị kết quả real-time
- ✅ Tạo biểu đồ so sánh đơn giản
- ✅ Phù hợp để debug và test nhanh

### Option 2: Full Analysis (Comprehensive)

```bash
python gradient_analysis.py
```

**Tính năng:**
- 📊 Training đầy đủ với logging chi tiết
- 📈 TensorBoard integration
- 📋 Báo cáo markdown tự động
- 🎨 Visualizations chuyên nghiệp
- 💾 Lưu trữ dữ liệu đầy đủ

---

## 📊 Kết quả và Metrics

### 1. **L2 Norm Comparison**
```
Text Encoder L2 Norm:     X.XXXXXX
Image Encoder L2 Norm:    Y.YYYYYY
Text/Image Ratio:         Z.ZZZ
```

### 2. **Layer-wise Analysis**
```
Text Encoder:
  Layer 0: X.XXXXXX
  Layer 1: X.XXXXXX
  ...
  
Image Encoder:
  Layer 0: Y.YYYYYY
  Layer 1: Y.YYYYYY
  ...
```

### 3. **Cosine Similarity (Consistency)**
```
Text Encoder Consistency:    High/Medium/Low (0.XXX)
Image Encoder Consistency:   High/Medium/Low (0.YYY)
```

---

## 📈 Visualization Outputs

### 1. **Gradient Norms per Layer**
- Bar charts comparing gradient magnitudes
- Separate plots for Text vs Image encoders
- Layer-wise breakdown

### 2. **Cosine Similarity Over Time**
- Line plots showing gradient consistency
- Trend analysis for stability assessment
- Drift detection

### 3. **Distribution Analysis**
- Histograms of gradient magnitudes
- Statistical comparisons
- Outlier detection

### 4. **Temporal Evolution**
- Running averages over training steps
- Comparative trends
- Stability metrics

---

## 🔧 Configuration

### Model Setup
```python
# All parameters are unfrozen for comprehensive analysis
model = unfreeze_all_parameters(model)

# Uses vqa_vast config by default
config_name = 'vqa_vast'
```

### Analysis Parameters
```python
# Quick test
batches_to_analyze = 5

# Full analysis
max_steps = 500
log_every_n_steps = 10
val_check_interval = 100
```

---

## 📋 Expected Insights

### 🎯 **Possible Findings:**

1. **Gradient Magnitude Imbalance**
   - Text encoder có gradient lớn hơn → Overtraining text
   - Image encoder có gradient nhỏ hơn → Undertraining image

2. **Gradient Instability**
   - CLIP gradients không ổn định (low cosine similarity)
   - RoBERTa gradients ổn định hơn

3. **Layer-wise Issues**
   - Certain layers của CLIP không được train hiệu quả
   - Gradient vanishing/exploding ở specific layers

4. **Cross-modal Interference**
   - Cross-modal layers ảnh hưởng không đồng đều
   - Text-image alignment issues

---

## 🔍 Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size in config
   'per_gpu_batchsize': 2  # Instead of 32
   ```

2. **No Gradients Found**
   - Check model is in train mode
   - Verify loss.backward() is called
   - Ensure parameters require_grad=True

3. **Import Errors**
   ```bash
   # Make sure you're in the correct environment
   cd /home/hoaithi/SCL
   python -c "import scl.config"
   ```

---

## 📊 Output Files

### Quick Analysis:
- `quick_gradient_analysis.png`: Comparison plots

### Full Analysis:
```
gradient_analysis_vqa_vast_YYYYMMDD_HHMMSS/
├── gradient_statistics.json          # Raw data
├── gradient_entropy.json            # Entropy analysis
├── gradient_analysis_report.md      # Summary report
├── gradient_norms_per_layer.png     # Layer comparison
├── gradient_cosine_similarity.png   # Consistency plots
└── gradient_distributions.png       # Statistical plots
```

### TensorBoard Logs:
```bash
tensorboard --logdir result/gradient_analysis_vqa_vast/
```

---

## 💡 Recommendations Based on Results

### If Text/Image Ratio > 2.0:
- Reduce learning rate for text encoder
- Increase learning rate for image encoder
- Add gradient clipping for text encoder

### If Cosine Similarity < 0.5:
- Increase warmup steps
- Reduce learning rate
- Add gradient accumulation

### If Gradient Entropy is Low:
- Check for gradient vanishing
- Verify layer-wise learning rates
- Consider different initialization

---

## 🔬 Advanced Usage

### Custom Analysis:
```python
# Modify analysis parameters
gradient_callback = GradientAnalysisCallback(
    save_dir="custom_analysis",
    log_every_n_steps=5
)

# Add custom metrics
def custom_gradient_metric(grad):
    return torch.norm(grad, p=1)  # L1 norm instead of L2
```

### Different Configs:
```python
# Test with different configurations
configs_to_test = ['vqa', 'vqa_kg', 'vqa_vast']
for config_name in configs_to_test:
    # Run analysis...
```

---

## ⚠️ Important Notes

1. **Memory Usage**: Full analysis requires significant GPU memory
2. **Time**: Complete analysis may take 1-2 hours
3. **Storage**: Results can be several GB for full analysis
4. **Dependencies**: Requires matplotlib, seaborn, tqdm

---

## 🎯 Expected Outcome

Sau khi chạy analysis, bạn sẽ có:

1. **Quantitative Evidence** về sự khác biệt gradient magnitude
2. **Visual Proof** về gradient instability patterns  
3. **Statistical Analysis** về gradient distribution
4. **Actionable Insights** để improve training strategy

Điều này sẽ giúp bạn hiểu **exactly why** CLIP fine-tuning performs worse và **how to fix it**! 🎯 