# Hướng dẫn Extract Teacher Logits cho Knowledge Distillation

## Tổng quan

Script này giúp bạn extract logits và question IDs từ teacher model để sử dụng cho knowledge distillation trong VQA task.

## Files được tạo

1. **`_config_vqa_distill`** trong `scl/config.py`: Config mới để extract logits
2. **`extract_teacher_logits.py`**: Script chính để extract logits
3. **`run_extract_logits.py`**: Script runner đơn giản
4. **`EXTRACT_LOGITS_README.md`**: File hướng dẫn này

## Cách sử dụng

### Bước 1: Cấu hình checkpoint path

Trong file `scl/config.py`, tìm `_config_vqa_distill` và điền đường dẫn checkpoint của bạn:

```python
_config_vqa_distill = {
    # ...
    'resume_from': "/path/to/your/teacher/checkpoint.ckpt",  # ← Điền đường dẫn này
    # ...
}
```

### Bước 2: Chạy extraction

#### Cách 1: Sử dụng script runner (Khuyến nghị)

```bash
python run_extract_logits.py --checkpoint /path/to/your/teacher/checkpoint.ckpt
```

Với tùy chọn tùy chỉnh:

```bash
python run_extract_logits.py \
    --checkpoint /path/to/your/teacher/checkpoint.ckpt \
    --output_dir ./my_teacher_outputs \
    --config vqa_distill
```

#### Cách 2: Sử dụng script chính trực tiếp

```bash
python extract_teacher_logits.py \
    --checkpoint /path/to/your/teacher/checkpoint.ckpt \
    --output_dir ./teacher_outputs \
    --config vqa_distill
```

### Bước 3: Kiểm tra kết quả

Sau khi chạy thành công, bạn sẽ có các files trong thư mục output:

```
teacher_outputs/
├── teacher_logits_train.npy      # Logits từ teacher model (shape: [num_samples, num_classes])
├── question_ids_train.json       # Question IDs tương ứng với từng logit
└── extraction_metadata.json      # Metadata về quá trình extraction
```

## Output Files

### 1. `teacher_logits_train.npy`
- **Format**: NumPy array
- **Shape**: `[num_samples, 3129]` (3129 là số classes trong VQA2.0)
- **Content**: Raw logits từ teacher model cho mỗi question

### 2. `question_ids_train.json`
- **Format**: JSON list
- **Content**: Question IDs tương ứng với từng row trong logits array

### 3. `extraction_metadata.json`
- **Format**: JSON object
- **Content**: 
  - Số lượng samples
  - Shape của logits
  - Config được sử dụng
  - Đường dẫn checkpoint
  - Sample questions và answers (10 đầu tiên)

## Ví dụ sử dụng extracted data

```python
import numpy as np
import json

# Load teacher logits
teacher_logits = np.load("teacher_outputs/teacher_logits_train.npy")
print(f"Teacher logits shape: {teacher_logits.shape}")

# Load question IDs
with open("teacher_outputs/question_ids_train.json", 'r') as f:
    question_ids = json.load(f)
print(f"Number of questions: {len(question_ids)}")

# Load metadata
with open("teacher_outputs/extraction_metadata.json", 'r') as f:
    metadata = json.load(f)
print(f"Extraction info: {metadata}")
```

## Lưu ý quan trọng

1. **Model mode**: Script tự động set model về `eval()` mode và sử dụng `torch.no_grad()`
2. **Dataset**: Script extract logits từ **training set** của VQA2.0
3. **Memory**: Logits được lưu trữ trong RAM trước khi save, đảm bảo có đủ memory
4. **Error handling**: Script có xử lý lỗi cơ bản và sẽ skip các batch bị lỗi
5. **Progress**: Script hiển thị progress bar và log mỗi 100 batches

## Troubleshooting

### Lỗi "Checkpoint file not found"
- Kiểm tra đường dẫn checkpoint có đúng không
- Đảm bảo file có extension `.ckpt`

### Lỗi "CUDA out of memory"
- Giảm `per_gpu_batchsize` trong config
- Hoặc chạy trên CPU bằng cách set `num_gpus: 0`

### Lỗi "No logits extracted"
- Kiểm tra data path trong config có đúng không
- Đảm bảo dataset được load thành công
- Kiểm tra model có layer `vqa_classifier` không

## Config customization

Bạn có thể tùy chỉnh config `_config_vqa_distill` trong `scl/config.py`:

```python
_config_vqa_distill = {
    # Paths
    'data_root': '/path/to/your/vqa/data',     # Đường dẫn data VQA
    'roberta_path': '/path/to/roberta-base',   # Đường dẫn RoBERTa
    'vit_path': '/path/to/vit/weights',        # Đường dẫn ViT weights
    
    # Batch size (giảm nếu bị out of memory)
    'per_gpu_batchsize': 8,
    
    # GPU settings
    'num_gpus': 1,                             # Set 0 để dùng CPU
    'precision': '16-mixed',                   # Hoặc 32 cho full precision
}
```

## Support

Nếu có vấn đề, hãy kiểm tra:
1. Log output của script
2. Config đường dẫn có đúng không
3. Checkpoint file có load được không
4. Dataset có được setup đúng không 