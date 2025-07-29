import os
import copy
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

from scl.config import config_dict
from scl.modules import SCLTransformer
from scl.datamodules.multitask_datamodule import MTDataModule


def analyze_single_batch_gradients(model, batch):
    """Phân tích gradient cho một batch"""
    
    # Forward pass
    model.train()
    output = model(batch)
    loss = sum([v for k, v in output.items() if "loss" in k])
    
    # Backward pass
    loss.backward()
    
    # Analyze gradients
    gradient_stats = {
        'text_encoder': {},
        'image_encoder': {},
        'cross_modal': {}
    }
    
    # Text encoder (RoBERTa)
    print("\n🔤 Text Encoder (RoBERTa) Gradients:")
    text_total_norm = 0
    text_layer_norms = {}
    
    for name, param in model.text_transformer.named_parameters():
        if param.grad is not None:
            grad_norm = torch.norm(param.grad).item()
            text_total_norm += grad_norm ** 2
            
            # Group by layer
            if 'encoder.layer.' in name:
                layer_idx = name.split('.')[3]  # Extract layer number
                if layer_idx not in text_layer_norms:
                    text_layer_norms[layer_idx] = 0
                text_layer_norms[layer_idx] += grad_norm ** 2
    
    text_total_norm = np.sqrt(text_total_norm)
    text_layer_norms = {k: np.sqrt(v) for k, v in text_layer_norms.items()}
    
    print(f"  Total L2 norm: {text_total_norm:.6f}")
    for layer, norm in sorted(text_layer_norms.items(), key=lambda x: int(x[0])):
        print(f"  Layer {layer}: {norm:.6f}")
    
    # Image encoder (CLIP ViT)
    print("\n🖼️ Image Encoder (CLIP ViT) Gradients:")
    image_total_norm = 0
    image_layer_norms = {}
    
    for name, param in model.vision_transformer.named_parameters():
        if param.grad is not None:
            grad_norm = torch.norm(param.grad).item()
            image_total_norm += grad_norm ** 2
            
            # Group by layer
            if 'visual.transformer.resblocks.' in name:
                layer_idx = name.split('.')[3]  # Extract layer number
                if layer_idx not in image_layer_norms:
                    image_layer_norms[layer_idx] = 0
                image_layer_norms[layer_idx] += grad_norm ** 2
    
    image_total_norm = np.sqrt(image_total_norm)
    image_layer_norms = {k: np.sqrt(v) for k, v in image_layer_norms.items()}
    
    print(f"  Total L2 norm: {image_total_norm:.6f}")
    for layer, norm in sorted(image_layer_norms.items(), key=lambda x: int(x[0])):
        print(f"  Layer {layer}: {norm:.6f}")
    
    # Cross-modal layers
    print("\n🔄 Cross-Modal Layers Gradients:")
    cross_total_norm = 0
    
    # Cross-modal text layers
    for i, layer in enumerate(model.cross_modal_text_layers):
        layer_norm = 0
        for name, param in layer.named_parameters():
            if param.grad is not None:
                grad_norm = torch.norm(param.grad).item()
                layer_norm += grad_norm ** 2
                cross_total_norm += grad_norm ** 2
        print(f"  Cross-text layer {i}: {np.sqrt(layer_norm):.6f}")
    
    # Cross-modal image layers
    for i, layer in enumerate(model.cross_modal_image_layers):
        layer_norm = 0
        for name, param in layer.named_parameters():
            if param.grad is not None:
                grad_norm = torch.norm(param.grad).item()
                layer_norm += grad_norm ** 2
                cross_total_norm += grad_norm ** 2
        print(f"  Cross-image layer {i}: {np.sqrt(layer_norm):.6f}")
    
    print(f"  Cross-modal total L2 norm: {np.sqrt(cross_total_norm):.6f}")
    
    # Comparison
    print(f"\n⚖️ Comparison:")
    print(f"  Text/Image ratio: {text_total_norm/image_total_norm:.3f}")
    print(f"  Text/Cross-modal ratio: {text_total_norm/np.sqrt(cross_total_norm):.3f}")
    print(f"  Image/Cross-modal ratio: {image_total_norm/np.sqrt(cross_total_norm):.3f}")
    
    # Clear gradients
    model.zero_grad()
    
    return {
        'text_total': text_total_norm,
        'image_total': image_total_norm,
        'cross_total': np.sqrt(cross_total_norm),
        'text_layers': text_layer_norms,
        'image_layers': image_layer_norms
    }


def unfreeze_all_parameters(model):
    """Unfreeze tất cả parameters"""
    for param in model.parameters():
        param.requires_grad = True
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔓 Total trainable parameters: {trainable_params:,}")
    
    return model


def main():
    """Test gradient analysis với một vài batches"""
    
    # Load config
    config_name = 'vqa_vast'
    _config = copy.deepcopy(config_dict[config_name])
    
    # Configure batch sizes for gradient analysis
    _config["batch_size"] = 4  # Small batch size for testing
    _config["per_gpu_batchsize"] = 2  # Conservative per GPU batch size
    _config["num_workers"] = 2  # Reduce workers
    
    print(f"🚀 Quick Gradient Analysis Test")
    print(f"📊 Config: {config_name}")
    print(f"⚙️ Batch size: {_config['batch_size']}, Per GPU: {_config['per_gpu_batchsize']}")
    
    # Create data module
    dm = MTDataModule(_config, dist=False)
    dm.setup('fit')
    
    # Create model
    model = SCLTransformer(_config)
    
    # Unfreeze all parameters
    model = unfreeze_all_parameters(model)
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"🔧 Using device: {device}")
    
    # Test with a few batches
    dataloader = dm.train_dataloader()
    
    results = []
    
    print(f"\n🔬 Analyzing gradients for 5 batches...")
    
    for i, batch in enumerate(tqdm(dataloader)):
        if i >= 5:  # Only test 5 batches
            break
            
        # Move batch to device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                batch[key] = [v.to(device) for v in value]
        
        print(f"\n--- Batch {i+1} ---")
        try:
            result = analyze_single_batch_gradients(model, batch)
            results.append(result)
        except Exception as e:
            print(f"❌ Error in batch {i+1}: {e}")
            continue
    
    # Summary statistics
    if results:
        print(f"\n📊 Summary Statistics (across {len(results)} batches):")
        
        text_norms = [r['text_total'] for r in results]
        image_norms = [r['image_total'] for r in results]
        cross_norms = [r['cross_total'] for r in results]
        
        print(f"  Text encoder average norm: {np.mean(text_norms):.6f} ± {np.std(text_norms):.6f}")
        print(f"  Image encoder average norm: {np.mean(image_norms):.6f} ± {np.std(image_norms):.6f}")
        print(f"  Cross-modal average norm: {np.mean(cross_norms):.6f} ± {np.std(cross_norms):.6f}")
        
        avg_ratio = np.mean([t/i for t, i in zip(text_norms, image_norms)])
        print(f"  Average Text/Image ratio: {avg_ratio:.3f}")
        
        if avg_ratio > 2.0:
            print("  ⚠️ Text encoder gradients are much larger - may indicate instability")
        elif avg_ratio < 0.5:
            print("  ⚠️ Image encoder gradients are much larger - may indicate instability")
        else:
            print("  ✅ Gradient magnitudes are relatively balanced")
            
        # Plot simple comparison
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(text_norms, 'b-', label='Text Encoder', marker='o')
        plt.plot(image_norms, 'g-', label='Image Encoder', marker='s')
        plt.plot(cross_norms, 'r-', label='Cross-Modal', marker='^')
        plt.xlabel('Batch')
        plt.ylabel('Gradient L2 Norm')
        plt.title('Gradient Magnitude Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        ratios = [t/i for t, i in zip(text_norms, image_norms)]
        plt.plot(ratios, 'purple', marker='o')
        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Equal gradients')
        plt.xlabel('Batch')
        plt.ylabel('Text/Image Gradient Ratio')
        plt.title('Gradient Ratio Over Batches')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('quick_gradient_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n📈 Plot saved as 'quick_gradient_analysis.png'")


if __name__ == '__main__':
    main() 