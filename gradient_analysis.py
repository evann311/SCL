import os
import copy
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
import json
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger

from scl.config import config_dict
from scl.modules import SCLTransformer
from scl.datamodules.multitask_datamodule import MTDataModule


class GradientAnalysisCallback(Callback):
    """Callback để phân tích gradient trong quá trình training"""
    
    def __init__(self, save_dir="gradient_analysis", log_every_n_steps=10):
        super().__init__()
        self.save_dir = save_dir
        self.log_every_n_steps = log_every_n_steps
        
        # Storage for gradient statistics
        self.gradient_stats = {
            'text_encoder': defaultdict(list),
            'image_encoder': defaultdict(list),
            'cross_modal': defaultdict(list)
        }
        
        # Storage for gradient history (for cosine similarity calculation)
        self.prev_gradients = {
            'text_encoder': {},
            'image_encoder': {},
            'cross_modal': {}
        }
        
        # Step counter
        self.step_count = 0
        
        os.makedirs(save_dir, exist_ok=True)
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Phân tích gradient sau mỗi batch"""
        
        if self.step_count % self.log_every_n_steps != 0:
            self.step_count += 1
            return
            
        # Collect gradients from different components
        self._analyze_text_encoder_gradients(pl_module)
        self._analyze_image_encoder_gradients(pl_module)
        self._analyze_cross_modal_gradients(pl_module)
        
        # Log statistics to tensorboard
        self._log_gradient_statistics(trainer, pl_module)
        
        self.step_count += 1
        
    def _analyze_text_encoder_gradients(self, pl_module):
        """Phân tích gradient của text encoder (RoBERTa)"""
        text_grads = {}
        
        for name, param in pl_module.text_transformer.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach()
                
                # Calculate L2 norm
                l2_norm = torch.norm(grad).item()
                
                # Calculate mean and std
                grad_mean = grad.mean().item()
                grad_std = grad.std().item()
                
                # Store for layer-wise analysis
                layer_info = self._extract_layer_info(name, 'text')
                if layer_info:
                    layer_idx, component = layer_info
                    key = f"layer_{layer_idx}_{component}"
                    
                    self.gradient_stats['text_encoder'][f'{key}_l2_norm'].append(l2_norm)
                    self.gradient_stats['text_encoder'][f'{key}_mean'].append(grad_mean)
                    self.gradient_stats['text_encoder'][f'{key}_std'].append(grad_std)
                    
                    # Store gradient for cosine similarity calculation
                    text_grads[key] = grad.flatten()
        
        # Calculate cosine similarity with previous step
        self._calculate_cosine_similarity(text_grads, 'text_encoder')
        
    def _analyze_image_encoder_gradients(self, pl_module):
        """Phân tích gradient của image encoder (CLIP ViT)"""
        image_grads = {}
        
        for name, param in pl_module.vision_transformer.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach()
                
                # Calculate L2 norm
                l2_norm = torch.norm(grad).item()
                
                # Calculate mean and std
                grad_mean = grad.mean().item()
                grad_std = grad.std().item()
                
                # Store for layer-wise analysis
                layer_info = self._extract_layer_info(name, 'image')
                if layer_info:
                    layer_idx, component = layer_info
                    key = f"layer_{layer_idx}_{component}"
                    
                    self.gradient_stats['image_encoder'][f'{key}_l2_norm'].append(l2_norm)
                    self.gradient_stats['image_encoder'][f'{key}_mean'].append(grad_mean)
                    self.gradient_stats['image_encoder'][f'{key}_std'].append(grad_std)
                    
                    # Store gradient for cosine similarity calculation
                    image_grads[key] = grad.flatten()
        
        # Calculate cosine similarity with previous step
        self._calculate_cosine_similarity(image_grads, 'image_encoder')
        
    def _analyze_cross_modal_gradients(self, pl_module):
        """Phân tích gradient của cross-modal layers (Fusion Encoder)"""
        cross_grads = {}
        
        # Cross-modal text layers
        for i, layer in enumerate(pl_module.cross_modal_text_layers):
            for name, param in layer.named_parameters():
                if param.grad is not None:
                    grad = param.grad.detach()
                    l2_norm = torch.norm(grad).item()
                    grad_mean = grad.mean().item()
                    grad_std = grad.std().item()
                    
                    key = f"cross_text_layer_{i}_{name}"
                    self.gradient_stats['cross_modal'][f'{key}_l2_norm'].append(l2_norm)
                    self.gradient_stats['cross_modal'][f'{key}_mean'].append(grad_mean)
                    self.gradient_stats['cross_modal'][f'{key}_std'].append(grad_std)
                    
                    cross_grads[key] = grad.flatten()
        
        # Cross-modal image layers
        for i, layer in enumerate(pl_module.cross_modal_image_layers):
            for name, param in layer.named_parameters():
                if param.grad is not None:
                    grad = param.grad.detach()
                    l2_norm = torch.norm(grad).item()
                    grad_mean = grad.mean().item()
                    grad_std = grad.std().item()
                    
                    key = f"cross_image_layer_{i}_{name}"
                    self.gradient_stats['cross_modal'][f'{key}_l2_norm'].append(l2_norm)
                    self.gradient_stats['cross_modal'][f'{key}_mean'].append(grad_mean)
                    self.gradient_stats['cross_modal'][f'{key}_std'].append(grad_std)
                    
                    cross_grads[key] = grad.flatten()
        
        # Cross-modal transforms và poolers
        cross_modal_components = [
            ('cross_modal_text_transform', pl_module.cross_modal_text_transform),
            ('cross_modal_image_transform', pl_module.cross_modal_image_transform),
            ('cross_modal_text_pooler', pl_module.cross_modal_text_pooler),
            ('cross_modal_image_pooler', pl_module.cross_modal_image_pooler),
            ('token_type_embeddings', pl_module.token_type_embeddings)
        ]
        
        for comp_name, component in cross_modal_components:
            for name, param in component.named_parameters():
                if param.grad is not None:
                    grad = param.grad.detach()
                    l2_norm = torch.norm(grad).item()
                    grad_mean = grad.mean().item()
                    grad_std = grad.std().item()
                    
                    key = f"{comp_name}_{name}"
                    self.gradient_stats['cross_modal'][f'{key}_l2_norm'].append(l2_norm)
                    self.gradient_stats['cross_modal'][f'{key}_mean'].append(grad_mean)
                    self.gradient_stats['cross_modal'][f'{key}_std'].append(grad_std)
                    
                    cross_grads[key] = grad.flatten()
        
        self._calculate_cosine_similarity(cross_grads, 'cross_modal')
        
    def _extract_layer_info(self, param_name, encoder_type):
        """Trích xuất thông tin layer từ tên parameter"""
        if encoder_type == 'text':
            # RoBERTa layer pattern: encoder.layer.X.attention.self.query.weight
            if 'encoder.layer.' in param_name:
                parts = param_name.split('.')
                layer_idx = int(parts[2])
                if 'attention' in param_name:
                    return layer_idx, 'attention'
                elif 'intermediate' in param_name:
                    return layer_idx, 'ffn'
                elif 'output' in param_name:
                    return layer_idx, 'output'
        
        elif encoder_type == 'image':
            # CLIP ViT pattern: visual.transformer.resblocks.X.attn.in_proj_weight
            if 'visual.transformer.resblocks.' in param_name:
                parts = param_name.split('.')
                layer_idx = int(parts[3])
                if 'attn' in param_name:
                    return layer_idx, 'attention'
                elif 'mlp' in param_name:
                    return layer_idx, 'ffn'
        
        return None
        
    def _calculate_cosine_similarity(self, current_grads, encoder_type):
        """Tính cosine similarity giữa gradient hiện tại và bước trước"""
        if encoder_type not in self.prev_gradients:
            self.prev_gradients[encoder_type] = current_grads
            return
            
        prev_grads = self.prev_gradients[encoder_type]
        
        for key in current_grads:
            if key in prev_grads:
                current_grad = current_grads[key]
                prev_grad = prev_grads[key]
                
                # Ensure same size (in case of shape changes)
                if current_grad.shape == prev_grad.shape:
                    cosine_sim = F.cosine_similarity(
                        current_grad.unsqueeze(0), 
                        prev_grad.unsqueeze(0)
                    ).item()
                    
                    self.gradient_stats[encoder_type][f'{key}_cosine_sim'].append(cosine_sim)
        
        # Update previous gradients
        self.prev_gradients[encoder_type] = current_grads
        
    def _log_gradient_statistics(self, trainer, pl_module):
        """Log gradient statistics to tensorboard"""
        step = trainer.global_step
        
        for encoder_type, stats in self.gradient_stats.items():
            for metric_name, values in stats.items():
                if values:  # Only log if there are values
                    recent_value = values[-1]  # Most recent value
                    pl_module.log(f"gradients/{encoder_type}/{metric_name}", recent_value, on_step=True)
    
    def on_train_end(self, trainer, pl_module):
        """Tạo báo cáo và visualizations cuối training"""
        self._save_gradient_statistics()
        self._create_visualizations()
        self._compute_gradient_entropy()
        self._generate_report()
        
    def _save_gradient_statistics(self):
        """Lưu statistics vào file JSON"""
        # Convert to serializable format
        serializable_stats = {}
        for encoder_type, stats in self.gradient_stats.items():
            serializable_stats[encoder_type] = {}
            for metric_name, values in stats.items():
                serializable_stats[encoder_type][metric_name] = values
        
        save_path = os.path.join(self.save_dir, 'gradient_statistics.json')
        with open(save_path, 'w') as f:
            json.dump(serializable_stats, f, indent=2)
        
        print(f"📊 Gradient statistics saved to {save_path}")
        
    def _create_visualizations(self):
        """Tạo các biểu đồ phân tích gradient"""
        
        # 1. Gradient L2 Norm per Layer
        self._plot_gradient_norms()
        
        # 2. Cosine Similarity (Gradient Drift)
        self._plot_cosine_similarity()
        
        # 3. Gradient Distribution Comparison
        self._plot_gradient_distributions()
        
    def _plot_gradient_norms(self):
        """Vẽ biểu đồ L2 norm của gradient theo layer"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Text encoder gradient norms
        text_layers = {}
        for metric_name, values in self.gradient_stats['text_encoder'].items():
            if 'l2_norm' in metric_name and values:
                layer_name = metric_name.replace('_l2_norm', '')
                text_layers[layer_name] = np.mean(values)
        
        if text_layers:
            axes[0].bar(range(len(text_layers)), list(text_layers.values()))
            axes[0].set_xlabel('Layer')
            axes[0].set_ylabel('Average L2 Norm')
            axes[0].set_title('Text Encoder (RoBERTa) - Gradient L2 Norms')
            axes[0].set_xticks(range(len(text_layers)))
            axes[0].set_xticklabels(list(text_layers.keys()), rotation=45, ha='right')
        
        # Image encoder gradient norms
        image_layers = {}
        for metric_name, values in self.gradient_stats['image_encoder'].items():
            if 'l2_norm' in metric_name and values:
                layer_name = metric_name.replace('_l2_norm', '')
                image_layers[layer_name] = np.mean(values)
        
        if image_layers:
            axes[1].bar(range(len(image_layers)), list(image_layers.values()))
            axes[1].set_xlabel('Layer')
            axes[1].set_ylabel('Average L2 Norm')
            axes[1].set_title('Image Encoder (CLIP ViT) - Gradient L2 Norms')
            axes[1].set_xticks(range(len(image_layers)))
            axes[1].set_xticklabels(list(image_layers.keys()), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'gradient_norms_per_layer.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_cosine_similarity(self):
        """Vẽ biểu đồ cosine similarity (gradient drift)"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Text encoder cosine similarity
        text_cosine_data = []
        text_labels = []
        for metric_name, values in self.gradient_stats['text_encoder'].items():
            if 'cosine_sim' in metric_name and values:
                text_cosine_data.append(values)
                text_labels.append(metric_name.replace('_cosine_sim', ''))
        
        if text_cosine_data:
            for i, (data, label) in enumerate(zip(text_cosine_data, text_labels)):
                axes[0].plot(data, label=label, alpha=0.7)
            axes[0].set_xlabel('Training Step')
            axes[0].set_ylabel('Cosine Similarity')
            axes[0].set_title('Text Encoder - Gradient Cosine Similarity (Consistency)')
            axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[0].grid(True, alpha=0.3)
        
        # Image encoder cosine similarity
        image_cosine_data = []
        image_labels = []
        for metric_name, values in self.gradient_stats['image_encoder'].items():
            if 'cosine_sim' in metric_name and values:
                image_cosine_data.append(values)
                image_labels.append(metric_name.replace('_cosine_sim', ''))
        
        if image_cosine_data:
            for i, (data, label) in enumerate(zip(image_cosine_data, image_labels)):
                axes[1].plot(data, label=label, alpha=0.7)
            axes[1].set_xlabel('Training Step')
            axes[1].set_ylabel('Cosine Similarity')
            axes[1].set_title('Image Encoder - Gradient Cosine Similarity (Consistency)')
            axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'gradient_cosine_similarity.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_gradient_distributions(self):
        """Vẽ biểu đồ phân phối gradient"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Text encoder L2 norms distribution
        text_l2_norms = []
        for metric_name, values in self.gradient_stats['text_encoder'].items():
            if 'l2_norm' in metric_name:
                text_l2_norms.extend(values)
        
        if text_l2_norms:
            axes[0, 0].hist(text_l2_norms, bins=50, alpha=0.7, color='blue')
            axes[0, 0].set_xlabel('L2 Norm')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Text Encoder - L2 Norm Distribution')
            axes[0, 0].axvline(np.mean(text_l2_norms), color='red', linestyle='--', label=f'Mean: {np.mean(text_l2_norms):.4f}')
            axes[0, 0].legend()
        
        # Image encoder L2 norms distribution
        image_l2_norms = []
        for metric_name, values in self.gradient_stats['image_encoder'].items():
            if 'l2_norm' in metric_name:
                image_l2_norms.extend(values)
        
        if image_l2_norms:
            axes[0, 1].hist(image_l2_norms, bins=50, alpha=0.7, color='green')
            axes[0, 1].set_xlabel('L2 Norm')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Image Encoder - L2 Norm Distribution')
            axes[0, 1].axvline(np.mean(image_l2_norms), color='red', linestyle='--', label=f'Mean: {np.mean(image_l2_norms):.4f}')
            axes[0, 1].legend()
        
        # Comparison of means over time
        if text_l2_norms and image_l2_norms:
            # Calculate running averages
            window_size = 10
            text_running_avg = self._running_average(text_l2_norms, window_size)
            image_running_avg = self._running_average(image_l2_norms, window_size)
            
            axes[1, 0].plot(text_running_avg, label='Text Encoder', color='blue')
            axes[1, 0].plot(image_running_avg, label='Image Encoder', color='green')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Running Average L2 Norm')
            axes[1, 0].set_title('Gradient Magnitude Comparison Over Training')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Cosine similarity comparison
        text_cosine_all = []
        image_cosine_all = []
        
        for metric_name, values in self.gradient_stats['text_encoder'].items():
            if 'cosine_sim' in metric_name:
                text_cosine_all.extend(values)
        
        for metric_name, values in self.gradient_stats['image_encoder'].items():
            if 'cosine_sim' in metric_name:
                image_cosine_all.extend(values)
        
        if text_cosine_all and image_cosine_all:
            axes[1, 1].hist(text_cosine_all, bins=30, alpha=0.7, color='blue', label='Text Encoder')
            axes[1, 1].hist(image_cosine_all, bins=30, alpha=0.7, color='green', label='Image Encoder')
            axes[1, 1].set_xlabel('Cosine Similarity')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Gradient Consistency Distribution')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'gradient_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def _running_average(self, data, window_size):
        """Calculate running average"""
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
        
    def _compute_gradient_entropy(self):
        """Tính entropy của phân phối gradient theo layer"""
        entropy_results = {}
        
        for encoder_type in ['text_encoder', 'image_encoder']:
            layer_l2_norms = {}
            
            # Collect L2 norms by layer
            for metric_name, values in self.gradient_stats[encoder_type].items():
                if 'l2_norm' in metric_name and values:
                    layer_name = metric_name.replace('_l2_norm', '')
                    layer_l2_norms[layer_name] = np.mean(values)
            
            if layer_l2_norms:
                # Normalize to create a probability distribution
                total = sum(layer_l2_norms.values())
                if total > 0:
                    probabilities = [v / total for v in layer_l2_norms.values()]
                    
                    # Calculate entropy
                    entropy = -sum(p * np.log(p + 1e-8) for p in probabilities if p > 0)
                    entropy_results[encoder_type] = {
                        'entropy': entropy,
                        'max_entropy': np.log(len(probabilities)),
                        'normalized_entropy': entropy / np.log(len(probabilities)) if len(probabilities) > 1 else 0,
                        'layer_distribution': dict(zip(layer_l2_norms.keys(), probabilities))
                    }
        
        # Save entropy results
        entropy_path = os.path.join(self.save_dir, 'gradient_entropy.json')
        with open(entropy_path, 'w') as f:
            json.dump(entropy_results, f, indent=2)
            
        return entropy_results
        
    def _generate_report(self):
        """Tạo báo cáo tổng hợp"""
        report = []
        report.append("# 🔬 Gradient Analysis Report")
        report.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary statistics
        report.append("## 📊 Summary Statistics")
        
        for encoder_type in ['text_encoder', 'image_encoder']:
            report.append(f"\n### {encoder_type.replace('_', ' ').title()}")
            
            # L2 norm statistics
            l2_norms = []
            for metric_name, values in self.gradient_stats[encoder_type].items():
                if 'l2_norm' in metric_name:
                    l2_norms.extend(values)
            
            if l2_norms:
                report.append(f"- **L2 Norm Mean**: {np.mean(l2_norms):.6f}")
                report.append(f"- **L2 Norm Std**: {np.std(l2_norms):.6f}")
                report.append(f"- **L2 Norm Min**: {np.min(l2_norms):.6f}")
                report.append(f"- **L2 Norm Max**: {np.max(l2_norms):.6f}")
            
            # Cosine similarity statistics
            cosine_sims = []
            for metric_name, values in self.gradient_stats[encoder_type].items():
                if 'cosine_sim' in metric_name:
                    cosine_sims.extend(values)
            
            if cosine_sims:
                report.append(f"- **Cosine Similarity Mean**: {np.mean(cosine_sims):.6f}")
                report.append(f"- **Cosine Similarity Std**: {np.std(cosine_sims):.6f}")
                report.append(f"- **Gradient Consistency**: {'High' if np.mean(cosine_sims) > 0.8 else 'Medium' if np.mean(cosine_sims) > 0.5 else 'Low'}")
        
        # Comparative analysis
        report.append("\n## ⚖️ Comparative Analysis")
        
        text_l2_all = []
        image_l2_all = []
        
        for metric_name, values in self.gradient_stats['text_encoder'].items():
            if 'l2_norm' in metric_name:
                text_l2_all.extend(values)
        
        for metric_name, values in self.gradient_stats['image_encoder'].items():
            if 'l2_norm' in metric_name:
                image_l2_all.extend(values)
        
        if text_l2_all and image_l2_all:
            text_mean = np.mean(text_l2_all)
            image_mean = np.mean(image_l2_all)
            ratio = text_mean / image_mean if image_mean > 0 else 0
            
            report.append(f"- **Text/Image Gradient Magnitude Ratio**: {ratio:.3f}")
            if ratio > 1.5:
                report.append("  - ⚠️ Text encoder gradients are significantly larger")
            elif ratio < 0.67:
                report.append("  - ⚠️ Image encoder gradients are significantly larger")
            else:
                report.append("  - ✅ Gradient magnitudes are relatively balanced")
        
        # Recommendations
        report.append("\n## 💡 Recommendations")
        
        if text_l2_all and image_l2_all:
            text_mean = np.mean(text_l2_all)
            image_mean = np.mean(image_l2_all)
            
            if text_mean > image_mean * 2:
                report.append("- Consider reducing learning rate for text encoder")
                report.append("- Text encoder may be overfitting faster than image encoder")
            elif image_mean > text_mean * 2:
                report.append("- Consider reducing learning rate for image encoder")
                report.append("- Image encoder may be overfitting faster than text encoder")
            else:
                report.append("- Gradient magnitudes appear balanced")
        
        # Save report
        report_path = os.path.join(self.save_dir, 'gradient_analysis_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"📋 Analysis report saved to {report_path}")




def unfreeze_all_parameters(model):
    """Unfreeze tất cả parameters của model"""
    total_params = 0
    unfrozen_params = 0
    
    for name, param in model.named_parameters():
        total_params += 1
        if not param.requires_grad:
            param.requires_grad = True
            unfrozen_params += 1
    
    print(f"🔓 Unfrozen {unfrozen_params} out of {total_params} parameters")
    print(f"📊 Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model


def main():
    """Main function để chạy gradient analysis"""
    
    # Load config
    config_name = 'vqa_vast'
    _config = copy.deepcopy(config_dict[config_name])
    
    # Modify config for analysis
    _config["max_steps"] = 200 # Reduce steps for analysis

    _config["val_check_interval"] = 200
    _config["log_every_n_steps"] = 1
    
    # Configure for 2 GPU setup
    _config["num_gpus"] = 2
    _config["num_nodes"] = 1
    _config["batch_size"] = 48
    _config["per_gpu_batchsize"] = 24
    
    print(f"🚀 Starting Gradient Analysis with config: {config_name}")
    print(f"📊 Analysis will run for {_config['max_steps']} steps")
    
    # Set seed
    pl.seed_everything(_config["seed"])
    
    # Create data module
    dm = MTDataModule(_config, dist=False)
    dm.setup('fit')
    
    # Create model
    model = SCLTransformer(_config)
    
    # Unfreeze all parameters
    model = unfreeze_all_parameters(model)
    
    # Print parameter statistics
    print("\n📋 Model Parameter Statistics:")
    model.print_parameter_statistics()
    
    # Create gradient analysis callback
    gradient_callback = GradientAnalysisCallback(
        save_dir=f"gradient_analysis_{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        log_every_n_steps=1
    )
    
    # Setup trainer
    logger = TensorBoardLogger(
        _config["log_dir"],
        name=f"gradient_analysis_{config_name}",
        version=f"unfrozen_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Setup strategy for multi-GPU or single GPU
    if _config.get("num_gpus", 1) > 1:
        from pytorch_lightning.strategies import DDPStrategy
        strategy = DDPStrategy(find_unused_parameters=True)
    else:
        strategy = "auto"
    
    trainer = pl.Trainer(
        accelerator="gpu" if _config.get("num_gpus", 0) > 0 else "cpu",
        devices=_config.get("num_gpus", 1),
        max_steps=_config["max_steps"],
        logger=logger,
        callbacks=[gradient_callback],
        log_every_n_steps=_config["log_every_n_steps"],
        enable_model_summary=True,
        deterministic=True,
        precision=_config.get("precision", 32),
        strategy=strategy
    )
    
    print(f"\n🎯 Training model for gradient analysis...")
    print(f"📊 Results will be saved to: {gradient_callback.save_dir}")
    print(f"📈 TensorBoard logs: {logger.log_dir}")
    
    # Start training
    trainer.fit(model, datamodule=dm)
    
    print(f"\n✅ Gradient analysis completed!")
    print(f"📊 Check results in: {gradient_callback.save_dir}")
    print(f"📈 View real-time metrics: tensorboard --logdir {logger.log_dir}")


if __name__ == '__main__':
    main() 