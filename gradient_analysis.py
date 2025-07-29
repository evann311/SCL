import os
import copy
import torch
import torch.nn.functional as F
import numpy as np
import json
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from scl.config import config_dict
from scl.modules import SCLTransformer
from scl.datamodules.multitask_datamodule import MTDataModule


class CosineAnalysisCallback(Callback):
    """Callback để phân tích cosine similarity của gradients"""
    
    def __init__(self, save_dir="cosine_analysis", target_encoder="all"):
        super().__init__()
        self.save_dir = save_dir
        self.target_encoder = target_encoder
        
        # Storage for cosine similarity only
        self.cosine_data = {
            'text_encoder': [],
            'image_encoder': [],
            'cross_modal': [],
            'vqa_head': []
        }
        
        # Storage for previous gradients (for cosine calculation)
        self.prev_gradients = {
            'text_encoder': {},
            'image_encoder': {},
            'cross_modal': {},
            'vqa_head': {}
        }
        
        # Step tracking
        self.step_count = 0
        self.step_results = []
        
        os.makedirs(save_dir, exist_ok=True)
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Phân tích cosine similarity sau mỗi batch"""
        
        current_step = trainer.global_step
        step_cosines = {'step': current_step}
        
        # Analyze based on target encoder
        if self.target_encoder in ['text', 'all']:
            text_cosines = self._analyze_text_encoder_gradients(pl_module)
            if text_cosines:
                step_cosines['text_avg_cosine'] = np.mean(list(text_cosines.values()))
        
        if self.target_encoder in ['image', 'all']:
            image_cosines = self._analyze_image_encoder_gradients(pl_module)
            if image_cosines:
                step_cosines['image_avg_cosine'] = np.mean(list(image_cosines.values()))
            
        if self.target_encoder in ['cross', 'all']:
            cross_cosines = self._analyze_cross_modal_gradients(pl_module)
            if cross_cosines:
                step_cosines['cross_avg_cosine'] = np.mean(list(cross_cosines.values()))
        
        # Always analyze VQA head
        vqa_cosines = self._analyze_vqa_head_gradients(pl_module)
        if vqa_cosines:
            step_cosines['vqa_avg_cosine'] = np.mean(list(vqa_cosines.values()))
        
        # Store step results
        self.step_results.append(step_cosines)
        
        # Print progress every 50 steps
        if current_step % 50 == 0:
            text_avg = step_cosines.get('text_avg_cosine', 0)
            image_avg = step_cosines.get('image_avg_cosine', 0)
            cross_avg = step_cosines.get('cross_avg_cosine', 0)
            vqa_avg = step_cosines.get('vqa_avg_cosine', 0)
            print(f"Step {current_step} [{self.target_encoder}]: Text={text_avg:.4f}, Image={image_avg:.4f}, Cross={cross_avg:.4f}, VQA={vqa_avg:.4f}")
        
        self.step_count += 1
        
    def _analyze_text_encoder_gradients(self, pl_module):
        """Phân tích cosine similarity của text encoder"""
        current_grads = {}
        
        for name, param in pl_module.text_transformer.named_parameters():
            if param.grad is not None and param.requires_grad:
                grad = param.grad.detach().flatten()
                layer_info = self._extract_layer_info(name, 'text')
                if layer_info:
                    layer_idx, component = layer_info
                    key = f"layer_{layer_idx}_{component}"
                    current_grads[key] = grad
        
        return self._calculate_cosine_similarity(current_grads, 'text_encoder')
        
    def _analyze_image_encoder_gradients(self, pl_module):
        """Phân tích cosine similarity của image encoder"""
        current_grads = {}
        
        for name, param in pl_module.vision_transformer.named_parameters():
            if param.grad is not None and param.requires_grad:
                grad = param.grad.detach().flatten()
                layer_info = self._extract_layer_info(name, 'image')
                if layer_info:
                    layer_idx, component = layer_info
                    key = f"layer_{layer_idx}_{component}"
                    current_grads[key] = grad
        
        return self._calculate_cosine_similarity(current_grads, 'image_encoder')
        
    def _analyze_cross_modal_gradients(self, pl_module):
        """Phân tích cosine similarity của cross-modal layers"""
        current_grads = {}
        
        # Cross-modal text layers
        for i, layer in enumerate(pl_module.cross_modal_text_layers):
            for name, param in layer.named_parameters():
                if param.grad is not None and param.requires_grad:
                    key = f"cross_text_layer_{i}_{name}"
                    current_grads[key] = param.grad.detach().flatten()
        
        # Cross-modal image layers
        for i, layer in enumerate(pl_module.cross_modal_image_layers):
            for name, param in layer.named_parameters():
                if param.grad is not None and param.requires_grad:
                    key = f"cross_image_layer_{i}_{name}"
                    current_grads[key] = param.grad.detach().flatten()
        
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
                if param.grad is not None and param.requires_grad:
                    key = f"{comp_name}_{name}"
                    current_grads[key] = param.grad.detach().flatten()
        
        return self._calculate_cosine_similarity(current_grads, 'cross_modal')
    
    def _analyze_vqa_head_gradients(self, pl_module):
        """Phân tích cosine similarity của VQA classifier head"""
        current_grads = {}
        
        # VQA classifier head
        if hasattr(pl_module, 'vqa_classifier'):
            for name, param in pl_module.vqa_classifier.named_parameters():
                if param.grad is not None and param.requires_grad:
                    key = f"vqa_head_{name}"
                    current_grads[key] = param.grad.detach().flatten()
        
        return self._calculate_cosine_similarity(current_grads, 'vqa_head')
        
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
        cosine_results = {}
        
        if encoder_type not in self.prev_gradients:
            self.prev_gradients[encoder_type] = current_grads
            return cosine_results
            
        prev_grads = self.prev_gradients[encoder_type]
        
        for key in current_grads:
            if key in prev_grads:
                current_grad = current_grads[key]
                prev_grad = prev_grads[key]
                
                # Ensure same size
                if current_grad.shape == prev_grad.shape:
                    cosine_sim = F.cosine_similarity(
                        current_grad.unsqueeze(0), 
                        prev_grad.unsqueeze(0)
                    ).item()
                    
                    cosine_results[key] = cosine_sim
                    self.cosine_data[encoder_type].append({
                        'step': self.step_count,
                        'layer': key,
                        'cosine': cosine_sim
                    })
        
        # Update previous gradients
        self.prev_gradients[encoder_type] = current_grads
        return cosine_results
    
    def on_train_end(self, trainer, pl_module):
        """Lưu kết quả cosine similarity"""
        self._save_cosine_results()
        self._generate_summary()
        
    def _save_cosine_results(self):
        """Lưu step-by-step cosine results"""
        
        # Save detailed step results
        step_results_path = os.path.join(self.save_dir, f'cosine_step_results_{self.target_encoder}.json')
        with open(step_results_path, 'w') as f:
            json.dump(self.step_results, f, indent=2)
        
        # Save raw cosine data
        cosine_data_path = os.path.join(self.save_dir, f'cosine_raw_data_{self.target_encoder}.json')
        with open(cosine_data_path, 'w') as f:
            json.dump(self.cosine_data, f, indent=2)
        
        print(f"📊 Cosine step results saved to {step_results_path}")
        print(f"📊 Cosine raw data saved to {cosine_data_path}")
        
    def _generate_summary(self):
        """Tạo summary cosine analysis"""
        summary = {'target_encoder': self.target_encoder}
        
        for encoder_type in ['text_encoder', 'image_encoder', 'cross_modal', 'vqa_head']:
            if self.cosine_data[encoder_type]:
                cosines = [item['cosine'] for item in self.cosine_data[encoder_type]]
                summary[encoder_type] = {
                    'mean_cosine': np.mean(cosines),
                    'std_cosine': np.std(cosines),
                    'min_cosine': np.min(cosines),
                    'max_cosine': np.max(cosines),
                    'final_cosine': cosines[-1] if cosines else 0,
                    'total_measurements': len(cosines)
                }
        
        # Save summary
        summary_path = os.path.join(self.save_dir, f'cosine_summary_{self.target_encoder}.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print(f"\n🔬 === COSINE SIMILARITY ANALYSIS [{self.target_encoder.upper()}] ===")
        for encoder_type, stats in summary.items():
            if encoder_type != 'target_encoder' and isinstance(stats, dict):
                print(f"\n{encoder_type.replace('_', ' ').title()}:")
                print(f"  Mean Cosine: {stats['mean_cosine']:.4f} ± {stats['std_cosine']:.4f}")
                print(f"  Final Cosine: {stats['final_cosine']:.4f}")
                print(f"  Range: [{stats['min_cosine']:.4f}, {stats['max_cosine']:.4f}]")
                print(f"  Measurements: {stats['total_measurements']}")
        
        print(f"\n📊 Summary saved to {summary_path}")


def freeze_all_except_target(model, target_encoder):
    """Freeze tất cả parameters trừ target encoder và VQA head"""
    
    total_params = 0
    frozen_params = 0
    unfrozen_params = 0
    
    for name, param in model.named_parameters():
        total_params += 1
        should_train = False
        
        # Always train VQA classifier
        if 'vqa_classifier' in name:
            should_train = True
            
        # Train target encoder
        elif target_encoder == 'text' and 'text_transformer' in name:
            should_train = True
        elif target_encoder == 'image' and 'vision_transformer' in name:
            should_train = True
        elif target_encoder == 'cross' and ('cross_modal' in name or 'token_type_embeddings' in name):
            should_train = True
            
        # Set requires_grad
        param.requires_grad = should_train
        
        if should_train:
            unfrozen_params += 1
        else:
            frozen_params += 1
    
    print(f"🎯 Target Encoder: {target_encoder}")
    print(f"🔒 Frozen parameters: {frozen_params}")
    print(f"🔓 Unfrozen parameters: {unfrozen_params}")
    print(f"📊 Total parameters: {total_params}")
    
    return model


def run_single_encoder_experiment(config_name, target_encoder, max_steps=500):
    """Chạy experiment cho một encoder cụ thể"""
    
    print(f"\n{'='*60}")
    print(f"🔬 RUNNING EXPERIMENT: {target_encoder.upper()} ENCODER")
    print(f"{'='*60}")
    
    # Load config
    _config = copy.deepcopy(config_dict[config_name])
    
    # Modify config for analysis
    _config["max_steps"] = max_steps
    _config["val_check_interval"] = max_steps + 100  # Disable validation
    _config["log_every_n_steps"] = 1
    
    # Configure for 1 GPU setup
    _config["num_gpus"] = 1
    _config["num_nodes"] = 1
    _config["batch_size"] = 64
    _config["per_gpu_batchsize"] = 64
    
    # Set seed
    pl.seed_everything(_config["seed"])
    
    # Create data module
    dm = MTDataModule(_config, dist=False)
    dm.setup('fit')
    
    # Create model
    model = SCLTransformer(_config)
    
    # Freeze all except target encoder
    model = freeze_all_except_target(model, target_encoder)
    
    # Create cosine analysis callback
    experiment_dir = f"encoder_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    cosine_callback = CosineAnalysisCallback(
        save_dir=experiment_dir,
        target_encoder=target_encoder
    )
    
    # Setup trainer
    trainer = pl.Trainer(
        accelerator="gpu" if _config.get("num_gpus", 0) > 0 else "cpu",
        devices=_config.get("num_gpus", 1),
        max_steps=_config["max_steps"],
        callbacks=[cosine_callback],
        enable_model_summary=False,
        deterministic=True,
        precision=_config.get("precision", 32),
        strategy="auto",
        enable_progress_bar=True,
        logger=False  # Disable all logging
    )
    
    print(f"\n🎯 Training {target_encoder} encoder for {max_steps} steps...")
    
    # Start training
    trainer.fit(model, datamodule=dm)
    
    print(f"\n✅ {target_encoder.upper()} encoder experiment completed!")
    
    return experiment_dir


def main():
    """Main function để chạy experiments cho từng encoder"""
    
    config_name = 'vqa_vast'
    max_steps = 500
    
    print(f"🚀 Starting Individual Encoder Analysis")
    print(f"📊 Config: {config_name}")
    print(f"📊 Steps per encoder: {max_steps}")
    
    # Run experiments for each encoder
    for encoder_type in ['text', 'image', 'cross']:
        exp_dir = run_single_encoder_experiment(config_name, encoder_type, max_steps)
        print(f"📁 {encoder_type.title()} experiment saved to: {exp_dir}")
    
    print(f"\n✅ All encoder experiments completed!")
    print(f"💡 Run 'python analyze_results.py' to compare results")


if __name__ == '__main__':
    main() 