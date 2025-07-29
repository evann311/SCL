import os
import copy
import torch
import torch.nn.functional as F
import numpy as np
import json
from datetime import datetime
from collections import defaultdict

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from scl.config import config_dict
from scl.modules import SCLTransformer
from scl.datamodules.multitask_datamodule import MTDataModule


class CosineAnalysisCallback(Callback):
    """Callback để phân tích cosine similarity của gradients"""
    
    def __init__(self, save_dir="cosine_analysis"):
        super().__init__()
        self.save_dir = save_dir
        
        # Storage for cosine similarity only
        self.cosine_data = {
            'text_encoder': [],
            'image_encoder': [],
            'cross_modal': []
        }
        
        # Storage for previous gradients (for cosine calculation)
        self.prev_gradients = {
            'text_encoder': {},
            'image_encoder': {},
            'cross_modal': {}
        }
        
        # Step tracking
        self.step_count = 0
        self.step_results = []  # Store results for each step
        
        os.makedirs(save_dir, exist_ok=True)
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Phân tích cosine similarity sau mỗi batch"""
        
        current_step = trainer.global_step
        step_cosines = {'step': current_step}
        
        # Analyze text encoder gradients
        text_cosines = self._analyze_text_encoder_gradients(pl_module)
        if text_cosines:
            step_cosines['text_avg_cosine'] = np.mean(list(text_cosines.values()))
            step_cosines['text_cosines'] = text_cosines
        
        # Analyze image encoder gradients  
        image_cosines = self._analyze_image_encoder_gradients(pl_module)
        if image_cosines:
            step_cosines['image_avg_cosine'] = np.mean(list(image_cosines.values()))
            step_cosines['image_cosines'] = image_cosines
            
        # Analyze cross-modal gradients
        cross_cosines = self._analyze_cross_modal_gradients(pl_module)
        if cross_cosines:
            step_cosines['cross_avg_cosine'] = np.mean(list(cross_cosines.values()))
            step_cosines['cross_cosines'] = cross_cosines
        
        # Store step results
        self.step_results.append(step_cosines)
        
        # Print progress every 50 steps
        if current_step % 50 == 0:
            text_avg = step_cosines.get('text_avg_cosine', 0)
            image_avg = step_cosines.get('image_avg_cosine', 0)
            cross_avg = step_cosines.get('cross_avg_cosine', 0)
            print(f"Step {current_step}: Text={text_avg:.4f}, Image={image_avg:.4f}, Cross={cross_avg:.4f}")
        
        self.step_count += 1
        
    def _analyze_text_encoder_gradients(self, pl_module):
        """Phân tích cosine similarity của text encoder"""
        current_grads = {}
        cosine_results = {}
        
        for name, param in pl_module.text_transformer.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach().flatten()
                layer_info = self._extract_layer_info(name, 'text')
                if layer_info:
                    layer_idx, component = layer_info
                    key = f"layer_{layer_idx}_{component}"
                    current_grads[key] = grad
        
        # Calculate cosine similarity with previous step
        cosine_results = self._calculate_cosine_similarity(current_grads, 'text_encoder')
        return cosine_results
        
    def _analyze_image_encoder_gradients(self, pl_module):
        """Phân tích cosine similarity của image encoder"""
        current_grads = {}
        cosine_results = {}
        
        for name, param in pl_module.vision_transformer.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach().flatten()
                layer_info = self._extract_layer_info(name, 'image')
                if layer_info:
                    layer_idx, component = layer_info
                    key = f"layer_{layer_idx}_{component}"
                    current_grads[key] = grad
        
        # Calculate cosine similarity with previous step
        cosine_results = self._calculate_cosine_similarity(current_grads, 'image_encoder')
        return cosine_results
        
    def _analyze_cross_modal_gradients(self, pl_module):
        """Phân tích cosine similarity của cross-modal layers"""
        current_grads = {}
        cosine_results = {}
        
        # Cross-modal text layers
        for i, layer in enumerate(pl_module.cross_modal_text_layers):
            for name, param in layer.named_parameters():
                if param.grad is not None:
                    key = f"cross_text_layer_{i}_{name}"
                    current_grads[key] = param.grad.detach().flatten()
        
        # Cross-modal image layers
        for i, layer in enumerate(pl_module.cross_modal_image_layers):
            for name, param in layer.named_parameters():
                if param.grad is not None:
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
                if param.grad is not None:
                    key = f"{comp_name}_{name}"
                    current_grads[key] = param.grad.detach().flatten()
        
        cosine_results = self._calculate_cosine_similarity(current_grads, 'cross_modal')
        return cosine_results
        
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
        step_results_path = os.path.join(self.save_dir, 'cosine_step_results.json')
        with open(step_results_path, 'w') as f:
            json.dump(self.step_results, f, indent=2)
        
        # Save raw cosine data
        cosine_data_path = os.path.join(self.save_dir, 'cosine_raw_data.json')
        with open(cosine_data_path, 'w') as f:
            json.dump(self.cosine_data, f, indent=2)
        
        print(f"📊 Cosine step results saved to {step_results_path}")
        print(f"📊 Cosine raw data saved to {cosine_data_path}")
        
    def _generate_summary(self):
        """Tạo summary cosine analysis"""
        summary = {}
        
        for encoder_type in ['text_encoder', 'image_encoder', 'cross_modal']:
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
        summary_path = os.path.join(self.save_dir, 'cosine_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print(f"\n🔬 === COSINE SIMILARITY ANALYSIS ===")
        for encoder_type, stats in summary.items():
            print(f"\n{encoder_type.replace('_', ' ').title()}:")
            print(f"  Mean Cosine: {stats['mean_cosine']:.4f} ± {stats['std_cosine']:.4f}")
            print(f"  Final Cosine: {stats['final_cosine']:.4f}")
            print(f"  Range: [{stats['min_cosine']:.4f}, {stats['max_cosine']:.4f}]")
            print(f"  Measurements: {stats['total_measurements']}")
        
        print(f"\n📊 Summary saved to {summary_path}")


def unfreeze_all_parameters(model):
    """Unfreeze tất cả parameters của model"""
    total_params = 0
    unfrozen_params = 0
    
    for name, param in model.named_parameters():
        total_params += 1
        if not param.requires_grad:
            param.requires_grad = True
            unfrozen_params += 1

        if "vision_transformer" in name:
            param.requires_grad = False
    
    print(f"🔓 Unfrozen {unfrozen_params} out of {total_params} parameters")
    print(f"📊 Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model


def main():
    """Main function để chạy cosine analysis"""
    
    # Load config
    config_name = 'vqa_vast'
    _config = copy.deepcopy(config_dict[config_name])
    
    # Modify config for analysis
    _config["max_steps"] = 1000 # Reduce steps for analysis
    _config["val_check_interval"] = 200
    _config["log_every_n_steps"] = 1
    
    # Configure for 1 GPU setup
    _config["num_gpus"] = 1
    _config["num_nodes"] = 1
    _config["batch_size"] = 64
    _config["per_gpu_batchsize"] = 64
    
    print(f"🚀 Starting Cosine Similarity Analysis with config: {config_name}")
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
    
    # Create cosine analysis callback
    cosine_callback = CosineAnalysisCallback(
        save_dir=f"cosine_analysis_{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Setup strategy for single GPU
    strategy = "auto"
    
    trainer = pl.Trainer(
        accelerator="gpu" if _config.get("num_gpus", 0) > 0 else "cpu",
        devices=_config.get("num_gpus", 1),
        max_steps=_config["max_steps"],
        callbacks=[cosine_callback],
        enable_model_summary=True,
        deterministic=True,
        precision=_config.get("precision", 32),
        strategy=strategy,
        enable_progress_bar=True,
        logger=False  # Disable all logging
    )
    
    print(f"\n🎯 Training model for cosine analysis...")
    print(f"📊 Results will be saved to: {cosine_callback.save_dir}")
    
    # Start training
    trainer.fit(model, datamodule=dm)
    
    print(f"\n✅ Cosine analysis completed!")
    print(f"📊 Check results in: {cosine_callback.save_dir}")
    print(f"📈 Run analyze_results.py to process the cosine data")


if __name__ == '__main__':
    main() 