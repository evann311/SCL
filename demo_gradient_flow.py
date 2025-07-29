import os
import sys
import copy
import torch
import argparse
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scl.config import config_dict
from scl.datamodules.multitask_datamodule import MTDataModule
from scl.modules import SCLTransformer


def visualize_gradient_flow(model, save_path="gradient_flow_visualization.png"):
    """
    Visualize gradient flow across different model components
    """
    # Collect gradients by module type
    module_gradients = {
        'image_encoder': [],
        'text_encoder': [],
        'cross_modal': [],
        'vqa_head': []
    }
    
    # Collect gradient norms
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            
            if 'vision_transformer' in name:
                module_gradients['image_encoder'].append(grad_norm)
            elif 'text_transformer' in name:
                module_gradients['text_encoder'].append(grad_norm)
            elif 'cross_modal' in name:
                module_gradients['cross_modal'].append(grad_norm)
            elif 'vqa_classifier' in name:
                module_gradients['vqa_head'].append(grad_norm)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Box plot for gradient distribution
    data_to_plot = []
    labels = []
    for module_name, grads in module_gradients.items():
        if grads:
            data_to_plot.append(grads)
            labels.append(module_name.replace('_', ' ').title())
    
    if data_to_plot:
        ax1.boxplot(data_to_plot, labels=labels)
        ax1.set_ylabel('Gradient Norm')
        ax1.set_title('Gradient Distribution by Module Type')
        ax1.set_yscale('log')
        
        # Bar plot for mean gradients
        means = [np.mean(grads) for grads in data_to_plot]
        x_pos = np.arange(len(labels))
        
        ax2.bar(x_pos, means, alpha=0.7, color=['blue', 'green', 'orange', 'red'])
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.set_ylabel('Mean Gradient Norm')
        ax2.set_title('Average Gradient Magnitude by Module')
        ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Gradient flow visualization saved to {save_path}")
    plt.close()


# Custom callback to visualize gradient flow
class GradientFlowCallback(pl.Callback):
    def __init__(self, log_dir, visualize_every_n_steps=100):
        self.log_dir = log_dir
        self.visualize_every_n_steps = visualize_every_n_steps
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.visualize_every_n_steps == 0 and trainer.global_step > 0:
            save_path = os.path.join(
                self.log_dir, 
                f'gradient_flow_step_{trainer.global_step}.png'
            )
            visualize_gradient_flow(pl_module, save_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Gradient Flow Demonstration for VQA')
    parser.add_argument("--task", type=str, default='vqa_vast', 
                       help='Task configuration to use (vqa, vqa_kg, vqa_vast)')
    parser.add_argument("--max_steps", type=int, default=200,
                       help='Maximum training steps for demo')
    parser.add_argument("--gradient_log_interval", type=int, default=50,
                       help='Log gradient statistics every N steps') 
    parser.add_argument("--visualization_interval", type=int, default=100,
                       help='Create gradient visualization every N steps')
    parser.add_argument("--demo_log_dir", type=str, default="gradient_flow_demo",
                       help='Directory to save demo logs and visualizations')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load configuration based on task
    _config = copy.deepcopy(config_dict[args.task])
    
    # Override config for demo purposes
    _config['max_steps'] = args.max_steps
    _config['val_check_interval'] = args.gradient_log_interval
    _config['log_dir'] = args.demo_log_dir
    _config['fast_dev_run'] = False
    _config['test_only'] = False
    
    # Điều chỉnh batch size cho demo nếu cần
    if args.task == 'vqa_vast':
        _config['batch_size'] = 16
        _config['per_gpu_batchsize'] = min(_config['per_gpu_batchsize'], 16)  # Giảm batch size cho demo
    
    pl.seed_everything(_config["seed"])
    
    # Create datamodule and model
    dm = MTDataModule(_config, dist=False)  # Disable distributed for demo
    model = SCLTransformer(_config)
    
    exp_name = f'gradient_demo_{_config["exp_name"]}'
    
    os.makedirs(_config["log_dir"], exist_ok=True)
    
    # Model save setting
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(_config["log_dir"], 'checkpoints'),
        filename='gradient-demo-{epoch:02d}-{step}',
        save_top_k=1,
        verbose=True,
        monitor='vqa/val/loss' if 'vqa' in args.task else 'val/the_metric',
        mode='min' if 'vqa' in args.task else 'max',
        save_last=True
    )
    
    # Logger
    logger = TensorBoardLogger(
        _config["log_dir"],
        name=f'{exp_name}_seed{_config["seed"]}',
    )
    
    # Learning rate monitor
    lr_callback = LearningRateMonitor(logging_interval="step")
    
    # Gradient flow callback
    gradient_callback = GradientFlowCallback(
        log_dir=_config["log_dir"],
        visualize_every_n_steps=args.visualization_interval
    )
    
    callbacks = [checkpoint_callback, lr_callback, gradient_callback]
    
    # Calculate gradient accumulation steps
    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )
    
    grad_steps = _config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    )
    
    # Create trainer
    trainer_kwargs = {
        "accelerator": "gpu" if _config.get("num_gpus", 0) > 0 else "cpu",
        "devices": _config.get("num_gpus", 1),
        "num_nodes": _config["num_nodes"],
        "precision": _config["precision"],
        "benchmark": True,
        "deterministic": True,
        "max_steps": _config["max_steps"],
        "callbacks": callbacks,
        "logger": logger,
        "accumulate_grad_batches": grad_steps,
        "enable_model_summary": True,
        "fast_dev_run": _config["fast_dev_run"],
        "val_check_interval": _config["val_check_interval"],
        "log_every_n_steps": 10,
        "gradient_clip_val": 1.0,  # Clip gradients để ổn định
    }
    
    # Only add strategy if using multiple GPUs
    if _config.get("num_gpus", 1) > 1:
        trainer_kwargs["strategy"] = DDPStrategy(find_unused_parameters=True)
    
    trainer = Trainer(**trainer_kwargs)
    
    # Start training
    print("\n" + "="*80)
    print("GRADIENT FLOW DEMONSTRATION FOR VQA")
    print("="*80)
    print(f"Task: {args.task}")
    print(f"Training for {_config['max_steps']} steps")
    print(f"Gradient analysis every {args.gradient_log_interval} steps")
    print(f"Visualization every {args.visualization_interval} steps")
    print(f"Results will be saved to: {_config['log_dir']}")
    print("="*80 + "\n")
    
    print("EXPECTED RESULTS:")
    print("- Image Encoder (CLIP): Should receive minimal gradients")
    print("- Text Encoder: Should receive moderate gradients") 
    print("- Cross-Modal layers: Should receive significant gradients")
    print("- VQA Head: Should receive the largest gradients")
    print("- Text pathway should dominate gradient flow\n")
    
    trainer.fit(model, datamodule=dm, ckpt_path=_config.get("resume_from", None))
    
    print("\n" + "="*80)
    print("GRADIENT FLOW DEMONSTRATION COMPLETED")
    print("="*80)
    print(f"Check {_config['log_dir']} for:")
    print("- Gradient flow visualizations (PNG files)")
    print("- Tensorboard logs with gradient statistics")
    print("- Model checkpoints")
    print("\nTo view tensorboard logs:")
    print(f"  tensorboard --logdir {_config['log_dir']}")
    print("\nTo analyze results:")
    print(f"  python analyze_gradient_results.py --log_dir {_config['log_dir']}")
    print("="*80)


if __name__ == "__main__":
    main() 