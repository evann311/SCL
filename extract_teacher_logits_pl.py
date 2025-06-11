import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import BasePredictionWriter
import numpy as np
import json
import os
import argparse
from pathlib import Path
from typing import Optional, Any, List
import h5py
from tqdm import tqdm

from scl.config import config_dict
from scl.modules import SCLTransformer
from scl.datamodules import _datamodules


class TeacherLogitsWriter(BasePredictionWriter):
    """Custom callback để save logits theo chunks"""
    
    def __init__(self, output_dir: str, write_interval: str = "batch"):
        super().__init__(write_interval)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage cho batch outputs
        self.logits_chunks = []
        self.qids_chunks = []
        self.chunk_size = 10000  # Save mỗi 10k samples
        self.chunk_idx = 0
        
    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        prediction: Any,
        batch_indices: List[int],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Save predictions sau mỗi batch"""
        if prediction is not None:
            logits, qids = prediction
            self.logits_chunks.append(logits.cpu().numpy())
            self.qids_chunks.extend(qids)
            
            # Save chunk khi đủ samples
            if len(self.qids_chunks) >= self.chunk_size:
                self._save_chunk()
    
    def write_on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        predictions: List[Any],
        batch_indices: List[int],
    ) -> None:
        """Save remaining data cuối epoch"""
        if self.logits_chunks:
            self._save_chunk()
        
        # Merge all chunks thành file cuối
        self._merge_chunks()
    
    def _save_chunk(self):
        """Save một chunk data"""
        if not self.logits_chunks:
            return
            
        chunk_logits = np.concatenate(self.logits_chunks, axis=0)
        chunk_qids = np.array(self.qids_chunks, dtype=object)
        
        chunk_path = self.output_dir / f"chunk_{self.chunk_idx}.npz"
        np.savez_compressed(
            chunk_path,
            logits=chunk_logits,
            qids=chunk_qids
        )
        
        print(f"Saved chunk {self.chunk_idx}: {len(chunk_qids)} samples to {chunk_path}")
        
        # Reset cho chunk tiếp theo
        self.logits_chunks = []
        self.qids_chunks = []
        self.chunk_idx += 1
    
    def _merge_chunks(self):
        """Merge tất cả chunks thành file cuối"""
        chunk_files = list(self.output_dir.glob("chunk_*.npz"))
        if not chunk_files:
            return
        
        all_logits = []
        all_qids = []
        
        print("Merging chunks...")
        for chunk_file in sorted(chunk_files):
            data = np.load(chunk_file, allow_pickle=True)
            all_logits.append(data['logits'])
            all_qids.extend(data['qids'])
            
            # Cleanup chunk file
            chunk_file.unlink()
        
        # Save final file
        final_logits = np.concatenate(all_logits, axis=0)
        final_qids = np.array(all_qids, dtype=object)
        
        final_path = self.output_dir / "teacher_samples_train.npz"
        np.savez_compressed(
            final_path,
            logits=final_logits,
            qids=final_qids
        )
        
        print(f"Merged {len(chunk_files)} chunks -> {final_path}")
        print(f"Final shape: logits {final_logits.shape}, qids {len(final_qids)}")


class TeacherLogitsExtractor(pl.LightningModule):
    """PyTorch Lightning module để extract teacher logits hiệu quả"""
    
    def __init__(self, config: dict, teacher_checkpoint: str):
        super().__init__()
        
        # Load teacher model
        self.teacher = SCLTransformer(config)
        
        # Load checkpoint
        ckpt = torch.load(teacher_checkpoint, map_location="cpu")
        if 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt["state_dict"]
        
        self.teacher.load_state_dict(state_dict, strict=False)
        print(f"✅ Loaded teacher checkpoint from {teacher_checkpoint}")
        
        # Freeze teacher model
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        self.teacher.eval()
        
        # Save config for metadata
        self.config = config
        self.teacher_checkpoint = teacher_checkpoint
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> tuple:
        """Extract logits cho một batch"""
        with torch.no_grad():
            # Forward pass với teacher
            infer = self.teacher.infer(batch, mask_text=False)
            
            # Get VQA logits
            teacher_logits = self.teacher.vqa_classifier(infer["cls_feats"])
            
            # Extract question IDs
            qids = batch.get('qid', [f"unknown_{batch_idx}_{i}" for i in range(len(teacher_logits))])
            
            return teacher_logits.detach(), qids
    
    def configure_optimizers(self):
        """Không cần optimizer cho inference"""
        return None


def extract_teacher_logits_pl(
    config_name: str = "vqa_distill",
    output_dir: str = "./teacher_outputs_pl",
    accelerator: str = "auto",
    devices: str = "auto",
    compile_model: bool = True
):

    
    # Load config vqa_distill với optimizations
    config = config_dict[config_name].copy()
    config['test_only'] = True
    
    # Get checkpoint path từ resume_from (đã config sẵn)
    checkpoint_path = config.get('resume_from', None) or config.get('load_path', None)
    if not checkpoint_path:
        raise ValueError(f"Cần set resume_from hoặc load_path trong config {config_name}!")
    
    # Tối ưu hóa num_workers để tăng hiệu suất data loading
    config['num_workers'] = max(16, config.get('num_workers', 8))
    
    # Sử dụng precision từ config (đã là '16-mixed')
    precision = config.get('precision', '16-mixed')
    
    print(f"🚀 Extracting teacher logits with PyTorch Lightning...")
    print(f"   Config: {config_name}")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Output: {output_dir}")
    print(f"   Batch size: {config['per_gpu_batchsize']} (từ config)")
    print(f"   Num workers: {config['num_workers']} (tối ưu từ {config.get('num_workers', 8)})")
    print(f"   Precision: {precision} (từ config)")
    print(f"   Accelerator: {accelerator}")
    print(f"   Devices: {devices}")
    print(f"   Torch compile: {compile_model}")
    
    # Setup datamodule
    dm = _datamodules["vqa"](config)
    dm.setup("fit")
    
    # Tạo prediction dataloader từ train dataset
    predict_dataloader = dm.train_dataloader()
    print(f"   Dataset size: {len(dm.train_dataset)} samples")
    print(f"   Num batches: {len(predict_dataloader)}")
    
    # Initialize extractor model
    extractor = TeacherLogitsExtractor(config, checkpoint_path)
    
    # Compile model nếu PyTorch >= 2.0
    if compile_model and hasattr(torch, 'compile'):
        try:
            extractor.teacher = torch.compile(extractor.teacher, mode="reduce-overhead")
            print("✅ Model compiled with torch.compile")
        except Exception as e:
            print(f"⚠️ Torch compile failed: {e}")
    
    # Setup callbacks
    writer_callback = TeacherLogitsWriter(output_dir)
    
    # Setup trainer với optimizations
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        callbacks=[writer_callback],
        enable_progress_bar=True,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
        deterministic=False,  # Faster training
        benchmark=True,       # Optimize CUDNN
    )
    
    # Run prediction
    print("\n🔄 Starting logits extraction...")
    trainer.predict(
        extractor,
        dataloaders=predict_dataloader,
        return_predictions=False  # Sử dụng callback thay vì return
    )
    
    # Save metadata
    metadata = {
        "num_samples": len(dm.train_dataset),
        "config_name": config_name,
        "checkpoint_path": checkpoint_path,
        "dataset": "vqa_train",
        "format": "npz_compressed",
        "batch_size": config['per_gpu_batchsize'],
        "num_workers": config['num_workers'],
        "precision": precision,
        "accelerator": accelerator,
        "devices": str(devices),
        "torch_compile": compile_model
    }
    
    metadata_path = Path(output_dir) / "extraction_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Extraction completed!")
    print(f"   Metadata: {metadata_path}")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Extract teacher logits using PyTorch Lightning với config vqa_distill đã tối ưu sẵn"
    )
    parser.add_argument("--config", type=str, default="vqa_distill", 
                       help="Config name từ config_dict (mặc định: vqa_distill)")
    parser.add_argument("--output_dir", type=str, default="./teacher_outputs_pl",
                       help="Thư mục để save outputs")
    parser.add_argument("--accelerator", type=str, default="auto",
                       choices=["auto", "gpu", "cpu", "tpu"],
                       help="PyTorch Lightning accelerator")
    parser.add_argument("--devices", type=str, default="auto",
                       help="Số devices để dùng (mặc định: auto)")
    parser.add_argument("--no_compile", action="store_true",
                       help="Tắt torch.compile()")
    
    args = parser.parse_args()
    
    print("🔧 PyTorch Lightning Teacher Logits Extraction")
    print("🚀 Using optimized vqa_distill config:")
    print("   ✅ Batch size: 384 (từ config)")
    print("   ✅ Workers: 16 (tối ưu từ 8)")  
    print("   ✅ Precision: 16-mixed (từ config)")
    print("   ✅ Checkpoint: /workspace/checkpoints/last.ckpt (từ config)")
    print("=" * 60)
    
    # Extract logits
    extract_teacher_logits_pl(
        config_name=args.config,
        output_dir=args.output_dir,
        accelerator=args.accelerator,
        devices=args.devices,
        compile_model=not args.no_compile
    )


if __name__ == "__main__":
    main() 