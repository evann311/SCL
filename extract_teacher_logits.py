import torch
import numpy as np
import json
import os
import argparse
import functools
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from scl.config import config_dict
from scl.modules import SCLTransformer
from scl.datamodules import _datamodules


def extract_teacher_logits(config_name, output_dir="./teacher_outputs"):
    """
    Extract logits and question IDs from teacher model for knowledge distillation
    
    Args:
        config_name: Name of config from config_dict (e.g., 'vqa_distill')
        output_dir: Directory to save extracted logits and question IDs
    """
    
    # Load config
    config = config_dict[config_name]
    config['test_only'] = True
    
    # Get checkpoint path from config
    checkpoint_path = config.get('resume_from', None)
    if not checkpoint_path:
        raise ValueError("resume_from not found in config. Please set resume_from in your config.")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model
    model = SCLTransformer(config)
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint from {checkpoint_path}")
    
    # Set model to eval mode
    model.eval()
    
    # Initialize datamodule for VQA
    dm = _datamodules["vqa"](config)
    dm.setup("fit")
    
    # Use train dataset for extracting teacher logits - create custom dataloader with mlm_collator
    train_loader = DataLoader(
        dm.train_dataset,
        batch_size=dm.batch_size,
        shuffle=False,  # Don't need shuffle for extraction
        drop_last=False,
        num_workers=dm.num_workers,
        pin_memory=True,
        collate_fn=functools.partial(
            dm.train_dataset.collate,
            mlm_collator=dm.mlm_collator
        ),
    )
    
    # Storage for logits and metadata
    all_logits = []
    all_qids = []
    save_counter = 0
    
    print("Extracting teacher logits from training set...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            try:
                # Move batch to device
                if torch.cuda.is_available():
                    model = model.cuda()
                    for key in batch:
                        if isinstance(batch[key], torch.Tensor):
                            batch[key] = batch[key].cuda()
                        elif isinstance(batch[key], list):
                            if len(batch[key]) > 0 and isinstance(batch[key][0], torch.Tensor):
                                batch[key] = [x.cuda() for x in batch[key]]
                
                # Forward pass to get features
                infer = model.infer(batch, mask_text=False)
                
                # Get VQA raw logits from teacher (no softmax needed)
                teacher_logits = model.vqa_classifier(infer["cls_feats"])
                
                # Store raw logits (preferred for knowledge distillation)
                all_logits.append(teacher_logits.detach().cpu().numpy())
                
                # Extract question IDs if available
                if 'qid' in batch:
                    all_qids.extend(batch['qid'])
                else:
                    raise ValueError("qid not found in batch")
                
                # Save every 100 batches
                if (batch_idx + 1) % 100 == 0:
                    # Concatenate current batch of logits
                    batch_logits = np.concatenate(all_logits, axis=0)
                    print(f"Saving chunk {save_counter}, logits shape: {batch_logits.shape}")
                    
                    # Combine logits and qids into samples for this chunk
                    samples = []
                    for i, (qid, logits) in enumerate(zip(all_qids, batch_logits)):
                        samples.append({
                            "qid": qid,
                            "logits": logits.tolist()
                        })
                    
                    # Save chunk file
                    chunk_path = os.path.join(output_dir, f"teacher_samples_train_chunk_{save_counter:04d}.json")
                    with open(chunk_path, 'w') as f:
                        json.dump(samples, f, indent=2)
                    print(f"Saved chunk {save_counter} to {chunk_path}")
                    
                    # Reset for next chunk
                    all_logits = []
                    all_qids = []
                    save_counter += 1
                    
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue
    
    # Save remaining data if any
    if all_logits:
        batch_logits = np.concatenate(all_logits, axis=0)
        print(f"Saving final chunk {save_counter}, logits shape: {batch_logits.shape}")
        
        # Combine logits and qids into samples for final chunk
        samples = []
        for i, (qid, logits) in enumerate(zip(all_qids, batch_logits)):
            samples.append({
                "qid": qid,
                "logits": logits.tolist()
            })
        
        # Save final chunk file
        chunk_path = os.path.join(output_dir, f"teacher_samples_train_chunk_{save_counter:04d}.json")
        with open(chunk_path, 'w') as f:
            json.dump(samples, f, indent=2)
        print(f"Saved final chunk {save_counter} to {chunk_path}")
        save_counter += 1
    
    # Save metadata about all chunks
    metadata = {
        "total_chunks": save_counter,
        "samples_per_chunk": "~100 batches each",
        "config_name": config_name,
        "checkpoint_path": checkpoint_path,
        "dataset": "vqa_train"
    }
    
    metadata_path = os.path.join(output_dir, "extraction_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")
    
    print(f"\nSuccessfully extracted teacher outputs:")
    print(f"- Total chunks: {save_counter}")
    print(f"- Files: teacher_samples_train_chunk_XXXX.json")
    print(f"- Metadata: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract teacher logits for knowledge distillation")
    parser.add_argument("--config", type=str, default="vqa_distill", 
                       help="Config name from config_dict")
    parser.add_argument("--output_dir", type=str, default="./teacher_outputs",
                       help="Directory to save extracted outputs")
    
    args = parser.parse_args()
    
    print(f"Extracting teacher logits...")
    print(f"Config: {args.config}")
    print(f"Output directory: {args.output_dir}")
    
    extract_teacher_logits(args.config, args.output_dir)


if __name__ == "__main__":
    main() 