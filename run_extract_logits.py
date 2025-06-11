#!/usr/bin/env python3
"""
Script to extract teacher logits for knowledge distillation
Usage: python run_extract_logits.py --checkpoint /path/to/teacher/checkpoint.ckpt
"""

import argparse
import os
from extract_teacher_logits import extract_teacher_logits


def main():
    parser = argparse.ArgumentParser(description="Extract teacher logits for VQA knowledge distillation")
    
    # Required arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to teacher model checkpoint (.ckpt file)")
    
    # Optional arguments
    parser.add_argument("--output_dir", type=str, default="./teacher_outputs",
                       help="Directory to save extracted logits and question IDs (default: ./teacher_outputs)")
    parser.add_argument("--config", type=str, default="vqa_distill",
                       help="Config name to use (default: vqa_distill)")
    
    args = parser.parse_args()
    
    # Validate checkpoint path
    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint file not found: {args.checkpoint}")
        return
    
    if not args.checkpoint.endswith('.ckpt'):
        print(f"⚠️  Warning: Checkpoint file should end with .ckpt: {args.checkpoint}")
    
    print("🔍 Extracting teacher logits for knowledge distillation...")
    print(f"📂 Teacher checkpoint: {args.checkpoint}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"⚙️  Config: {args.config}")
    print("-" * 50)
    
    try:
        # Extract teacher logits
        extract_teacher_logits(
            config_name=args.config,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir
        )
        
        print("\n✅ Teacher logits extraction completed successfully!")
        print(f"📁 Check output directory: {args.output_dir}")
        print("\nFiles generated:")
        print("  - teacher_logits_train.npy: Teacher model logits")
        print("  - question_ids_train.json: Question IDs corresponding to logits")
        print("  - extraction_metadata.json: Metadata about the extraction")
        
    except Exception as e:
        print(f"\n❌ Error during extraction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 