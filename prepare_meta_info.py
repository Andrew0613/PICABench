#!/usr/bin/env python3
"""
Generate standard meta_info.json from HuggingFace PICABench dataset + model output images.

Solves the problem:
- Users get a Dataset object from load_dataset but don't know how to organize it into the JSON format required by evaluation scripts
- Automatically saves input images to filesystem, maps output image paths, and generates complete metadata

Usage:
  pip install datasets pillow tqdm
  
  # Basic usage: assuming output images are in outputs/ directory with filenames matching dataset index
  python prepare_meta_info.py --output_image_dir outputs --save_dir PICABench_data
  
  # Specify output image naming pattern (if not named by index)
  python prepare_meta_info.py --output_image_dir my_results --save_dir data \
    --output_name_pattern "{index:05d}.png"

Output:
  - save_dir/input_img/        # Input images (saved from HF dataset)
  - save_dir/meta_info.json    # Standard format JSON, ready for evaluation
"""

import argparse
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def find_output_image(output_dir: Path, index: int, pattern: str) -> Optional[str]:
    """
    Find output image based on index and naming pattern.
    pattern supports {index} placeholder, e.g. "{index:05d}.jpg"
    """
    # Try user-specified pattern
    filename = pattern.format(index=index)
    candidate = output_dir / filename
    if candidate.exists():
        return filename
    
    # Try common extensions
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        for fmt in [f"{index:05d}{ext}", f"{index:04d}{ext}", f"{index}{ext}"]:
            candidate = output_dir / fmt
            if candidate.exists():
                return fmt
    
    return None


def save_input_image(img: Image.Image, path: Path) -> None:
    """Save input image to specified path"""
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(path, quality=95)


def build_meta_item(
    idx: int,
    example: Dict[str, Any],
    input_filename: str,
    output_filename: Optional[str],
    output_dir_for_json: str,
) -> Dict[str, Any]:
    """Build a single meta_info record"""
    if output_filename:
        output_path = (
            str(PurePosixPath(output_dir_for_json) / output_filename)
            if output_dir_for_json
            else output_filename
        )
    else:
        output_path = None

    item = {
        "index": idx,
        "input_path": f"input_img/{input_filename}",
        "output_path": output_path,
        "edit_instruction": example.get("edit_instruction", ""),
        "physics_category": example.get("physics_category", "unknown"),
        "physics_law": example.get("physics_law", "unknown"),
        "edit_operation": example.get("edit_operation", "unknown"),
        "difficulty": example.get("difficulty", "unknown"),
        "annotated_qa_pairs": example.get("annotated_qa_pairs", []),
        "edit_area": example.get("edit_area", "unknown"),
    }
    return item


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate meta_info.json from HF PICABench dataset + output image directory"
    )
    parser.add_argument(
        "--hf_repo",
        type=str,
        default="PICABench",
        help="HuggingFace dataset repository name",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="picabench",
        help="Dataset split name",
    )
    parser.add_argument(
        "--output_image_dir",
        type=str,
        required=True,
        help="Directory containing model-generated output images (relative to save_dir or absolute path)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Root directory to save meta_info.json and input_img/",
    )
    parser.add_argument(
        "--output_name_pattern",
        type=str,
        default="{index:05d}.jpg",
        help="Output image filename pattern, supports {index} placeholder, default {index:05d}.jpg",
    )
    parser.add_argument(
        "--allow_missing",
        action="store_true",
        help="Allow missing output images, still generate JSON (output_path will be null)",
    )
    parser.add_argument(
        "--force_input_save",
        action="store_true",
        help="Overwrite input images even if they already exist under save_dir/input_img",
    )
    args = parser.parse_args()

    save_dir = Path(args.save_dir).resolve()
    input_dir = save_dir / "input_img"
    input_dir.mkdir(parents=True, exist_ok=True)

    # Output image directory: supports relative path (relative to save_dir) or absolute path
    output_dir_arg = Path(args.output_image_dir)
    output_dir = output_dir_arg if output_dir_arg.is_absolute() else save_dir / output_dir_arg
    output_dir = output_dir.resolve()

    try:
        output_dir_for_json = output_dir.relative_to(save_dir).as_posix()
    except ValueError:
        output_dir_for_json = output_dir.as_posix()

    if output_dir_for_json in ("", "."):
        output_dir_for_json = ""

    print(f"Loading dataset: {args.hf_repo} (split={args.split})")
    dataset = load_dataset(args.hf_repo, split=args.split)
    print(f"Dataset size: {len(dataset)}")

    meta_info: List[Dict[str, Any]] = []
    missing_outputs: List[int] = []

    for idx, example in tqdm(enumerate(dataset), total=len(dataset), desc="Processing samples"):
        # 1. Save input image
        input_img = example.get("input_image")
        if input_img is None:
            # Fallback: try loading from image_path
            img_path = example.get("image_path")
            if img_path and os.path.exists(img_path):
                input_img = Image.open(img_path)
            else:
                print(f"Warning: sample {idx} has no input image, skipping")
                continue

        input_filename = f"{idx:05d}.jpg"
        input_path = input_dir / input_filename
        if input_path.exists():
            if args.force_input_save:
                save_input_image(input_img, input_path)
        else:
            save_input_image(input_img, input_path)

        # 2. Find corresponding output image
        output_filename = find_output_image(output_dir, idx, args.output_name_pattern)
        if output_filename is None:
            missing_outputs.append(idx)
            if not args.allow_missing:
                continue  # Skip samples without output images

        # 3. Build meta_info entry
        item = build_meta_item(idx, example, input_filename, output_filename, output_dir_for_json)
        meta_info.append(item)

    # Save JSON
    json_path = save_dir / "meta_info.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Successfully generated meta_info.json: {json_path}")
    print(f"  - Total samples: {len(meta_info)}")
    print(f"  - Input images saved to: {input_dir}")
    
    if missing_outputs:
        print(f"\n⚠ Warning: {len(missing_outputs)} samples missing output images")
        if len(missing_outputs) <= 10:
            print(f"  Missing indices: {missing_outputs}")
        else:
            print(f"  Missing indices (first 10): {missing_outputs[:10]}")
        
        if not args.allow_missing:
            print("  These samples are excluded from JSON")
        else:
            print("  These samples have output_path set to null")


if __name__ == "__main__":
    main()
