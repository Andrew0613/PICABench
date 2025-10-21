import argparse
import json
import os
from typing import Dict, List, Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Non-edited Region Quality Assessment - PSNR only')
    parser.add_argument(
        '--meta_info_path',
        required=True,
        help='Path to meta_info.json file',
    )
    parser.add_argument(
        '--base_dir',
        required=False,
        help='Base directory for image files, defaults to meta_info.json directory if not specified',
    )
    parser.add_argument(
        '--size',
        default=None,
        type=int,
        help='Resize image to this size, keep original size if not specified',
    )

    return parser.parse_args()


def create_edit_mask(image_size, edit_areas):
    """Create edit region mask, returns non-edited region mask"""
    width, height = image_size
    # Create white image (all regions are non-edited)
    mask_img = Image.new('L', (width, height), 255)
    draw = ImageDraw.Draw(mask_img)
    
    # Mark edited regions as black (0)
    for area in edit_areas:
        x = area['x']
        y = area['y']
        w = area['width'] 
        h = area['height']
        # Draw rectangle, edited region in black
        draw.rectangle([x, y, x + w, y + h], fill=0)
    
    # Convert to tensor, non-edited region=1, edited region=0
    mask_array = np.array(mask_img) / 255.0  # Convert to [0,1]
    mask_tensor = torch.tensor(mask_array, dtype=torch.float32)
    return mask_tensor


def load_image_with_mask(image_path: str, edit_areas: List[Dict], size=None):
    """Load image and create non-edited region mask"""
    try:
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        if size:
            image = image.resize((size, size), Image.BILINEAR)
            # Adjust edit_areas coordinates
            scale_x = size / original_size[0]
            scale_y = size / original_size[1]
            scaled_edit_areas = []
            for area in edit_areas:
                scaled_area = {
                    'x': area['x'] * scale_x,
                    'y': area['y'] * scale_y,
                    'width': area['width'] * scale_x,
                    'height': area['height'] * scale_y
                }
                scaled_edit_areas.append(scaled_area)
            edit_areas = scaled_edit_areas
            mask_size = (size, size)
        else:
            mask_size = original_size
        
        # Create mask
        mask = create_edit_mask(mask_size, edit_areas)
        
        # Convert image to tensor
        image_array = np.array(image) / 255.0  # Normalize to [0, 1]
        image_array = image_array.astype(np.float32)
        image_tensor = torch.tensor(image_array).permute(2, 0, 1).unsqueeze(0)  # Shape: (1, 3, H, W)
        
        # Expand mask dimensions to match image (1, 1, H, W)
        mask = mask.unsqueeze(0).unsqueeze(0)
        
        return image_tensor, mask
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None, None


def compute_masked_psnr(output_img, input_img, mask):
    """Compute PSNR only on masked region (mask=1 pixels)"""
    mask_bool = mask > 0.5
    mask_3ch = mask_bool.expand_as(output_img)
    
    # Extract non-edited region pixels
    output_pixels = output_img[mask_3ch]
    input_pixels = input_img[mask_3ch]
    
    if output_pixels.numel() == 0:
        return None
    
    # Calculate MSE (only on non-edited region)
    mse = torch.mean((output_pixels - input_pixels) ** 2)
    
    if mse < 1e-10:
        return 100.0  # Near infinity, return a large value
    
    psnr = 20 * torch.log10(torch.tensor(1.0) / torch.sqrt(mse))
    return psnr.item()


def load_image_simple(image_path: str, size=None):
    """Simple image loading without mask processing"""
    try:
        image = Image.open(image_path).convert('RGB')
        if size:
            image = image.resize((size, size), Image.BILINEAR)
        image = np.array(image) / 255.0  # Normalize to [0, 1]
        image = image.astype(np.float32)
        image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)  # Shape: (1, 3, H, W)
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def compute_full_image_psnr(output_img, input_img):
    """Compute PSNR on full image"""
    mse = torch.mean((output_img - input_img) ** 2)
    
    if mse < 1e-10:
        return 100.0
    
    psnr = 20 * torch.log10(torch.tensor(1.0) / torch.sqrt(mse))
    return psnr.item()


def evaluate_single_item(item: Dict[str, Any], base_dir: str, size=None) -> float:
    """Evaluate single meta_info item and return PSNR"""
    # Build full paths
    input_path = os.path.join(base_dir, item['input_path'])
    output_path = os.path.join(base_dir, item['output_path'])
    
    # Check if files exist
    if not os.path.exists(input_path):
        print(f"Warning: input image not found: {input_path}")
        return None
    
    if not os.path.exists(output_path):
        print(f"Warning: output image not found: {output_path}")
        return None
    
    # Get edit area information
    edit_areas = item.get('edit_area', [])
    use_full_image = False
    
    # Handle edit_area as string
    if isinstance(edit_areas, str):
        if edit_areas == "unknown":
            use_full_image = True
        else:
            print(f"Warning: unexpected edit_area format for item {item.get('index', 'unknown')}: {edit_areas}")
            return None
    elif not edit_areas or len(edit_areas) == 0:
        # No edit_area info, use full image evaluation
        use_full_image = True
    
    if use_full_image:
        # Full image evaluation
        input_image = load_image_simple(input_path, size)
        output_image = load_image_simple(output_path, size)
        
        if input_image is None or output_image is None:
            return None
        
        # Ensure images have same size
        if input_image.shape != output_image.shape:
            h, w = input_image.shape[2], input_image.shape[3]
            output_image = torch.nn.functional.interpolate(output_image, size=(h, w), mode='bilinear', align_corners=False)
        
        # Compute full image PSNR
        psnr = compute_full_image_psnr(output_image, input_image)
        
    else:
        # Non-edited region evaluation
        input_image, mask = load_image_with_mask(input_path, edit_areas, size)
        output_image, _ = load_image_with_mask(output_path, edit_areas, size)
        
        if input_image is None or output_image is None or mask is None:
            return None
        
        # Ensure images have same size
        if input_image.shape != output_image.shape:
            h, w = input_image.shape[2], input_image.shape[3]
            output_image = torch.nn.functional.interpolate(output_image, size=(h, w), mode='bilinear', align_corners=False)
        
        # Compute masked region PSNR
        psnr = compute_masked_psnr(output_image, input_image, mask)
    
    return psnr


def process_meta_info(meta_info_path: str, base_dir: str, size=None):
    """Process entire meta_info.json file"""
    
    # Load meta_info.json
    with open(meta_info_path, 'r', encoding='utf-8') as f:
        meta_info = json.load(f)
    
    print(f"Loaded {len(meta_info)} items from {meta_info_path}")
    
    # Detect evaluation mode
    has_edit_area = False
    
    # Check first few items for edit_area status
    for item in meta_info[:10]:
        edit_areas = item.get('edit_area', [])
        if isinstance(edit_areas, list) and len(edit_areas) > 0:
            has_edit_area = True
            break
    
    if not has_edit_area:
        print("Warning: No valid edit_area found in dataset, using full image evaluation mode")
        evaluation_type = "full_image"
    else:
        evaluation_type = "non_edited_region"
    
    # Store results
    detailed_results = []
    valid_scores = []
    
    # Evaluate each item
    for idx, item in tqdm(enumerate(meta_info), desc="Evaluating items"):
        psnr = evaluate_single_item(item, base_dir, size)
        sample_id = item.get('index', idx)
        
        # Add to detailed results
        result_item = {
            'id': sample_id,
            'input_path': item['input_path'],
            'output_path': item['output_path'],
            'physics_category': item.get('physics_category', 'unknown'),
            'physics_law': item.get('physics_law', 'unknown'),
            'edit_operation': item.get('edit_operation', 'unknown'),
            'difficulty': item.get('difficulty', 'unknown'),
            'psnr': psnr
        }
        detailed_results.append(result_item)
        
        # Collect valid scores
        if psnr is not None:
            valid_scores.append(psnr)
    
    # Calculate overall statistics
    if valid_scores:
        overall_stats = {
            'count': len(valid_scores),
            'mean': float(np.mean(valid_scores)),
            'std': float(np.std(valid_scores)),
            'min': float(np.min(valid_scores)),
            'max': float(np.max(valid_scores)),
            'median': float(np.median(valid_scores))
        }
    else:
        overall_stats = {
            'count': 0,
            'mean': None,
            'std': None,
            'min': None,
            'max': None,
            'median': None
        }
    
    # Statistics by physics_category
    physics_category_stats = {}
    categories = set(item.get('physics_category', 'unknown') for item in meta_info)
    
    for category in categories:
        category_items = [item for item in detailed_results if item['physics_category'] == category]
        category_scores = [item['psnr'] for item in category_items if item['psnr'] is not None]
        
        if category_scores:
            physics_category_stats[category] = {
                'count': len(category_scores),
                'mean': float(np.mean(category_scores)),
                'std': float(np.std(category_scores))
            }
        else:
            physics_category_stats[category] = {
                'count': 0,
                'mean': None,
                'std': None
            }
    
    # Statistics by physics_law
    physics_law_stats = {}
    laws = set(item.get('physics_law', 'unknown') for item in meta_info)
    
    for law in laws:
        law_items = [item for item in detailed_results if item['physics_law'] == law]
        law_scores = [item['psnr'] for item in law_items if item['psnr'] is not None]
        
        if law_scores:
            physics_law_stats[law] = {
                'count': len(law_scores),
                'mean': float(np.mean(law_scores)),
                'std': float(np.std(law_scores))
            }
        else:
            physics_law_stats[law] = {
                'count': 0,
                'mean': None,
                'std': None
            }
    
    # Prepare analysis results
    final_analysis = {
        'meta_info_path': meta_info_path,
        'total_items': len(meta_info),
        'evaluation_type': evaluation_type,
        'overall_statistics': overall_stats,
        'physics_category_statistics': physics_category_stats,
        'physics_law_statistics': physics_law_stats
    }
    
    return final_analysis, detailed_results


def main():
    args = parse_args()
    
    # Determine base directory
    if args.base_dir:
        base_dir = args.base_dir
    else:
        base_dir = os.path.dirname(args.meta_info_path)
    
    print(f"Using base directory: {base_dir}")
    
    # Process meta_info
    analysis_results, detailed_results = process_meta_info(
        args.meta_info_path, 
        base_dir, 
        args.size
    )
    
    # Generate output file names
    meta_info_dir = os.path.dirname(args.meta_info_path)
    meta_info_name = os.path.splitext(os.path.basename(args.meta_info_path))[0]
    
    analysis_output_path = os.path.join(meta_info_dir, f"{meta_info_name}_psnr_analysis.json")
    detailed_output_path = os.path.join(meta_info_dir, f"{meta_info_name}_psnr_output.json")
    
    # Save results
    with open(analysis_output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    with open(detailed_output_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved:")
    print(f"Analysis: {analysis_output_path}")
    print(f"Detailed: {detailed_output_path}")
    
    # Print overall results
    eval_mode = analysis_results['evaluation_type']
    eval_label = "Full Image" if eval_mode == "full_image" else "Non-edited Region"
    print(f"\n=== Overall PSNR Results ({eval_label}) ===")
    stats = analysis_results['overall_statistics']
    if stats['mean'] is not None:
        print(f"PSNR: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})")
        print(f"  Min: {stats['min']:.4f}, Max: {stats['max']:.4f}, Median: {stats['median']:.4f}")
    else:
        print(f"PSNR: No valid scores")
    
    # Print physics_category results
    print(f"\n=== PSNR Results by Physics Category ===")
    for category, category_stats in analysis_results['physics_category_statistics'].items():
        if category_stats['mean'] is not None:
            print(f"{category}: {category_stats['mean']:.4f} ± {category_stats['std']:.4f} (n={category_stats['count']})")
        else:
            print(f"{category}: No valid scores")
    
    # Print physics_law results
    print(f"\n=== PSNR Results by Physics Law ===")
    for law, law_stats in analysis_results['physics_law_statistics'].items():
        if law_stats['mean'] is not None:
            print(f"{law}: {law_stats['mean']:.4f} ± {law_stats['std']:.4f} (n={law_stats['count']})")
        else:
            print(f"{law}: No valid scores")


if __name__ == "__main__":
    main()
