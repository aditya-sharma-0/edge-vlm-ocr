"""
02_format_dataset.py — Building the Baseline (Axis-Aligned Boxes)

PURPOSE:
    Convert the raw ICDAR 2015 dataset into a HuggingFace-compatible JSONL format.
    Following the user's strategy:
    1. We start with the BASELINE: Axis-aligned boxes (4 coordinates).
    2. We use <ref> and <box> tags (consistent with SmolVLM).
    3. We normalize coordinates to [0, 999].
    4. We keep images clean (no degradations yet) to prove the pipeline works first.

OUTPUT FORMAT (JSONL):
    {"messages": [
        {"role": "user", "content": [{"type": "image", "image": "path/to/img.jpg"}, {"type": "text", "text": "Extract all text regions and their bounding boxes."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "<ref>TEXT</ref><box>[[x1,y1,x2,y2]]</box>..."}]}
    ]}

RUN:
    cd "Summer Project"
    .venv/bin/python scripts/02_format_dataset.py
"""

import json
from pathlib import Path
from PIL import Image

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path("data/icdar2015")
TRAIN_IMG_DIR = DATA_DIR / "train_images"
TRAIN_GT_DIR = DATA_DIR / "train_gt"
TEST_IMG_DIR = DATA_DIR / "test_images"
TEST_GT_DIR = DATA_DIR / "test_gt"

TRAIN_OUTPUT = DATA_DIR / "train.jsonl"
TEST_OUTPUT = DATA_DIR / "test.jsonl"

PROMPT_TEXT = "Extract all text regions and their bounding boxes."

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_gt_file(gt_path: Path) -> list[dict]:
    """Parse a single ICDAR 2015 ground truth file (from Script 01)."""
    annotations = []
    with open(gt_path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 9:
                continue
            
            try:
                coords = [int(p) for p in parts[:8]]
            except ValueError:
                continue
                
            transcription = ",".join(parts[8:])
            
            # We only want legible text for training
            if transcription != "###":
                annotations.append({
                    "quad": coords,
                    "transcription": transcription
                })
    return annotations

def normalize_bbox(quad: list[int], img_w: int, img_h: int) -> list[int]:
    """
    Convert 8-point quad to 4-point axis-aligned bounding box,
    then normalize to [0, 999] integer scale.
    """
    xs = [quad[0], quad[2], quad[4], quad[6]]
    ys = [quad[1], quad[3], quad[5], quad[7]]
    
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    
    # Clip to image boundaries just in case annotations go slightly out of bounds
    xmin = max(0, min(xmin, img_w - 1))
    xmax = max(0, min(xmax, img_w - 1))
    ymin = max(0, min(ymin, img_h - 1))
    ymax = max(0, min(ymax, img_h - 1))
    
    # Normalize to [0, 999]
    norm_xmin = int((xmin / img_w) * 999)
    norm_xmax = int((xmax / img_w) * 999)
    norm_ymin = int((ymin / img_h) * 999)
    norm_ymax = int((ymax / img_h) * 999)
    
    # Ensure min <= max (though they should be)
    return [norm_xmin, norm_ymin, norm_xmax, norm_ymax]

def process_split(img_dir: Path, gt_dir: Path, output_file: Path):
    """Process an entire split (train or test) and save to JSONL."""
    print(f"Processing {img_dir.name}...")
    
    if not img_dir.exists() or not gt_dir.exists():
        print(f"  WARNING: Directory not found. Skipping.")
        return

    img_files = sorted(img_dir.glob("img_*.jpg"))
    if not img_files:
        print(f"  WARNING: No images found in {img_dir}.")
        return

    processed_count = 0
    empty_count = 0
    
    with open(output_file, "w", encoding="utf-8") as out_f:
        for img_path in img_files:
            # Match image file to ground truth file (img_1.jpg -> gt_img_1.txt)
            img_num = img_path.stem.split("_")[1]
            gt_path = gt_dir / f"gt_img_{img_num}.txt"
            
            if not gt_path.exists():
                continue
                
            # Parse annotations
            annotations = parse_gt_file(gt_path)
            if not annotations:
                empty_count += 1
                continue # Skip images with no legible text
                
            # Get image dimensions for normalization
            try:
                with Image.open(img_path) as img:
                    img_w, img_h = img.size
            except Exception as e:
                print(f"  Error opening {img_path}: {e}")
                continue
                
            # Build the assistant's response string
            # Format: <ref>text</ref><box>[[xmin,ymin,xmax,ymax]]</box>
            response_parts = []
            for ann in annotations:
                bbox = normalize_bbox(ann["quad"], img_w, img_h)
                # Keep transcription clean of explicit xml tags to avoid breaking format
                clean_text = ann["transcription"].replace("<", "").replace(">", "")
                response_parts.append(f"<ref>{clean_text}</ref><box>[{bbox}]</box>")
            
            assistant_response = " ".join(response_parts)
            
            # Construct the HuggingFace conversation format
            record = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(img_path)},
                            {"type": "text", "text": PROMPT_TEXT}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": assistant_response}
                        ]
                    }
                ]
            }
            
            out_f.write(json.dumps(record) + "\n")
            processed_count += 1
            
    print(f"  Saved {processed_count} records to {output_file.name}")
    print(f"  Skipped {empty_count} images with no legible text")

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    print("=" * 60)
    print("Formatting Dataset for Baseline (Axis-Aligned)")
    print("=" * 60)
    
    process_split(TRAIN_IMG_DIR, TRAIN_GT_DIR, TRAIN_OUTPUT)
    process_split(TEST_IMG_DIR, TEST_GT_DIR, TEST_OUTPUT)
    
    print("\nDataset formatting complete!")
    print("You can now load this dataset using:")
    print("  from datasets import load_dataset, Image")
    print("  ds = load_dataset('json', data_files={'train': 'data/icdar2015/train.jsonl'})")
    print("  ds = ds.cast_column('messages', ... # custom cast depending on HF version)")

if __name__ == "__main__":
    main()
