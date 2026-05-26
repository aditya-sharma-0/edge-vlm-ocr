"""
01_explore_dataset.py — Understand Your Data Before You Touch It

PURPOSE:
    Parse raw ICDAR 2015 ground truth files, extract quadrilateral
    coordinates + transcriptions, and print statistics so we know
    exactly what we're working with.

WHAT YOU'LL LEARN:
    1. How ICDAR 2015 annotation files are structured
    2. Three encoding/parsing traps that silently break pipelines
    3. The difference between axis-aligned boxes and quadrilaterals
    4. What your actual data distribution looks like

RUN:
    cd "Summer Project"
    .venv/bin/python scripts/01_explore_dataset.py
"""

import os
from pathlib import Path
from PIL import Image


# =============================================================================
# STEP 1: Define where the data lives
# =============================================================================
# We use pathlib.Path instead of string concatenation. Why?
# - It handles OS differences (/ vs \) automatically
# - It has useful methods like .glob(), .stem, .exists()
# - It's the modern Python way. Raw string paths are fragile.

DATA_DIR = Path("data/icdar2015")
TRAIN_IMG_DIR = DATA_DIR / "train_images"
TRAIN_GT_DIR = DATA_DIR / "train_gt"


# =============================================================================
# STEP 2: Parse a single annotation file
# =============================================================================
# This is the core function. Let's understand every line.
#
# A raw ICDAR 2015 GT line looks like:
#     377,117,463,117,465,130,378,130,Genaxis Theatre
#
# That's: x1,y1,x2,y2,x3,y3,x4,y4,transcription
#
# THREE TRAPS you need to know about:
#
# TRAP 1: UTF-8 BOM (Byte Order Mark)
#   The files start with bytes EF BB BF (invisible \ufeff character).
#   If you don't handle this, the first coordinate of the first line
#   becomes "\ufeff377" instead of "377", and int() will crash.
#   Fix: open with encoding="utf-8-sig" (the "sig" strips the BOM).
#
# TRAP 2: Transcription can contain commas
#   If the text says "Hello, World", a naive line.split(",") gives
#   you 10 fields instead of 9. The trick: split on comma, take the
#   first 8 as coordinates, and rejoin everything else as transcription.
#
# TRAP 3: Windows line endings (\r\n)
#   The files may have \r at the end of each line.
#   Fix: .strip() removes both \r and \n.

def parse_gt_file(gt_path: Path) -> list[dict]:
    """
    Parse a single ICDAR 2015 ground truth file.

    Args:
        gt_path: Path to a gt_img_XXX.txt file

    Returns:
        List of dicts, each with:
            - 'quad': list of 8 ints [x1,y1,x2,y2,x3,y3,x4,y4]
            - 'transcription': the text string
            - 'is_legible': False if transcription is "###"
    """
    annotations = []

    # encoding="utf-8-sig" handles TRAP 1 (BOM stripping)
    with open(gt_path, encoding="utf-8-sig") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()  # handles TRAP 3 (\r\n)

            if not line:
                continue  # skip empty lines

            # Split on comma — handles TRAP 2
            parts = line.split(",")

            # We need at least 9 parts: 8 coordinates + 1 transcription
            if len(parts) < 9:
                print(f"  WARNING: {gt_path.name}:{line_num} — "
                      f"expected >=9 fields, got {len(parts)}: {line[:60]}")
                continue

            # First 8 fields are coordinates
            try:
                coords = [int(p) for p in parts[:8]]
            except ValueError as e:
                print(f"  WARNING: {gt_path.name}:{line_num} — "
                      f"bad coordinate: {e}")
                continue

            # Everything after the 8th comma is the transcription
            # We rejoin with comma in case the text itself had commas
            transcription = ",".join(parts[8:])

            annotations.append({
                "quad": coords,
                "transcription": transcription,
                "is_legible": transcription != "###",
            })

    return annotations


# =============================================================================
# STEP 3: Compute geometric properties of a quadrilateral
# =============================================================================
# This function helps us understand our data distribution.
# For each quad, we compute:
#   - The axis-aligned bounding box (what DeepSeek uses)
#   - The area of that box
#   - Whether the text is significantly rotated
#
# WHY THIS MATTERS:
# If most text is nearly horizontal, the axis-aligned box is fine.
# If lots of text is rotated 20°+, that's where our quad approach wins.

def analyze_quad(quad: list[int]) -> dict:
    """
    Given 8 coordinates [x1,y1,x2,y2,x3,y3,x4,y4], compute geometric stats.
    """
    # Unpack the 4 corner points
    x1, y1 = quad[0], quad[1]
    x2, y2 = quad[2], quad[3]
    x3, y3 = quad[4], quad[5]
    x4, y4 = quad[6], quad[7]

    xs = [x1, x2, x3, x4]
    ys = [y1, y2, y3, y4]

    # Axis-aligned bounding box (the DeepSeek approach)
    aabb_xmin, aabb_xmax = min(xs), max(xs)
    aabb_ymin, aabb_ymax = min(ys), max(ys)
    aabb_width = aabb_xmax - aabb_xmin
    aabb_height = aabb_ymax - aabb_ymin
    aabb_area = aabb_width * aabb_height

    # Quad area using the Shoelace formula
    # This is the actual area of the quadrilateral polygon.
    # The Shoelace formula: for vertices (x1,y1)...(xn,yn), the area is:
    #   0.5 * |sum(xi*y(i+1) - x(i+1)*yi)|
    # We need this to compute how much area the axis-aligned box wastes.
    points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    shoelace_sum = 0
    n = len(points)
    for i in range(n):
        j = (i + 1) % n
        shoelace_sum += points[i][0] * points[j][1]
        shoelace_sum -= points[j][0] * points[i][1]
    quad_area = abs(shoelace_sum) / 2.0

    # Area ratio: how much of the axis-aligned box is actually the text?
    # 1.0 = perfect (text is horizontal, box fits perfectly)
    # 0.5 = terrible (text is at 45°, box is 2x larger than needed)
    area_ratio = quad_area / aabb_area if aabb_area > 0 else 0

    # Is the top edge horizontal? Check if y1 ≈ y2
    # If the text is truly horizontal, the top two y-coords are nearly equal
    top_edge_dy = abs(y1 - y2)
    is_rotated = top_edge_dy > 5  # more than 5px difference = rotated

    return {
        "aabb": [aabb_xmin, aabb_ymin, aabb_xmax, aabb_ymax],
        "aabb_area": aabb_area,
        "quad_area": quad_area,
        "area_ratio": area_ratio,
        "is_rotated": is_rotated,
        "top_edge_dy": top_edge_dy,
    }


# =============================================================================
# STEP 4: Run the full analysis
# =============================================================================

def main():
    print("=" * 60)
    print("ICDAR 2015 Dataset Explorer")
    print("=" * 60)

    # Verify the data exists
    if not TRAIN_GT_DIR.exists():
        print(f"\nERROR: {TRAIN_GT_DIR} does not exist!")
        print("Make sure you extracted the ICDAR 2015 data into data/icdar2015/")
        return

    # Get all GT files, sorted
    gt_files = sorted(TRAIN_GT_DIR.glob("gt_img_*.txt"))
    print(f"\nFound {len(gt_files)} ground truth files")

    # Parse all annotations
    total_annotations = 0
    total_legible = 0
    total_illegible = 0
    total_rotated = 0
    area_ratios = []
    all_transcriptions = []
    images_with_no_legible = 0

    for gt_file in gt_files:
        annotations = parse_gt_file(gt_file)
        legible = [a for a in annotations if a["is_legible"]]
        illegible = [a for a in annotations if not a["is_legible"]]

        total_annotations += len(annotations)
        total_legible += len(legible)
        total_illegible += len(illegible)

        if len(legible) == 0:
            images_with_no_legible += 1

        for ann in legible:
            stats = analyze_quad(ann["quad"])
            area_ratios.append(stats["area_ratio"])
            if stats["is_rotated"]:
                total_rotated += 1
            all_transcriptions.append(ann["transcription"])

    # =========================================================================
    # STEP 5: Print the results
    # =========================================================================
    print(f"\n{'─' * 60}")
    print("ANNOTATION STATISTICS")
    print(f"{'─' * 60}")
    print(f"  Total annotations:      {total_annotations}")
    print(f"  Legible (usable):       {total_legible}")
    print(f"  Illegible (###):        {total_illegible}")
    print(f"  Illegible ratio:        {total_illegible/total_annotations:.1%}")
    print(f"  Images with no text:    {images_with_no_legible}")

    print(f"\n{'─' * 60}")
    print("ROTATION ANALYSIS (this is why quads matter)")
    print(f"{'─' * 60}")
    print(f"  Rotated text instances:  {total_rotated} / {total_legible} "
          f"({total_rotated/total_legible:.1%})")
    print(f"  Horizontal text:         {total_legible - total_rotated}")

    if area_ratios:
        avg_ratio = sum(area_ratios) / len(area_ratios)
        below_90 = sum(1 for r in area_ratios if r < 0.90)
        below_80 = sum(1 for r in area_ratios if r < 0.80)
        below_70 = sum(1 for r in area_ratios if r < 0.70)

        print(f"\n{'─' * 60}")
        print("AREA EFFICIENCY (quad area / axis-aligned box area)")
        print(f"{'─' * 60}")
        print(f"  Average ratio:           {avg_ratio:.3f}")
        print(f"  (1.0 = perfect fit, 0.5 = box is 2x too large)")
        print(f"  Instances < 90% fit:     {below_90} ({below_90/len(area_ratios):.1%})")
        print(f"  Instances < 80% fit:     {below_80} ({below_80/len(area_ratios):.1%})")
        print(f"  Instances < 70% fit:     {below_70} ({below_70/len(area_ratios):.1%})")

    # Transcription length stats
    if all_transcriptions:
        lengths = [len(t) for t in all_transcriptions]
        print(f"\n{'─' * 60}")
        print("TRANSCRIPTION STATISTICS")
        print(f"{'─' * 60}")
        print(f"  Total unique words:      {len(set(all_transcriptions))}")
        print(f"  Avg word length:         {sum(lengths)/len(lengths):.1f} chars")
        print(f"  Shortest:                {min(lengths)} chars")
        print(f"  Longest:                 {max(lengths)} chars")

        # Show some example transcriptions
        print(f"\n  Sample transcriptions:")
        seen = set()
        count = 0
        for t in all_transcriptions:
            if t not in seen and len(t) > 2:
                print(f"    • \"{t}\"")
                seen.add(t)
                count += 1
                if count >= 15:
                    break

    # Check image sizes
    print(f"\n{'─' * 60}")
    print("IMAGE SIZE DISTRIBUTION")
    print(f"{'─' * 60}")
    img_files = sorted(TRAIN_IMG_DIR.glob("img_*.jpg"))
    if img_files:
        widths, heights = [], []
        for img_path in img_files[:50]:  # sample 50 images for speed
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
        print(f"  Sampled {len(widths)} images:")
        print(f"  Width range:   {min(widths)} — {max(widths)} px")
        print(f"  Height range:  {min(heights)} — {max(heights)} px")
        print(f"  Most common:   ~{max(set(widths), key=widths.count)}x"
              f"{max(set(heights), key=heights.count)}")
    else:
        print(f"  No images found in {TRAIN_IMG_DIR}")

    print(f"\n{'═' * 60}")
    print("KEY TAKEAWAYS FOR YOUR PIPELINE:")
    print(f"{'═' * 60}")
    if area_ratios:
        avg_ratio = sum(area_ratios) / len(area_ratios)
        print(f"  • {total_legible} usable text instances across {len(gt_files)} images")
        print(f"  • {total_rotated/total_legible:.0%} of text is rotated "
              f"→ quad representation will help")
        print(f"  • Avg axis-aligned box wastes {(1-avg_ratio)*100:.1f}% of its area")
        print(f"  • {total_illegible} illegible entries to filter out")


if __name__ == "__main__":
    main()
