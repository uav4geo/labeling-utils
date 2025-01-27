import argparse
import sys
import glob
import os
from PIL import Image
import random

def main():
    parser = argparse.ArgumentParser(description="Convert DOTA annotations to YOLO")
    parser.add_argument(
        "input", 
        type=str, 
        help="Path to DOTA dataset"
    )
    parser.add_argument(
        "--classes",
        type=str,
        required=False,
        default=None,
        help="List of classes to keep/merge, comma separated (e.g. plane,small-vehicle)"
    )

    args = parser.parse_args()
    
    dir_sets = ["train", "test", "valid"]

    keep_cls = []
    if args.classes is not None:
        keep_cls = [c for c in args.classes.split(",") if c.strip() != ""]

    class_d = {}
    cur_cls = 0

    for ds in dir_sets:
        annotations_path = os.path.join(args.input, ds, "labels")
        ann_files = glob.glob(f"{annotations_path}/*.txt")
        ann_meta = {}

        for ann in ann_files:
            img_path = os.path.join(args.input, ds, "images", os.path.basename(ann).replace(".txt", "") + ".png")
            with Image.open(img_path) as img:
                ann_meta[ann] = {
                    'width': img.width,
                    'height': img.height,
                }

        for ann in ann_files:
            out_lines = []

            with open(ann, 'r') as f:
                lines = [l.strip() for l in f.read().split("\n") if l.strip() != ""]
                for line in lines:
                    parts = line.split(" ")
                    if len(parts) == 10:
                        x1, y1, x2, y2, x3, y3, x4, y4 = [float(v) for v in parts[:8]]
                        category = parts[8].lower()
                        if len(keep_cls) > 0 and not category in keep_cls:
                            continue
                        
                        difficult = parts[9] == "1"

                        if category not in class_d:
                            class_d[category] = cur_cls
                            cur_cls += 1
                        
                        cid = class_d[category]
                        
                        xmin = min(x1, x2, x3, x4)
                        xmax = max(x1, x2, x3, x4)
                        ymin = min(y1, y2, y3, y4)
                        ymax = max(y1, y2, y3, y4)
                        width = ann_meta[ann]['width']
                        height = ann_meta[ann]['height']

                        xmin /= width
                        xmax /= width
                        ymin /= height
                        ymax /= height
                        
                        cx = (xmin + xmax) / 2
                        cy = (ymin + ymax) / 2
                        w = xmax - xmin
                        h = ymax - ymin

                        out_lines.append(f"{cid} {cx} {cy} {w} {h}")
            
            with open(ann, 'w') as f:
                f.write("\n".join(out_lines))
            print(f"Updated {ann}")

    print(class_d)

if __name__ == "__main__":
    main()