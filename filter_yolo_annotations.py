import argparse
import sys
import glob
import os
import xml.etree.ElementTree as ET
import shutil
from PIL import Image
import random

def main():
    parser = argparse.ArgumentParser(description="Filter YOLO datasets")
    parser.add_argument(
        "input", 
        type=str, 
        help="Path to YOLO dataset"
    )
    parser.add_argument(
        "classes",
        type=str,
        help="List of classes to keep/merge, comma separated (e.g. 3,4)"
    )

    args = parser.parse_args()
    
    dir_sets = ["train", "test", "valid"]
    keep_cls = [int(c) for c in args.classes.split(",") if c.strip() != ""]

    for ds in dir_sets:
        annotations_path = os.path.join(args.input, ds, "labels")
        ann_files = glob.glob(f"{annotations_path}/*.txt")
        for ann in ann_files:
            out_lines = []
            with open(ann, 'r') as f:
                lines = [l.strip() for l in f.read().split("\n") if l.strip() != ""]
                for line in lines:
                    parts = line.split(" ")
                    c = int(parts[0])
                    if c in keep_cls:
                        out_lines.append(f"0 {' '.join(parts[1:])}")
            
            with open(ann, 'w') as f:
                f.write("\n".join(out_lines))
            print(f"Updated {ann}")
        
if __name__ == "__main__":
    main()