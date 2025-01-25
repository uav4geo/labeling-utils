import argparse
import sys
import glob
import os
from PIL import Image, ImageDraw

def main():
    parser = argparse.ArgumentParser(description="Draw YOLO boxes to images")
    parser.add_argument(
        "input", 
        type=str, 
        help="Path to image or directory from yolo dataset"
    )

    args = parser.parse_args()

    images = []
    annotations = []
    annotations_path = None

    if os.path.isfile(args.input):
        images = [args.input]
        annotations_path = os.path.join(os.path.dirname(args.input), "..", "labels")
        
    elif os.path.isdir(args.input):
        for ext in ["jpg", "tif", "png", "jpeg", "tiff"]:
            p = os.path.join(args.input, "images")
            images += glob.glob(f"{p}/*.{ext}")
        annotations_path = os.path.join(args.input, "labels")

    if not os.path.isdir(annotations_path):
        print(f"Cannot find annotations directory: {annotations_path}")
        exit(1)

    print(f"Found {len(images)} images")

    pairs = []
    for im in images:
        p, ext = os.path.splitext(im)
        ann_file = os.path.join(annotations_path, os.path.basename(p) + ".txt")

        if os.path.isfile(ann_file):
            pairs += [(im, ann_file)]
        else:
            print(f"Cannot find {ann_file}")
    
    if len(pairs) == len(images):
        print("All annotations match")
    else:
        print(f"Skipped {len(images) - len(pairs)} images")

    output_dir = os.path.abspath(os.path.join(annotations_path, "..", "draw_output"))
    os.makedirs(output_dir, exist_ok=True)
    
    for im,ann in pairs:
        img = Image.open(im).convert("RGB")
        draw = ImageDraw.Draw(img)
        color = (255, 0, 0)  # Red for the bounding box

        boxes = []
        classes = []
        with open(ann, "r") as f:
            lines = [p for p in f.read().strip().split("\n") if p != ""]
            for line in lines:
                parts = line.split(" ")
                if len(parts) != 5:
                    print(f"Invalid format: {os.path.basename(ann)}: {line}")
                    continue
                classes.append(int(parts[0]))
                boxes.append([float(p) for p in parts[1:]])
        
        # Draw bounding boxes with scores
        for b, cla in zip(boxes, classes):
            x, y, w, h = b

            x1 = (x - w / 2) * img.width
            y1 = (y - h / 2) * img.height
            x2 = (x + w / 2) * img.width
            y2 = (y + h / 2) * img.height

            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            draw.text((x1, y1 - 10), str(cla), fill=color)

        img.save(os.path.join(output_dir, os.path.basename(im)))
    
    print(f"Wrote images to {output_dir}")


if __name__ == "__main__":
    main()