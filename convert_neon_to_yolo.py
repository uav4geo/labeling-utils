import argparse
import sys
import glob
import os
import xml.etree.ElementTree as ET
import shutil
from PIL import Image
import random

def main():
    parser = argparse.ArgumentParser(description="Convert NEON dataset to YOLOv7 format")
    parser.add_argument(
        "input", 
        type=str, 
        help="Path to NEON dataset"
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.9,
        help="Split for train/eval datasets"
    )

    args = parser.parse_args()

    images_path = os.path.join(args.input, "evaluation", "RGB")
    images = []

    for ext in ["jpg", "tif", "png", "jpeg", "tiff"]:
        images += glob.glob(f"{images_path}/*.{ext}")
    
    annotations_path = os.path.join(args.input, "annotations")

    print(f"Found {len(images)} images")

    annotations = []
    ann_files = glob.glob(f"{annotations_path}/*.xml")
    for ann in ann_files:
        tree = ET.parse(ann)
        root = tree.getroot()
        folder = root.find('folder').text
        filename = root.find('filename').text
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        if width > 2500:
            print(f"Skipping large image: {filename}")
            continue
        
        bboxes = []
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            bboxes.append([xmin, ymin, xmax, ymax])
        
        annotations.append({
            'folder': folder,
            'filename': filename,
            'width': width,
            'height': height,
            'bboxes': bboxes
        })
    
    print(f"Found {len(annotations)} annotations")
    img_d = {}
    for im in images:
        img_d[os.path.basename(im)] = im
    
    pairs = []
    for ann in annotations:
        fname = ann['filename']
        if fname in img_d:
            pairs += [(fname, ann)]
        else:
            print(f"Cannot find {fname}")
    
    if len(pairs) == len(images):
        print("All annotations match")
    else:
        print(f"Skipped {len(images) - len(pairs)} images, total: {len(pairs)}")
    
    output_dir = os.path.join(args.input, "output")
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)

    random.shuffle(pairs)
    num_train = int(len(pairs) * args.split)
    
    train_set = pairs[:num_train]
    valid_set = pairs[num_train:]
    for pair_set, dir_set in zip([train_set, valid_set], ["train", "valid"]):
        out_images_dir = os.path.join(output_dir, dir_set, "images")
        out_labels_dir = os.path.join(output_dir, dir_set, "labels")
        for d in [out_images_dir, out_labels_dir]:
            os.makedirs(d, exist_ok=True)

        c = 0
        for im, ann in pair_set:
            img = Image.open(img_d[im]).convert("RGB")
            if img.width != ann['width'] or img.height != ann['height']:
                print(f"Width/height mismatch: {im}")
                continue
            
            p, ext = os.path.splitext(im)
            img.save(os.path.join(out_images_dir, p + ".png"))

            with open(os.path.join(out_labels_dir, p + ".txt"), "w") as f:
                for bbox in ann['bboxes']:
                    xmin = bbox[0] / ann['width']
                    xmax = bbox[2] / ann['width']
                    ymin = bbox[1] / ann['height']
                    ymax = bbox[3] / ann['height']
                    
                    cx = (xmin + xmax) / 2
                    cy = (ymin + ymax) / 2
                    w = xmax - xmin
                    h = ymax - ymin

                    f.write(f"0 {cx} {cy} {w} {h}\n")
            c += 1
        print(f"Wrote {out_images_dir} ({c} images)")
        print(f"Wrote {out_labels_dir} ({c} annotations)")

        
if __name__ == "__main__":
    main()