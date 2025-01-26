from ultralytics import YOLO
import onnx
import argparse
import os
import json
from onnxsim import simplify
from onnxruntime.quantization import quantize_dynamic, QuantType,  quantize_static, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process

parser = argparse.ArgumentParser(description="Convert YOLO models to ONNX")
parser.add_argument(
    "input", 
    type=str, 
    help="Path YOLO model weights (.pt)"
)

args = parser.parse_args()
model = YOLO(args.input)
model.export(format='onnx')
out_model = os.path.splitext(args.input)[0] + ".onnx"
out_model_optim = os.path.splitext(args.input)[0] + ".optim.onnx"
out_model_quant = os.path.splitext(args.input)[0] + ".quant.onnx"

params = {
    'model_type': 'Detector',
    'det_iou_thresh': 0.3,
    'det_type': 'YOLO_v8',
    'resolution': 10, 
    'class_names': {"0": "tree"}, 
    'det_conf': 0.3, 
    'tiles_overlap': 5, 
}

m = onnx.load(out_model)
for k,v in params.items():
    meta = m.metadata_props.add()
    meta.key = k
    meta.value = json.dumps(v)

onnx.save(m, out_model)


model_simp, check = simplify(m)
onnx.save(m, out_model)
print(f"Wrote {out_model}")

quant_pre_process(out_model, out_model_optim, skip_symbolic_shape=True)
quantized_model = quantize_dynamic(out_model_optim, out_model_quant, weight_type=QuantType.QUInt8)
os.unlink(out_model_optim)

print(f"Wrote {out_model_quant}")