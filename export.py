import os
import warnings
from typing import Optional, List, Dict, Any
import torch
import torch.nn as nn
from .config import ModelArchConfig
from .model import GPT
def export_to_torchscript_trace(model, example_input, output_path, optimize=True):
    model.eval()
    original_device = next(model.parameters()).device
    model = model.cpu()
    example_input = example_input.cpu()
    original_flash = model.config.flash_attention
    model.config.flash_attention = False
    with torch.no_grad():
        traced = torch.jit.trace(model, example_input, check_trace=False)
    if optimize:
        traced = torch.jit.optimize_for_inference(traced)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    traced.save(output_path)
    model.config.flash_attention = original_flash
    model = model.to(original_device)
    print(f"[INFO] Exported TorchScript to {output_path}")
    return output_path
def export_to_torchscript_script(model, output_path, optimize=True):
    model.eval()
    original_device = next(model.parameters()).device
    model = model.cpu()
    original_flash = model.config.flash_attention
    model.config.flash_attention = False
    scripted = torch.jit.script(model)
    if optimize:
        scripted = torch.jit.optimize_for_inference(scripted)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    scripted.save(output_path)
    model.config.flash_attention = original_flash
    model = model.to(original_device)
    print(f"[INFO] Exported TorchScript to {output_path}")
    return output_path
def export_to_torchscript(model, example_input, output_path, method='trace', optimize=True):
    if method == 'trace':
        return export_to_torchscript_trace(model, example_input, output_path, optimize)
    elif method == 'script':
        return export_to_torchscript_script(model, output_path, optimize)
    else:
        raise ValueError(f"Unknown method: {method}")
def export_to_onnx(model, example_input, output_path, opset_version=14, do_constant_folding=True,
                   dynamic_axes=None, use_fp16=False):
    model.eval()
    original_device = next(model.parameters()).device
    model = model.cpu()
    example_input = example_input.cpu()
    original_flash = model.config.flash_attention
    model.config.flash_attention = False
    if dynamic_axes is None:
        dynamic_axes = {
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size', 1: 'sequence_length'},
        }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        torch.onnx.export(
            model, example_input, output_path,
            opset_version=opset_version,
            do_constant_folding=do_constant_folding,
            input_names=['input_ids'],
            output_names=['logits'],
            dynamic_axes=dynamic_axes,
            verbose=False,
        )
    model.config.flash_attention = original_flash
    model = model.to(original_device)
    print(f"[INFO] Exported ONNX to {output_path}")
    if use_fp16:
        output_fp16 = output_path.replace('.onnx', '_fp16.onnx')
        _convert_onnx_to_fp16(output_path, output_fp16)
        return output_fp16
    return output_path
def _convert_onnx_to_fp16(input_path, output_path):
    try:
        import onnx
        from onnxruntime.transformers.float16 import convert_float_to_float16
        model = onnx.load(input_path)
        model_fp16 = convert_float_to_float16(model)
        onnx.save(model_fp16, output_path)
        print(f"[INFO] Converted ONNX to FP16: {output_path}")
    except ImportError as e:
        print(f"[WARN] Cannot convert ONNX to FP16: {e}. Install onnxruntime-gpu.")
    except Exception as e:
        print(f"[WARN] FP16 conversion failed: {e}")
def export_to_tensorrt(model, example_input, output_path, fp16=True, int8=False, workspace_size=1<<30):
    try:
        from torch2trt import torch2trt
        model.eval()
        model = model.cuda()
        example_input = example_input.cuda()
        original_flash = model.config.flash_attention
        model.config.flash_attention = False
        trt_model = torch2trt(model, [example_input], fp16_mode=fp16, int8_mode=int8, max_workspace_size=workspace_size)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(trt_model.state_dict(), output_path)
        model.config.flash_attention = original_flash
        print(f"[INFO] Exported TensorRT to {output_path}")
        return output_path
    except ImportError:
        print("[WARN] torch2trt not installed. Install with: git clone https://github.com/NVIDIA-AI-IOT/torch2trt")
        return None
def quantize_dynamic(model, output_path):
    model.eval()
    model = model.cpu()
    original_flash = model.config.flash_attention
    model.config.flash_attention = False
    quantized = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(quantized.state_dict(), output_path)
    model.config.flash_attention = original_flash
    print(f"[INFO] Exported dynamically quantized model to {output_path}")
    return output_path
class Exporter:
    def __init__(self, model, config):
        self.model = model
        self.config = config
    def to_torchscript(self, example_input, output_path, method='trace', optimize=True):
        return export_to_torchscript(self.model, example_input, output_path, method, optimize)
    def to_onnx(self, example_input, output_path, opset_version=14, use_fp16=False):
        return export_to_onnx(self.model, example_input, output_path, opset_version, use_fp16=use_fp16)
    def to_tensorrt(self, example_input, output_path, fp16=True):
        return export_to_tensorrt(self.model, example_input, output_path, fp16)
    def quantize(self, output_path):
        return quantize_dynamic(self.model, output_path)
    def export_all(self, example_input, base_path, formats=['torchscript', 'onnx']):
        results = {}
        if 'torchscript' in formats:
            results['torchscript'] = self.to_torchscript(example_input, f"{base_path}_traced.pt")
        if 'onnx' in formats:
            results['onnx'] = self.to_onnx(example_input, f"{base_path}.onnx")
        if 'tensorrt' in formats:
            trt = self.to_tensorrt(example_input, f"{base_path}_trt.pt")
            if trt:
                results['tensorrt'] = trt
        if 'quantized' in formats:
            results['quantized'] = self.quantize(f"{base_path}_quantized.pt")
        return results
