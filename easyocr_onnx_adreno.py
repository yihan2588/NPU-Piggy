"""
EasyOCR ONNX Inference Script for Qualcomm Adreno X1-85 GPU
============================================================
Runs EasyOCR text detection + recognition using ONNX Runtime
with DirectML execution provider for GPU acceleration on
Microsoft Surface Windows with Qualcomm Adreno X1-85.

Requirements:
    pip install onnxruntime-directml easyocr Pillow numpy

ONNX models required (place next to this script):
    - EasyOCRDetector.onnx
    - EasyOCRRecognizer.onnx

These can be exported from the Qualcomm AI Hub Models repo via:
    python -m qai_hub_models.models.easyocr.export --target-runtime onnx
"""

import sys
import os
import math

import numpy as np
from PIL import Image, ImageDraw

# EasyOCR utilities for post-processing (installed via `pip install easyocr`)
from easyocr.craft_utils import getDetBoxes
from easyocr.easyocr import Reader
from easyocr.recognition import custom_mean
from easyocr.utils import diff, four_point_transform, group_text_box


# ---------------------------------------------------------------------------
# Constants (matching Qualcomm's EasyOCR implementation)
# ---------------------------------------------------------------------------
DETECTOR_INPUT_H, DETECTOR_INPUT_W = 608, 800
RECOGNIZER_INPUT_H, RECOGNIZER_INPUT_W = 64, 800

DETECTOR_ARGS = {
    "text_threshold": 0.7,
    "link_threshold": 0.4,
    "low_text": 0.4,
    "poly": False,
    "slope_ths": 0.1,
    "ycenter_ths": 0.5,
    "height_ths": 0.5,
    "width_ths": 1.0,
    "add_margin": 0.1,
    "min_size": 20,
}

CONTRAST_THS = 0.1
LANG_LIST = ["en"]


# ---------------------------------------------------------------------------
# Image processing helpers (reimplemented from qai_hub_models utils,
# using only PIL + numpy — no OpenCV)
# ---------------------------------------------------------------------------

def resize_pad(image_np, dst_size, horizontal_float="center", pad_value=0.0):
    """
    Resize NCHW float32 image to fit dst_size preserving aspect ratio, then pad.

    Returns (padded, scale, (pad_left, pad_top)).
    """
    _, c, h, w = image_np.shape
    dst_h, dst_w = dst_size
    scale = min(dst_h / h, dst_w / w)
    new_h = int(math.floor(h * scale))
    new_w = int(math.floor(w * scale))

    # Bilinear resize per channel via PIL
    resized_channels = []
    for ch in range(c):
        pil_ch = Image.fromarray(image_np[0, ch], mode="F")
        pil_ch = pil_ch.resize((new_w, new_h), Image.BILINEAR)
        resized_channels.append(np.asarray(pil_ch))
    resized = np.stack(resized_channels, axis=0)[np.newaxis]  # [1, C, new_h, new_w]

    pad_h = dst_h - new_h
    pad_w = dst_w - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    if horizontal_float == "left":
        pad_left, pad_right = 0, pad_w
    else:
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

    padded = np.pad(
        resized,
        ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=pad_value,
    )
    return padded, scale, (pad_left, pad_top)


def rgb_to_grey(rgb_np):
    """Convert [H, W, 3] uint8 RGB to [H, W] uint8 grayscale (ITU-R BT.601)."""
    return np.dot(rgb_np[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)


# ---------------------------------------------------------------------------
# ONNX Runtime session setup
# ---------------------------------------------------------------------------

def create_ort_session(model_path):
    """
    Create an ONNX Runtime inference session.
    Tries DirectML (Adreno GPU) first, falls back to CPU.
    """
    import onnxruntime as ort

    available = ort.get_available_providers()
    print(f"  Providers: {available}")

    if "DmlExecutionProvider" in available:
        print(f"  -> DmlExecutionProvider (Adreno GPU) for {os.path.basename(model_path)}")
        return ort.InferenceSession(
            model_path,
            providers=["DmlExecutionProvider", "CPUExecutionProvider"],
        )
    else:
        print(f"  -> CPU fallback for {os.path.basename(model_path)}")
        return ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class EasyOCROnnxPipeline:
    """Full EasyOCR pipeline: CRAFT detector + VGG/BiLSTM/CTC recognizer via ONNX Runtime."""

    def __init__(self, detector_path, recognizer_path):
        print("Loading ONNX models...")
        self.det_sess = create_ort_session(detector_path)
        self.rec_sess = create_ort_session(recognizer_path)

        self.det_in = self.det_sess.get_inputs()[0].name
        self.det_out = self.det_sess.get_outputs()[0].name
        self.rec_in = self.rec_sess.get_inputs()[0].name
        self.rec_out = self.rec_sess.get_outputs()[0].name

        # Character set & CTC decoder from upstream EasyOCR
        print("Loading character set and CTC decoder...")
        reader = Reader(LANG_LIST, gpu=False, quantize=False)
        self.character = reader.character
        self.lang_char = reader.lang_char
        self.converter = reader.converter

        ignore_char = "".join(set(self.character) - set(self.lang_char))
        self.ignore_char_idx = [self.character.index(ch) + 1 for ch in ignore_char]

    # ---- detector --------------------------------------------------------

    def detector_preprocess(self, rgb_np):
        """[H,W,3] uint8 RGB -> [1,3,608,800] float32 [0,1], scale, padding."""
        img = rgb_np.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis]         # NCHW
        img, scale, padding = resize_pad(img, (DETECTOR_INPUT_H, DETECTOR_INPUT_W))
        # ImageNet normalization is baked into the ONNX model
        return img, scale, padding

    def run_detector(self, tensor):
        return self.det_sess.run([self.det_out], {self.det_in: tensor})[0]

    def detector_postprocess(self, det_output, scale, padding):
        """Score maps -> grouped boxes in original-image coordinates."""
        out = det_output[0]
        score_text = out[:, :, 0]
        score_link = out[:, :, 1]

        h_raw, f_raw, _ = getDetBoxes(
            textmap=score_text, linkmap=score_link,
            text_threshold=DETECTOR_ARGS["text_threshold"],
            link_threshold=DETECTOR_ARGS["link_threshold"],
            low_text=DETECTOR_ARGS["low_text"],
            poly=DETECTOR_ARGS["poly"],
        )

        # Denormalize — CRAFT output is half input resolution
        half_s = scale / 2
        half_p = (padding[0] // 2, padding[1] // 2)
        detections = []
        for idx in range(len(h_raw)):
            box = np.array(f_raw[idx] if f_raw[idx] is not None else h_raw[idx],
                           dtype=np.float32).reshape(-1, 2)
            box[:, 0] = ((box[:, 0] - half_p[0]) / half_s).astype(np.int32)
            box[:, 1] = ((box[:, 1] - half_p[1]) / half_s).astype(np.int32)
            detections.append(box.astype(np.int32).reshape(-1))

        h_list, f_list = group_text_box(
            detections,
            slope_ths=DETECTOR_ARGS["slope_ths"],
            ycenter_ths=DETECTOR_ARGS["ycenter_ths"],
            height_ths=DETECTOR_ARGS["height_ths"],
            width_ths=DETECTOR_ARGS["width_ths"],
            add_margin=DETECTOR_ARGS["add_margin"],
        )
        h_list = [tuple(x) for x in h_list]
        f_list = [tuple(tuple(y) for y in x) for x in f_list]

        ms = DETECTOR_ARGS["min_size"]
        if ms:
            h_list = [b for b in h_list if max(b[1]-b[0], b[3]-b[2]) > ms]
            f_list = [b for b in f_list
                      if max(diff([c[0] for c in b]), diff([c[1] for c in b])) > ms]
        return h_list, f_list

    # ---- recognizer ------------------------------------------------------

    def crop_text_regions(self, grey, h_boxes, f_boxes):
        """Crop text cutouts from grayscale image. Returns [(img, box, y_min)]."""
        if not h_boxes and not f_boxes:
            h, w = grey.shape
            h_boxes = [(0, w, 0, h)]
            f_boxes = []

        cuts = []
        for fb in f_boxes:
            rect = np.array(fb, dtype="float32")
            c = four_point_transform(grey, rect)
            if 0 not in c.shape:
                cuts.append((c, fb, min(rect[0][1], rect[1][1], rect[2][1], rect[3][1])))

        for b in h_boxes:
            x0, x1 = max(0, b[0]), min(b[1], grey.shape[1])
            y0, y1 = max(0, b[2]), min(b[3], grey.shape[0])
            if y1 - y0 > 0 and x1 - x0 > 0:
                cuts.append((grey[y0:y1, x0:x1], b, y0))

        cuts.sort(key=lambda t: t[2])
        return cuts

    def prepare_recognizer_input(self, cutout):
        """[H,W] uint8 grey -> [1,1,64,800] float32 [0,1], left-aligned, bg-color padded."""
        img = cutout.astype(np.float32) / 255.0
        if img.ndim == 2:
            img = img[np.newaxis, np.newaxis]
        else:
            img = np.transpose(img, (2, 0, 1))[np.newaxis]
        pad_val = float(img[0, 0, 0, 0])
        img, _, _ = resize_pad(img, (RECOGNIZER_INPUT_H, RECOGNIZER_INPUT_W),
                               horizontal_float="left", pad_value=pad_val)
        return img

    def run_recognizer(self, tensor):
        # (x-0.5)/0.5 normalization is baked into the ONNX model
        return self.rec_sess.run([self.rec_out], {self.rec_in: tensor})[0]

    def decode_predictions(self, preds_list):
        """Softmax -> ignore chars -> greedy CTC decode -> confidence."""
        if not preds_list:
            return []
        preds = np.concatenate(preds_list, axis=0)

        # Stable softmax
        e = np.exp(preds - preds.max(axis=2, keepdims=True))
        prob = e / e.sum(axis=2, keepdims=True)

        if self.ignore_char_idx:
            prob[:, :, self.ignore_char_idx] = 0.0
            n = prob.sum(axis=2, keepdims=True)
            prob = prob / np.where(n == 0, 1.0, n)

        idx = prob.argmax(axis=2)
        sizes = np.array([preds.shape[1]] * preds.shape[0], dtype=np.int32)
        texts = self.converter.decode_greedy(idx.reshape(-1), sizes)

        vals = prob.max(axis=2)
        results = []
        for txt, v, i in zip(texts, vals, idx):
            mp = v[i != 0]
            results.append((txt, custom_mean(mp) if len(mp) else np.float64(0)))
        return results

    def recognize_with_contrast_retry(self, cutouts):
        """Run recognizer; retry low-confidence cutouts with 10x contrast."""
        inputs = [self.prepare_recognizer_input(c[0]) for c in cutouts]
        preds = [self.run_recognizer(inp) for inp in inputs]
        predictions = self.decode_predictions(preds)

        # Retry low-confidence with enhanced contrast
        hc = {}
        for i, (_, conf) in enumerate(predictions):
            if conf < CONTRAST_THS:
                f = cutouts[i][0].astype(np.float32) / 255.0
                m = f.mean()
                enhanced = np.clip((f - m) * (1.0 / CONTRAST_THS) + m, 0, 1)
                inp = self.prepare_recognizer_input((enhanced * 255).astype(np.uint8))
                dec = self.decode_predictions([self.run_recognizer(inp)])
                if dec:
                    hc[i] = dec[0]

        results = []
        for i, ((_, box, _), (txt, conf)) in enumerate(zip(cutouts, predictions)):
            if i in hc and hc[i][1] > conf:
                txt, conf = hc[i]
            if not txt:
                continue
            txt = txt.strip()
            if txt and txt[-1] in ["]", "|"]:
                txt = txt[:-1].strip()
            if txt:
                results.append((box, txt, conf))
        return results

    # ---- full pipeline ---------------------------------------------------

    def predict(self, rgb_np):
        """
        [H,W,3] uint8 RGB -> list of (box_coords, text, confidence).
        """
        det_in, scale, pad = self.detector_preprocess(rgb_np)
        det_out = self.run_detector(det_in)
        h_boxes, f_boxes = self.detector_postprocess(det_out, scale, pad)

        grey = rgb_to_grey(rgb_np)
        cutouts = self.crop_text_regions(grey, h_boxes, f_boxes)
        if not cutouts:
            return []
        return self.recognize_with_contrast_retry(cutouts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def prompt(msg):
    print(msg, end="", flush=True)
    return sys.stdin.readline().strip()


def main():
    print("=== EasyOCR ONNX Inference Engine (Adreno GPU) ===", flush=True)
    print("Initializing ONNX Runtime and loading models...\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    det_path = os.path.join(script_dir, "EasyOCRDetector.onnx")
    rec_path = os.path.join(script_dir, "EasyOCRRecognizer.onnx")

    for p, n in [(det_path, "EasyOCRDetector.onnx"), (rec_path, "EasyOCRRecognizer.onnx")]:
        if not os.path.exists(p):
            print(f"[ERROR] {n} not found at: {p}")
            sys.exit(1)

    pipeline = EasyOCROnnxPipeline(det_path, rec_path)
    print("\n[SUCCESS] Models loaded and ready for inference!\n")

    while True:
        image_name = prompt("Enter image file name (or 'quit' to exit): ")
        if image_name.lower() in ["quit", "q", "exit"]:
            print("Shutting down OCR engine...")
            break
        if not image_name:
            continue

        image_path = image_name if os.path.isabs(image_name) \
            else os.path.join(script_dir, image_name)
        if not os.path.exists(image_path):
            print(f"[ERROR] File not found: {image_path}\n")
            continue

        print(f"Running OCR on: {image_name}...")
        try:
            pil_img = Image.open(image_path).convert("RGB")
            rgb_np = np.array(pil_img)

            results = pipeline.predict(rgb_np)
            if not results:
                print("No text detected.\n")
                continue

            print("Predicted texts & confidence:")
            print("-" * 50)

            draw = ImageDraw.Draw(pil_img)
            for box, text, conf in results:
                print(f"  {text:<40}  {conf:.4f}")
                if isinstance(box[0], tuple):
                    pts = [c for corner in box for c in corner]
                    draw.polygon(pts, outline="green", width=2)
                else:
                    xmin, xmax, ymin, ymax = box
                    draw.rectangle([xmin, ymin, xmax, ymax], outline="green", width=2)

            base = os.path.splitext(os.path.basename(image_path))[0]
            out_path = os.path.join(os.path.dirname(image_path), f"{base}_output.png")
            pil_img.save(out_path)
            print("-" * 50)
            print(f"Annotated image saved to: {out_path}\n")

        except Exception as e:
            print(f"[ERROR] {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
