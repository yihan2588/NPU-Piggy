"""
EasyOCR ONNX Inference Script for Qualcomm Adreno X1-85 GPU
============================================================
Runs EasyOCR text detection + recognition using ONNX Runtime
with DirectML execution provider for GPU acceleration on
Microsoft Surface Windows with Qualcomm Adreno X1-85.

Requirements:
    pip install onnxruntime-directml easyocr opencv-python-headless Pillow numpy

ONNX models required (place next to this script):
    - EasyOCRDetector.onnx
    - EasyOCRRecognizer.onnx

These can be exported from the Qualcomm AI Hub Models repo via:
    python -m qai_hub_models.models.easyocr.export --target-runtime onnx
"""

import sys
import os
import math

import cv2
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
    "width_ths": 1.0,  # wider grouping for merging horizontal text
    "add_margin": 0.1,
    "min_size": 20,
}

CONTRAST_THS = 0.1
LANG_LIST = ["en"]


# ---------------------------------------------------------------------------
# Image processing helpers (reimplemented from qai_hub_models utils)
# ---------------------------------------------------------------------------

def resize_pad(image_np, dst_size, horizontal_float="center", pad_value=0.0):
    """
    Resize image (NCHW float32) to fit within dst_size, preserving aspect ratio,
    then pad to exactly dst_size.

    Parameters
    ----------
    image_np : np.ndarray
        Shape [1, C, H, W], float32.
    dst_size : tuple
        (dst_H, dst_W).
    horizontal_float : str
        "center" or "left".
    pad_value : float
        Value used for padding.

    Returns
    -------
    padded : np.ndarray [1, C, dst_H, dst_W]
    scale : float
    padding : (pad_left, pad_top)
    """
    _, c, h, w = image_np.shape
    dst_h, dst_w = dst_size

    h_ratio = dst_h / h
    w_ratio = dst_w / w
    scale = min(h_ratio, w_ratio)

    new_h = int(math.floor(h * scale))
    new_w = int(math.floor(w * scale))

    # Resize each channel via cv2
    resized_channels = []
    for b in range(image_np.shape[0]):
        channels = []
        for ch in range(c):
            resized_ch = cv2.resize(
                image_np[b, ch], (new_w, new_h), interpolation=cv2.INTER_LINEAR
            )
            channels.append(resized_ch)
        resized_channels.append(np.stack(channels, axis=0))
    resized = np.stack(resized_channels, axis=0)  # [1, C, new_h, new_w]

    # Compute padding
    pad_h = dst_h - new_h
    pad_w = dst_w - new_w

    # Vertical: always center
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    # Horizontal
    if horizontal_float == "left":
        pad_left = 0
        pad_right = pad_w
    else:  # center
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

    padded = np.pad(
        resized,
        ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=pad_value,
    )

    return padded, scale, (pad_left, pad_top)


def denormalize_coordinates(coords, scale, pad):
    """
    Map detector output coordinates back to original image pixel space.
    coords: np.ndarray shape [N, 2] where columns are (y, x) or (x, y) depending on context.
    The CRAFT output is at half the input resolution, so scale and pad are already halved by caller.
    """
    coords[:, 0] = ((coords[:, 0] - pad[0]) / scale).astype(np.int32)
    coords[:, 1] = ((coords[:, 1] - pad[1]) / scale).astype(np.int32)


# ---------------------------------------------------------------------------
# ONNX Runtime session setup
# ---------------------------------------------------------------------------

def create_ort_session(model_path):
    """
    Create an ONNX Runtime inference session.
    Tries DirectML (Adreno GPU) first, falls back to CPU.
    """
    import onnxruntime as ort

    available_providers = ort.get_available_providers()
    print(f"  Available ONNX Runtime providers: {available_providers}")

    # Try DirectML for Qualcomm Adreno GPU on Windows ARM64
    if "DmlExecutionProvider" in available_providers:
        print(f"  -> Using DmlExecutionProvider (Adreno GPU) for {os.path.basename(model_path)}")
        session = ort.InferenceSession(
            model_path,
            providers=["DmlExecutionProvider", "CPUExecutionProvider"],
        )
    else:
        print(f"  -> DmlExecutionProvider not available, falling back to CPU for {os.path.basename(model_path)}")
        session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

    return session


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class EasyOCROnnxPipeline:
    """Full EasyOCR inference pipeline using ONNX Runtime."""

    def __init__(self, detector_path, recognizer_path):
        print("Loading ONNX models...")
        self.detector_session = create_ort_session(detector_path)
        self.recognizer_session = create_ort_session(recognizer_path)

        # Get input/output names
        self.det_input_name = self.detector_session.get_inputs()[0].name
        self.det_output_name = self.detector_session.get_outputs()[0].name
        self.rec_input_name = self.recognizer_session.get_inputs()[0].name
        self.rec_output_name = self.recognizer_session.get_outputs()[0].name

        # Initialize EasyOCR reader (CPU, just for the character set & decoder)
        print("Loading character set and CTC decoder from EasyOCR...")
        ocr_reader = Reader(LANG_LIST, gpu=False, quantize=False)
        self.character = ocr_reader.character
        self.lang_char = ocr_reader.lang_char
        self.converter = ocr_reader.converter

        # Build ignore char index list (chars in vocab but not in selected language)
        ignore_char = "".join(set(self.character) - set(self.lang_char))
        self.ignore_char_idx = [self.character.index(ch) + 1 for ch in ignore_char]

    # -- Detector --

    def detector_preprocess(self, rgb_image_np):
        """
        Prepare an RGB image for the detector.

        Parameters
        ----------
        rgb_image_np : np.ndarray
            [H, W, 3] uint8 RGB image.

        Returns
        -------
        input_tensor : np.ndarray [1, 3, 608, 800] float32
        scale : float
        padding : (pad_left, pad_top)
        """
        # HWC uint8 -> NCHW float32 [0, 1]
        img = rgb_image_np.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # CHW
        img = np.expand_dims(img, 0)  # NCHW

        # Resize + pad to (608, 800)
        img_resized, scale, padding = resize_pad(img, (DETECTOR_INPUT_H, DETECTOR_INPUT_W))

        # NOTE: ImageNet normalization is baked into the ONNX model
        # (EasyOCRDetector.forward() calls normalize_image_torchvision internally)
        # So we pass [0, 1] range directly.

        return img_resized, scale, padding

    def run_detector(self, input_tensor):
        """Run the ONNX detector and return raw output."""
        result = self.detector_session.run(
            [self.det_output_name],
            {self.det_input_name: input_tensor},
        )
        return result[0]  # [1, H/2, W/2, 2]

    def detector_postprocess(self, det_output, scale, padding):
        """
        Convert detector output to bounding boxes in original image coordinates.

        Returns
        -------
        horizontal_boxes : list of (xmin, xmax, ymin, ymax)
        free_boxes : list of ((x1,y1), (x2,y2), (x3,y3), (x4,y4))
        """
        out = det_output[0]  # [H/2, W/2, 2]
        score_text = out[:, :, 0]
        score_link = out[:, :, 1]

        # Get detection boxes from CRAFT score maps
        horizontal_boxes_raw, free_boxes_raw, _ = getDetBoxes(
            textmap=score_text,
            linkmap=score_link,
            text_threshold=DETECTOR_ARGS["text_threshold"],
            link_threshold=DETECTOR_ARGS["link_threshold"],
            low_text=DETECTOR_ARGS["low_text"],
            poly=DETECTOR_ARGS["poly"],
        )

        # Combine boxes, preferring free boxes
        detections = []
        for idx in range(len(horizontal_boxes_raw)):
            if free_boxes_raw[idx] is not None:
                box = np.array(free_boxes_raw[idx], dtype=np.float32)
            else:
                box = np.array(horizontal_boxes_raw[idx], dtype=np.float32)

            # Denormalize: detector output is at half the input resolution
            # so we divide scale and padding by 2
            half_scale = scale / 2
            half_pad = (padding[0] // 2, padding[1] // 2)
            box_flat = box.reshape(-1, 2)  # [4, 2] — each row is (x, y)
            box_flat[:, 0] = ((box_flat[:, 0] - half_pad[0]) / half_scale).astype(np.int32)
            box_flat[:, 1] = ((box_flat[:, 1] - half_pad[1]) / half_scale).astype(np.int32)
            detections.append(box_flat.astype(np.int32).reshape(-1))

        # Group text boxes
        horizontal_list_raw, free_list_raw = group_text_box(
            detections,
            slope_ths=DETECTOR_ARGS["slope_ths"],
            ycenter_ths=DETECTOR_ARGS["ycenter_ths"],
            height_ths=DETECTOR_ARGS["height_ths"],
            width_ths=DETECTOR_ARGS["width_ths"],
            add_margin=DETECTOR_ARGS["add_margin"],
        )

        horizontal_list = [tuple(x) for x in horizontal_list_raw]
        free_list = [tuple(tuple(y) for y in x) for x in free_list_raw]

        # Filter by minimum size
        min_size = DETECTOR_ARGS["min_size"]
        if min_size:
            horizontal_list = [
                i for i in horizontal_list
                if max(i[1] - i[0], i[3] - i[2]) > min_size
            ]
            free_list = [
                i for i in free_list
                if max(diff([c[0] for c in i]), diff([c[1] for c in i])) > min_size
            ]

        return horizontal_list, free_list

    # -- Recognizer --

    def crop_text_regions(self, grey_image, horizontal_boxes, free_boxes):
        """
        Crop text regions from the grayscale image.

        Returns
        -------
        cutouts : list of (cutout_image, box_coords, y_min)
        """
        if not horizontal_boxes and not free_boxes:
            y_max, x_max = grey_image.shape
            horizontal_boxes = [(0, x_max, 0, y_max)]
            free_boxes = []

        cutouts = []

        # Free boxes: perspective warp
        for free_box in free_boxes:
            rect = np.array(free_box, dtype="float32")
            cutout = four_point_transform(grey_image, rect)
            if 0 in cutout.shape:
                continue
            y_min = min(rect[0][1], rect[1][1], rect[2][1], rect[3][1])
            cutouts.append((cutout, free_box, y_min))

        # Horizontal boxes: direct crop
        for box in horizontal_boxes:
            x_min = max(0, box[0])
            x_max = min(box[1], grey_image.shape[1])
            y_min = max(0, box[2])
            y_max = min(box[3], grey_image.shape[0])
            if y_max - y_min <= 0 or x_max - x_min <= 0:
                continue
            cutout = grey_image[y_min:y_max, x_min:x_max]
            cutouts.append((cutout, box, y_min))

        # Sort by vertical position
        cutouts.sort(key=lambda item: item[2])
        return cutouts

    def prepare_recognizer_input(self, cutout_grey):
        """
        Prepare a single grayscale cutout for the recognizer.

        Parameters
        ----------
        cutout_grey : np.ndarray [H, W] uint8

        Returns
        -------
        input_tensor : np.ndarray [1, 1, 64, 800] float32 in [0, 1]
        """
        # Convert to NCHW float32 [0, 1]
        img = cutout_grey.astype(np.float32) / 255.0
        if img.ndim == 2:
            img = img[np.newaxis, np.newaxis, :, :]  # [1, 1, H, W]
        else:
            img = np.transpose(img, (2, 0, 1))[np.newaxis, :, :, :]  # [1, C, H, W]

        # Use top-left pixel as pad value (prevents hallucination from black padding)
        pad_value = img[0, 0, 0, 0]

        # Resize + pad to (64, 800), left-aligned
        img_resized, _, _ = resize_pad(
            img, (RECOGNIZER_INPUT_H, RECOGNIZER_INPUT_W),
            horizontal_float="left",
            pad_value=float(pad_value),
        )

        return img_resized

    def run_recognizer(self, input_tensor):
        """Run the ONNX recognizer and return raw output logits."""
        # NOTE: The (x - 0.5) / 0.5 normalization is baked into the ONNX model
        # (EasyOCRRecognizer.forward() does it internally)
        # So we pass [0, 1] range directly.
        result = self.recognizer_session.run(
            [self.rec_output_name],
            {self.rec_input_name: input_tensor},
        )
        return result[0]  # [1, T, num_classes]

    def decode_predictions(self, preds_list):
        """
        Decode recognizer output logits into text and confidence.

        Parameters
        ----------
        preds_list : list of np.ndarray, each [1, T, num_classes]

        Returns
        -------
        results : list of (text, confidence)
        """
        if not preds_list:
            return []

        preds = np.concatenate(preds_list, axis=0)  # [N, T, num_classes]

        # Softmax
        preds_exp = np.exp(preds - np.max(preds, axis=2, keepdims=True))
        preds_prob = preds_exp / np.sum(preds_exp, axis=2, keepdims=True)

        # Zero out ignored characters and renormalize
        if self.ignore_char_idx:
            preds_prob[:, :, self.ignore_char_idx] = 0.0
            norm = np.sum(preds_prob, axis=2, keepdims=True)
            norm = np.where(norm == 0, 1.0, norm)  # avoid div by zero
            preds_prob = preds_prob / norm

        # Greedy decode
        preds_index = np.argmax(preds_prob, axis=2)  # [N, T]
        preds_size = np.array([preds.shape[1]] * preds.shape[0], dtype=np.int32)
        preds_str = self.converter.decode_greedy(
            preds_index.reshape(-1), preds_size
        )

        # Confidence scores
        values = np.max(preds_prob, axis=2)
        indices = np.argmax(preds_prob, axis=2)
        results = []
        for pred_text, v, idx in zip(preds_str, values, indices):
            max_probs = v[idx != 0]
            if len(max_probs) > 0:
                confidence = custom_mean(max_probs)
            else:
                confidence = np.float64(0)
            results.append((pred_text, confidence))

        return results

    def recognize_with_contrast_retry(self, cutouts):
        """
        Run recognition on cutouts, retrying low-confidence ones with enhanced contrast.

        Parameters
        ----------
        cutouts : list of (cutout_grey_image, box_coords, y_min)

        Returns
        -------
        results : list of (box_coords, text, confidence)
        """
        # First pass: prepare and run all cutouts
        rec_inputs = []
        for cutout_img, _, _ in cutouts:
            rec_input = self.prepare_recognizer_input(cutout_img)
            rec_inputs.append(rec_input)

        # Run recognizer on all cutouts
        preds_list = []
        for rec_input in rec_inputs:
            pred = self.run_recognizer(rec_input)
            preds_list.append(pred)

        predictions = self.decode_predictions(preds_list)

        # Second pass: retry low-confidence with high contrast
        low_conf_indices = [
            i for i, (_, conf) in enumerate(predictions)
            if conf < CONTRAST_THS
        ]

        high_contrast_predictions = {}
        if low_conf_indices:
            contrast_factor = 1.0 / CONTRAST_THS  # 10x
            for idx in low_conf_indices:
                cutout_img = cutouts[idx][0]
                # Enhance contrast
                img_float = cutout_img.astype(np.float32) / 255.0
                mean_val = np.mean(img_float)
                enhanced = np.clip((img_float - mean_val) * contrast_factor + mean_val, 0, 1)
                enhanced_uint8 = (enhanced * 255).astype(np.uint8)

                rec_input = self.prepare_recognizer_input(enhanced_uint8)
                pred = self.run_recognizer(rec_input)
                decoded = self.decode_predictions([pred])
                if decoded:
                    high_contrast_predictions[idx] = decoded[0]

        # Combine results
        results = []
        for i, ((cutout_img, box_coords, _), (text, conf)) in enumerate(
            zip(cutouts, predictions)
        ):
            if i in high_contrast_predictions:
                hc_text, hc_conf = high_contrast_predictions[i]
                if hc_conf > conf:
                    text, conf = hc_text, hc_conf

            if not text:
                continue

            # Clean up hallucination artifacts
            text = text.strip()
            if text and text[-1] in ["]", "|"]:
                text = text[:-1].strip()

            if text:
                results.append((box_coords, text, conf))

        return results

    # -- Full pipeline --

    def predict(self, rgb_image_np):
        """
        Run full OCR pipeline on an RGB image.

        Parameters
        ----------
        rgb_image_np : np.ndarray [H, W, 3] uint8 RGB

        Returns
        -------
        results : list of (box_coords, text, confidence)
        """
        # 1. Detector
        det_input, scale, padding = self.detector_preprocess(rgb_image_np)
        det_output = self.run_detector(det_input)
        horizontal_boxes, free_boxes = self.detector_postprocess(det_output, scale, padding)

        # 2. Convert to grayscale for recognizer
        grey = cv2.cvtColor(rgb_image_np, cv2.COLOR_RGB2GRAY)

        # 3. Crop text regions
        cutouts = self.crop_text_regions(grey, horizontal_boxes, free_boxes)

        if not cutouts:
            return []

        # 4. Recognize
        results = self.recognize_with_contrast_retry(cutouts)

        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def prompt(msg):
    """Print a prompt and read a line — works correctly in Git Bash."""
    print(msg, end="", flush=True)
    return sys.stdin.readline().strip()


def main():
    print("=== EasyOCR ONNX Inference Engine (Adreno GPU) ===", flush=True)
    print("Initializing ONNX Runtime and loading models...\n")

    # Locate ONNX models next to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    detector_path = os.path.join(script_dir, "EasyOCRDetector.onnx")
    recognizer_path = os.path.join(script_dir, "EasyOCRRecognizer.onnx")

    for path, name in [(detector_path, "EasyOCRDetector.onnx"), (recognizer_path, "EasyOCRRecognizer.onnx")]:
        if not os.path.exists(path):
            print(f"[ERROR] {name} not found at: {path}")
            print("Please place the ONNX model files next to this script.")
            sys.exit(1)

    pipeline = EasyOCROnnxPipeline(detector_path, recognizer_path)

    print("\n[SUCCESS] Models loaded and ready for inference!\n")

    # Continuous inference loop
    while True:
        image_name = prompt("Enter image file name (or type 'quit' to exit): ")

        if image_name.lower() in ["quit", "q", "exit"]:
            print("Shutting down OCR engine...")
            break

        if not image_name:
            continue

        if not os.path.isabs(image_name):
            image_path = os.path.join(script_dir, image_name)
        else:
            image_path = image_name

        if not os.path.exists(image_path):
            print(f"[ERROR] File not found: {image_path}\n")
            continue

        print(f"Running OCR on: {image_name}...")

        try:
            # Load image as RGB
            pil_img = Image.open(image_path).convert("RGB")
            rgb_np = np.array(pil_img)

            # Run OCR
            results = pipeline.predict(rgb_np)

            if not results:
                print("No text detected.\n")
                continue

            print("Predicted texts & confidence:")
            print("-" * 50)

            # Draw bounding boxes on the image
            draw = ImageDraw.Draw(pil_img)

            for box_coords, text, conf in results:
                print(f"  {text:<40}  {conf:.4f}")

                # Draw bounding box
                if isinstance(box_coords[0], tuple):
                    # Free box: 4 corners ((x1,y1), (x2,y2), (x3,y3), (x4,y4))
                    points = [coord for corner in box_coords for coord in corner]
                    draw.polygon(points, outline="green", width=2)
                else:
                    # Horizontal box: (xmin, xmax, ymin, ymax)
                    xmin, xmax, ymin, ymax = box_coords
                    draw.rectangle([xmin, ymin, xmax, ymax], outline="green", width=2)

            # Save annotated output
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(os.path.dirname(image_path), f"{base_name}_output.png")
            pil_img.save(output_path)

            print("-" * 50)
            print(f"Annotated image saved to: {output_path}\n")

        except Exception as e:
            print(f"[ERROR] Something went wrong during inference: {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
