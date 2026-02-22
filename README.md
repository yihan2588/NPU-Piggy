# OCR TFLite Dataset Tester

This workspace now includes `ocr_tflite_test.py` to run your:
- Detector: `export_assets/easyocr-tflite-w8a8/EasyOCRDetector.tflite`
- Recognizer: `export_assets/easyocr-tflite-w8a8/EasyOCRRecognizer.tflite`

against images in `dataset/`.

## 1) Install dependencies

```powershell
python -m pip install -r requirements.txt
```

## 2) Run OCR test

```powershell
python ocr_tflite_test.py \
  --dataset dataset \
  --detector-model export_assets/easyocr-tflite-w8a8/EasyOCRDetector.tflite \
  --recognizer-model export_assets/easyocr-tflite-w8a8/EasyOCRRecognizer.tflite \
  --output-dir outputs
```

## 3) Outputs

- `outputs/ocr_results.json` (all OCR rows)
- `outputs/ocr_results.csv` (tabular output)
- `outputs/visualizations/` (images with boxes + text overlays)

## Optional tuning

```powershell
python ocr_tflite_test.py --text-threshold 0.50 --min-area 50
```

If recognizer text quality is weak, provide your own charset file:

```powershell
python ocr_tflite_test.py --charset-file path\to\charset.txt
```