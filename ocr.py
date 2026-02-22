import sys
import os
import easyocr
from PIL import Image, ImageDraw

def prompt(msg):
    """Print a prompt and read a line — works correctly in Git Bash."""
    print(msg, end="", flush=True)
    return sys.stdin.readline().strip()

def main():
    print("=== EasyOCR Inference Engine ===", flush=True)
    print("Initializing ML libraries and loading weights into memory...")

    # --- 1. Initialize pure EasyOCR ---
    # We use gpu=False because x64 emulation on ARM64 relies on CPU execution
    reader = easyocr.Reader(['en'], gpu=False)
    
    print("\n[SUCCESS] Model is warm and loaded in memory! Ready for rapid inference.\n")

    # --- 2. Continuous Inference Loop ---
    while True:
        image_name = prompt("Enter image file name (or type 'quit' to exit): ")
        
        if image_name.lower() in ['quit', 'q', 'exit']:
            print("Shutting down OCR engine...")
            break
            
        if not image_name:
            continue

        if not os.path.isabs(image_name):
            image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), image_name)
        else:
            image_path = image_name

        if not os.path.exists(image_path):
            print(f"[ERROR] File not found: {image_path}\n")
            continue

        print(f"Running OCR on: {image_name}...")
        try:
            # --- 3. THE FIX: width_ths=1.0 ---
            # This forces the detector to merge text blocks that are separated horizontally
            results = reader.readtext(image_path, width_ths=1.0)

            if not results:
                print("No text detected.\n")
                continue

            print("Predicted texts & confidence:")
            print("-" * 40)
            
            # Load the original image so we can draw the bounding boxes
            img_pil = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(img_pil)

            for bbox, text, conf in results:
                print(f"  {text:<30}  {conf:.4f}")
                
                # bbox contains the 4 corner coordinates: [top_left, top_right, bottom_right, bottom_left]
                p0, p1, p2, p3 = bbox
                # Draw a green polygon around the merged text
                draw.polygon([*p0, *p1, *p2, *p3], outline="green", width=2)

            # --- 4. Save annotated output ---
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(os.path.dirname(image_path), f"{base_name}_output.png")
            img_pil.save(output_path)
            
            print("-" * 40)
            print(f"Annotated image saved to: {output_path}\n")
            
        except Exception as e:
            print(f"[ERROR] Something went wrong during inference: {e}\n")

if __name__ == "__main__":
    main()
