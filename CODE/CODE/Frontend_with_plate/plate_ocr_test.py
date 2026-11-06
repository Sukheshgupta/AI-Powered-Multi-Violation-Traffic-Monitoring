import argparse
import os
import re

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import easyocr
except ImportError:
    easyocr = None

try:
    import pytesseract
except ImportError:
    pytesseract = None


PLATE_MODEL_PATH = 'plate_best.pt'
UNKNOWN_TEXT = 'UNKNOWN'


def normalize_plate_text(value):
    if not value:
        return ''
    return ''.join(ch for ch in value.upper() if ch.isalnum())


def is_reasonable_plate(value):
    return bool(
        re.fullmatch(r'[A-Z0-9]{6,12}', value or '')
        and any(ch.isalpha() for ch in value)
        and any(ch.isdigit() for ch in value)
    )


def load_plate_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f'Plate model not found: {model_path}. Train/export plate_best.pt first.'
        )
    return YOLO(model_path)


def detect_best_plate(model, image, confidence):
    result = model.predict(image, conf=confidence, verbose=False)[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None, None

    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros(len(boxes))
    best_index = int(np.argmax(confs))
    xyxy = boxes.xyxy.cpu().numpy()[best_index]
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None, None

    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = int(box_w * 0.16)
    pad_y = int(box_h * 0.35)
    crop_x1 = max(0, x1 - pad_x)
    crop_y1 = max(0, y1 - pad_y)
    crop_x2 = min(w, x2 + pad_x)
    crop_y2 = min(h, y2 + pad_y)
    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
    return crop, (x1, y1, x2, y2, float(confs[best_index]))


def preprocess_plate(crop):
    if crop is None or crop.size == 0:
        return None

    h, w = crop.shape[:2]
    if w > 0 and w < 420:
        scale = 420 / float(w)
        crop = cv2.resize(crop, (420, int(h * scale)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    return gray


def read_with_easyocr(crop):
    if easyocr is None:
        return ''
    try:
        reader = easyocr.Reader(['en'], gpu=False)
        results = reader.readtext(crop, detail=0, paragraph=False)
        candidates = [normalize_plate_text(text) for text in results]
        candidates = [text for text in candidates if is_reasonable_plate(text)]
        return candidates[0] if candidates else ''
    except Exception:
        return ''


def read_with_tesseract(crop):
    if pytesseract is None:
        return ''
    try:
        processed = preprocess_plate(crop)
        if processed is None:
            return ''
        configs = [
            '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        ]
        for config in configs:
            raw_text = pytesseract.image_to_string(processed, config=config)
            for token in re.findall(r'[A-Z0-9]{6,12}', raw_text.upper()):
                candidate = normalize_plate_text(token)
                if is_reasonable_plate(candidate):
                    return candidate
    except Exception:
        return ''
    return ''


def read_plate_text(crop, ocr_engine):
    if crop is None or crop.size == 0:
        return UNKNOWN_TEXT

    if ocr_engine in {'easyocr', 'auto'}:
        text = read_with_easyocr(crop)
        if text:
            return text

    if ocr_engine in {'tesseract', 'auto'}:
        text = read_with_tesseract(crop)
        if text:
            return text

    return UNKNOWN_TEXT


def annotate_image(image, box, plate_text):
    output = image.copy()
    if box is None:
        return output

    x1, y1, x2, y2, confidence = box
    label = f'{plate_text} {confidence:.2f}'
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 220, 80), 2)
    cv2.putText(
        output,
        label,
        (x1, max(24, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 220, 80),
        2,
        cv2.LINE_AA,
    )
    return output


def main():
    parser = argparse.ArgumentParser(description='Detect and OCR a number plate from one image.')
    parser.add_argument('image_path', help='Path to the test image')
    parser.add_argument('--model', default=PLATE_MODEL_PATH, help='Path to plate_best.pt')
    parser.add_argument('--conf', type=float, default=0.25, help='YOLO confidence threshold')
    parser.add_argument('--ocr', choices=['auto', 'easyocr', 'tesseract'], default='auto')
    parser.add_argument('--out-dir', default='plate_ocr_outputs', help='Output folder')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    try:
        model = load_plate_model(args.model)
    except FileNotFoundError as exc:
        print(exc)
        return 1

    image = cv2.imread(args.image_path)
    if image is None:
        print(f'Unable to read image: {args.image_path}')
        return 1

    crop, box = detect_best_plate(model, image, args.conf)
    if crop is None:
        print('No plate detected.')
        plate_text = UNKNOWN_TEXT
        annotated = image.copy()
    else:
        plate_text = read_plate_text(crop, args.ocr)
        annotated = annotate_image(image, box, plate_text)
        cv2.imwrite(os.path.join(args.out_dir, 'cropped_plate.jpg'), crop)

    cv2.imwrite(os.path.join(args.out_dir, 'annotated_output.jpg'), annotated)
    print(f'Detected plate text: {plate_text}')
    print(f'Outputs saved to: {args.out_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
