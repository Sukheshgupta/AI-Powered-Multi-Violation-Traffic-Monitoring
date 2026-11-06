import os
from ultralytics import YOLO

UPLOAD_DIR = 'uploads'
OUT_DIR = os.path.join(UPLOAD_DIR, 'results')
os.makedirs(OUT_DIR, exist_ok=True)

model_path = 'best.pt'
if not os.path.exists(model_path):
    print('Model file best.pt not found in repo root.')
    raise SystemExit(1)

model = YOLO(model_path)

image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.jfif', '.webp', '.avif'}
images = [f for f in os.listdir(UPLOAD_DIR) if os.path.splitext(f)[1].lower() in image_exts]
if not images:
    print('No images found in uploads/.')
    raise SystemExit(0)

summary = []
for img in images:
    path = os.path.join(UPLOAD_DIR, img)
    print('Processing', path)
    try:
        results = model.predict(source=path, save=True, save_dir=OUT_DIR, conf=0.25, imgsz=640)
        r = results[0]
        names = r.names if hasattr(r, 'names') else {}
        boxes = getattr(r, 'boxes', None)
        dets = []
        if boxes is not None and len(boxes) > 0:
            cls_arr = boxes.cls.cpu().numpy() if hasattr(boxes, 'cls') else []
            conf_arr = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else []
            for i, c in enumerate(cls_arr):
                label = names.get(int(c), str(int(c)))
                conf = float(conf_arr[i]) if i < len(conf_arr) else 0.0
                dets.append((label, conf))
        summary.append((img, len(dets), dets))
        print(f"Saved annotated result for {img}")
    except Exception as e:
        print('Error processing', img, e)

print('\nSummary:')
for img, count, dets in summary:
    print(f'- {img}: {count} detections')
    for lab, conf in dets:
        print(f'   - {lab} {conf:.2f}')

print('\nAnnotated images saved to', OUT_DIR)
