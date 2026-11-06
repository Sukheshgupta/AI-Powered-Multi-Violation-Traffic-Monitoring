Training scaffold

- Place your dataset under `training/dataset/` in YOLO format:
  - `images/train`, `images/val`
  - `labels/train`, `labels/val` (YOLO .txt: class x y w h normalized)
- Edit `training/data.yaml` to point to your dataset and adjust `nc`/`names`.

Quick train command (Ultralytics YOLOv8 installed):

```bash
# from repo root
yolo detect train data=training/data.yaml model=yolov8n.pt epochs=100 imgsz=640 batch=16 device=0 name=exp1
```

Or use the Python API:

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='training/data.yaml', epochs=100, imgsz=640, batch=16, device=0, name='exp1')
```

Recommendations:
- Start with `yolov8n` or `yolov8s` and scale up.
- Use Mosaic, MixUp, AutoAugment and random perspective.
- Monitor validation mAP and do error analysis with FiftyOne or simple visualization scripts.
- For OCR, crop plates and train a dedicated recognizer (CRNN/TrOCR) or fine-tune Tesseract.
