#!/bin/bash
# Template to run YOLOv8 training locally
# Edit data path in training/data.yaml first

MODEL=${1:-yolov8n.pt}
EPOCHS=${2:-100}
IMGSZ=${3:-640}
BATCH=${4:-16}
NAME=${5:-exp1}

yolo detect train data=training/data.yaml model=$MODEL epochs=$EPOCHS imgsz=$IMGSZ batch=$BATCH device=0 name=$NAME
