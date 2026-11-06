from ultralytics import YOLO
import cv2

# Load the pre-trained YOLOv8 model
model = YOLO('best.pt')

# Load your input image
image = cv2.imread('images.jpg')

# Perform inference
results = model(image)

# The results are in a list, so access the first (and only) item in the list
result = results[0]  # This is a detection result object

# Get the bounding boxes (xywh format) and class labels from the result
boxes = result.boxes.xywh.cpu().numpy()  # Get bounding boxes in xywh format
labels = result.names  # Class names (labels)

# Loop through the detected objects and draw bounding boxes
for i, box in enumerate(boxes):
    x_center, y_center, width, height = box  # Extract xywh
    class_id = int(result.boxes.cls[i])  # Get the class index from the result
    label_name = labels[class_id]  # Get the class name from the class index

    # Convert from xywh to x1, y1, x2, y2 for drawing
    x1 = int((x_center - width / 2))
    y1 = int((y_center - height / 2))
    x2 = int((x_center + width / 2))
    y2 = int((y_center + height / 2))

    # Draw bounding box and label on the image
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw box
    cv2.putText(image, label_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the image with bounding boxes and labels
cv2.imshow('Detection Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, save the output image with bounding boxes
cv2.imwrite('image_with_boxes.jpg', image)
