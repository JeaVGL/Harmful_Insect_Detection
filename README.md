# Harmful_Insect_Detection

This repository contains everything needed to train, convert, and deploy a **lightweight multi-object detection model** for **real-time insect detection** on an **ESP32-S3** microcontroller.

The system is designed for **low-power agricultural monitoring**: detecting multiple insect species in a single camera frame, entirely **on-device**, without cloud inference.

---

##  Features

- **Lightweight model** based on MobileNetV2 + YOLO-style head
- Detects **multiple insects per frame** (24 species in dataset)
- Fully **on-device inference** on ESP32-S3
- **TFLite INT8 quantization** for small size & speed
- Ready-to-use **Arduino sketch** for real-time camera detection
- Companion **user app** for visualization and control

---

##  Requirements

### For training & conversion
- Python 3.10+
- TensorFlow 2.x (with GPU support recommended)
- OpenCV, NumPy, Matplotlib
- Pascal VOC-style dataset (XML annotations)

// The requirements.txt file describes what environment was used (not all packages featured might be useful for the training / export).


### For deployment
- Arduino IDE with **ESP32-S3 board support** (the ESP core by ESPRESSIF is required).
- ESP32-S3 board (tested on ESP32-S3-EYE, ESP32-S3-WROOM)
- OV2640 camera module


---
