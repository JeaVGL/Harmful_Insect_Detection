# Harmful_Insect_Detection

This repository contains everything needed to **train, convert, test, and deploy** a **lightweight multi‑object detection model** for **real‑time insect detection** on an **ESP32‑S3** microcontroller.

The system targets **low‑power agricultural monitoring**: detect multiple insect species in a single camera frame, entirely **on‑device** — no cloud inference required.

---

## Features

- **Lightweight model** based on MobileNetV2 + YOLO‑style head  
- Detects **multiple insects per frame** (24 species in the dataset)  
- Fully **on‑device inference** on ESP32‑S3  
- **TFLite INT8** quantization for small size & speed  
- Ready‑to‑use **Arduino sketch** (ESP32‑S3 + OV2640) for live camera detection  
- **User app** (companion) for visualization and control

---

## Requirements

### For training & conversion
- Python 3.10+
- TensorFlow 2.x (GPU support recommended)
- OpenCV, NumPy, Matplotlib
- Pascal VOC–style dataset (XML annotations)

> The `requirements.txt` describes the environment used (some packages may not be strictly necessary for training/export).

### For deployment
- Arduino IDE with **ESP32‑S3** board support (Espressif core)
- ESP32‑S3 board (tested on ESP32‑S3‑EYE, ESP32‑S3‑WROOM)
- OV2640 camera module

---

## Detailed explanations for each file

### Model training

As of **August 2025**, **MobileNetV2‑based** models are the only approach that consistently fits ESP32 constraints **and** runs reliably end‑to‑end in TFLite Micro. (YOLO‑nano variants exceeded memory; NanoDet variants produced zero outputs under TFLite Micro on Arduino.)

- The training script expects a **Pascal VOC XML** dataset (e.g., Pest24 from Kaggle).  
- To use another format (e.g., YOLO), adapt the loader. Since the script encodes labels into a **YOLO‑like** format internally, adapting for YOLO datasets is straightforward.  
- Data split into **train/val/test** is handled by `get_dataloaders` in `dataloader_patch.py` — you don’t need a manual split.  
- **Class imbalance** (notably in Pest24) is mitigated via per‑class counts → **class weights** → **weighted sampling** in a custom generator.  
- Input resolution is **224×224 RGB**. ESP32‑S3 compute headroom may allow increases, but verify memory and latency.

**Expected dataset structure:**
DATASET_PATH/<br>
├── images/<br>
│   ├── image1.jpg<br>
│   ├── image2.jpg<br>
│   └── ...<br>
└── Annotations/<br>
├── image1.xml<br>
├── image2.xml<br>
└── ...<br>

Each `.xml` must include:
- `<object>` with `<name>` and `<bndbox>`: `xmin`, `ymin`, `xmax`, `ymax`  
- `<size>` with `width`, `height`

All classes are collected automatically to build the label map.

**Key configuration parameters**
| Parameter         | Default | Meaning |
|-------------------|---------|---------|
| `IMG_SIZE`        | 224     | Input image size (square) |
| `ALPHA`           | 1.0     | MobileNetV2 width multiplier (smaller = lighter model) |
| `FREEZE_BACKBONE` | False   | Whether to freeze MobileNetV2 weights |
| `BATCH_SIZE`      | 16      | Training batch size |
| `EPOCHS`          | 60      | Maximum number of training epochs |
| `A`               | 3       | Anchors per grid cell |
| `ANCHORS`         | `[[0.10, 0.08], [0.18, 0.15], [0.28, 0.24]]` | Anchor sizes relative to input dims |

**TO USE THE TRAINING SCRIPT:**  
Update the dataset path and ensure it matches the structure above. Then run the training entry point as documented in the script.

---

### Model export and quantization

The export/quantization script converts the trained **`.h5`** model to a **`.tflite`** model and generates a **C header**.

- Quantization calibration uses ~100 **camera‑like** images (224×224×3).  
- Quantization is **full INT8** (inputs, weights, outputs) to **minimize model size**.  
- The script validates the converted model by checking input/output tensor types & shapes, runs a **random INT8 inference**, and compares float vs quantized outputs.  
- Model characteristics are dumped to a **JSON** file.

**C header generation (single header):**
```bash
xxd -i <tflite_model.tflite> > <output_C_header>.h
# Example:
# xxd -i mdet_int8_final.tflite > mdet_int8_final_model.h
````
NOTE: Some tools generate .h + .c arrays, but Arduino typically expects a single .h.
**TO USE THE EXPORT AND QUANTIZATION SCRIPT:**  
Ensure the `.h5` model path/name matches your output from training. (Default names align with the training script, and both scripts live in the same directory.) After export, inspect the JSON to confirm everything looks correct.

---

### Offline testing on images — `test_inference.py`

This utility runs the **quantized TFLite** model on local PNG images and saves annotated results.

**What it does**
- Loads the TFLite model (default: `mdet_int8_final.tflite`).  
- Infers quantization params (scale/zero‑point) and handles **INT8 preprocessing**.  
- Applies **YOLO‑style decoding** (S×S grid, 3 anchors) and **NMS**.  
- Uses the built‑in **24‑class** label list and generates colored overlays.

**Folder & file expectations**
project_root/<br>
├── mdet_int8_final.tflite<br>
├── test_inference.py<br>
└── test_img_inference/<br>
├── img_001.png<br>
├── img_002.png<br>
└── ...<br>

- Model path default: `mdet_int8_final.tflite` (same directory as the script).  
- Test images must be **PNG** files in `test_img_inference/`.  
- Results are saved as `results_<image>.png`; an additional copy with info in the filename (e.g., `detected_<class>_<score>_<image>.png`) is also saved.

**Run it**
```bash
python3 test_inference.py
```
- Default **confidence threshold** is `0.25`; change by editing `InsectDetector(model_path, conf_threshold=0.25)`.  
- IoU for NMS is `0.45`.  
- The script prints input/output tensor **shapes & dtypes**, number of **detections**, and per‑detection **class / score / box (normalized)**.

**TO USE THE TEST INFERENCE SCRIPT:**  
Place your TFLite model at the script root (or adjust `model_path`), put PNG images into `test_img_inference/`, then run `python3 test_inference.py`. Check console logs and saved result images for detections.

---

### Arduino deployment

The Arduino sketch performs **real‑time capture** from OV2640, converts frames to RGB, runs **TFLite Micro INT8** inference, applies **post‑processing** (decoding + NMS), and serves results via the device’s web UI.

**Files required in the sketch directory**
- `pest_class_names.h` — maps **class indices → names** (used to render human‑readable labels).  
- The model as a **single C header** (e.g., `mdet_int8_final_model.h`) created with `xxd -i`.  
  - Due to ESP32 limits, the header should be **well under ~10 MB**, otherwise the **APP partition** may overflow when linking the sketch binary.

**Arduino IDE configuration**
- **Board:** *ESP32S3 Dev Module* (install Espressif’s ESP32 core via Boards Manager).  
- **Additional Boards Manager URLs:**

https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_dev_index.json,
https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json

- **USB CDC On Boot:** Disabled  
- **Flash Size:** 16MB (128Mb)  
- **Partition Scheme:** **16MB Flash (3MB APP / 9.9MB FATFS)** — best non‑custom option to fit binary + assets  
- **PSRAM:** Enabled → **OPI PSRAM**

**Libraries / packages**
- `Arduino.h`, `WiFi.h`, `WebServer.h` — default (ESP core)  
- `WiFiManager.h` — install via Library Manager  
- `Chirale_TensorFlowLite` — install via Library Manager  
- Rationale: reliable wrapper for TFLite Micro on ESP32; many alternatives online are deprecated.  
- (Depending on your sketch) TFLite Micro headers may also be included directly, e.g.:  
`tensorflow/lite/micro/all_ops_resolver.h`, `tensorflow/lite/micro/micro_interpreter.h`, `tensorflow/lite/schema/schema_generated.h`

**Camera configuration**
- Current GPIO mapping fits the tested ESP32‑S3 dev module; other boards may require pin map edits.  
- `frame_size`: **QVGA (320×240)** works well for memory & latency.  
- **Pixel format:** capture as **JPEG**, then convert to RGB using Arduino’s JPEG decoder.  
- Capturing directly as RGB caused setup‑time memory crashes; JPEG → RGB conversion has been stable.  
- Keep `fb_location = CAMERA_FB_IN_PSRAM` to avoid DRAM overflow.

**Model preprocessing**
- The sketch expects **INT8** input with simple normalization matching the training/export pipeline.  
- Ensure your header model uses preprocessing consistent with training so quantization ranges align.

**Build size & memory tips**
- If you hit **“text section exceeds available space”**:  
- Reduce model size (smaller `ALPHA`, fewer heads/anchors, prune classes).  
- Ensure the **16MB (3MB APP / 9.9MB FATFS)** partition is selected.  
- Strip optional debug, remove web debug endpoints & assets.  
- Keep **PSRAM enabled** and frame buffers in PSRAM.  
- Avoid large global arrays and unnecessary static buffers.

---

## Quick repo checklist
- [ ] Dataset organized as shown (VOC XML + images)  
- [ ] Train with the provided script; verify logs and saved checkpoints  
- [ ] Export to INT8 TFLite; validate JSON; generate `.h` with `xxd -i`  
- [ ] Test locally with `test_inference.py` on sample PNGs  
- [ ] Flash Arduino sketch with correct board, partitions, PSRAM & libraries  
- [ ] Confirm live detections and memory stability
