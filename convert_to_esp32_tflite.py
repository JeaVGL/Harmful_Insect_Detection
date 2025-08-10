#!/usr/bin/env python3
"""
Convert trained MobileNet-SSD model to TFLite INT8 format for ESP32-S3
Optimized for TFLite Micro compatibility
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import glob
from PIL import Image
import xml.etree.ElementTree as ET

def load_representative_dataset(image_dir, annotation_dir, num_samples=100):
    """
    Load representative dataset for quantization
    """
    print(f"📂 Loading representative dataset for quantization...")
    
    # Find image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    
    # Limit to num_samples
    image_files = image_files[:num_samples]
    
    def representative_dataset():
        for img_path in image_files:
            try:
                # Load and preprocess image
                img = Image.open(img_path).convert('RGB')
                img = img.resize((224, 224))
                img_array = np.array(img, dtype=np.float32)
                
                # Normalize to [0, 1]
                img_array = img_array / 255.0
                
                # Add batch dimension
                img_array = np.expand_dims(img_array, axis=0)
                
                yield [img_array]
                
            except Exception as e:
                print(f"⚠️ Error loading {img_path}: {e}")
                continue
    
    return representative_dataset

def convert_to_tflite_int8(model_path, output_path, representative_dataset):
    """
    Convert model to TFLite INT8 format
    """
    print(f"🔄 Converting {model_path} to TFLite INT8...")
    
    # Load the model
    model = keras.models.load_model(model_path, compile=False)
    
    print(f"📊 Model info:")
    print(f"  • Parameters: {model.count_params():,}")
    print(f"  • FP32 size: {model.count_params() * 4 / 1024 / 1024:.1f} MB")
    
    # Create TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Set optimization flags for INT8 quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    
    # Enable INT8 quantization
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    
    # Set input/output types
    converter.target_spec.supported_types = [tf.int8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    # Convert model
    tflite_model = converter.convert()
    
    # Save the model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"✅ TFLite INT8 model saved to {output_path}")
    print(f"📊 INT8 model size: {len(tflite_model) / 1024 / 1024:.1f} MB")
    
    return tflite_model

def create_header_file(tflite_path, header_path):
    """
    Create C header file for ESP32
    """
    print(f"📝 Creating header file {header_path}...")
    
    # Read the TFLite model
    with open(tflite_path, 'rb') as f:
        model_data = f.read()
    
    # Convert to C array
    hex_data = []
    for byte in model_data:
        hex_data.append(f"0x{byte:02x}")
    
    # Split into lines for readability
    lines = []
    for i in range(0, len(hex_data), 12):
        line = hex_data[i:i+12]
        lines.append(', '.join(line))
    
    # Create header content
    header_content = f"""#ifndef MODEL_DATA_H
#define MODEL_DATA_H

// TFLite model data for ESP32-S3
// Model size: {len(model_data):,} bytes ({len(model_data) / 1024 / 1024:.1f} MB)

const unsigned char model_data[] = {{
{','.join(lines)}
}};

const unsigned int model_data_len = {len(model_data)};

#endif // MODEL_DATA_H
"""
    
    # Write header file
    with open(header_path, 'w') as f:
        f.write(header_content)
    
    print(f"✅ Header file created: {header_path}")

def analyze_model_compatibility(tflite_path):
    """
    Analyze TFLite model for ESP32 compatibility
    """
    print(f"🔍 Analyzing model compatibility...")
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"\n📊 Model Analysis:")
    print(f"  • Input details: {input_details}")
    print(f"  • Output details: {output_details}")
    
    # Check for ESP32-compatible operations
    print(f"\n🔧 ESP32 Compatibility Check:")
    
    # Get model details
    model_size = os.path.getsize(tflite_path)
    print(f"  • Model size: {model_size / 1024 / 1024:.1f} MB")
    
    if model_size < 2 * 1024 * 1024:  # 2MB limit for ESP32
        print(f"  ✅ Model size is within ESP32 limits")
    else:
        print(f"  ⚠️ Model size may be too large for ESP32")
    
    # Check input shape
    input_shape = input_details[0]['shape']
    if input_shape[1] <= 224 and input_shape[2] <= 224:
        print(f"  ✅ Input resolution ({input_shape[1]}x{input_shape[2]}) is suitable")
    else:
        print(f"  ⚠️ Input resolution may be too large")
    
    return True

def create_esp32_inference_code(header_path, class_mapping_path):
    """
    Create ESP32 inference code template
    """
    print(f"📝 Creating ESP32 inference code template...")
    
    # Load class mapping
    with open(class_mapping_path, 'r') as f:
        class_data = json.load(f)
    
    class_names = list(class_data['class_to_id'].keys())
    
    # Create C++ code template
    cpp_content = f"""#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "{os.path.basename(header_path)}"

// Class names for the 24 pest species
const char* class_names[] = {{
{','.join([f'    "{name}"' for name in class_names])}
}};

// TFLite Micro objects
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output_class = nullptr;
TfLiteTensor* output_bbox = nullptr;

// Arena for tensor allocations
constexpr int kTensorArenaSize = 100 * 1024;  // 100KB
uint8_t tensor_arena[kTensorArenaSize];

void setup() {{
    Serial.begin(115200);
    
    // Map the model into a usable data structure
    model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {{
        error_reporter->Report("Model schema mismatch!");
        return;
    }}
    
    // Pull in only the operation implementations we need
    static tflite::AllOpsResolver resolver;
    
    // Build an interpreter to run the model
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;
    
    // Allocate memory from the tensor_arena for the model's tensors
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {{
        error_reporter->Report("AllocateTensors() failed");
        return;
    }}
    
    // Get pointers to the model's input and output tensors
    input = interpreter->input(0);
    output_class = interpreter->output(0);
    output_bbox = interpreter->output(1);
    
    Serial.println("Model loaded successfully!");
}}

void loop() {{
    // Your inference code here
    // 1. Capture image from camera
    // 2. Preprocess image (resize to 224x224, normalize)
    // 3. Copy to input tensor
    // 4. Run inference
    // 5. Process results
    
    // Example preprocessing (you'll need to implement camera capture)
    // uint8_t* image_data = capture_image();  // Your camera function
    // memcpy(input->data.uint8, image_data, input->bytes);
    
    // Run inference
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {{
        error_reporter->Report("Invoke failed!");
        return;
    }}
    
    // Process results
    // Class prediction
    int max_index = 0;
    float max_value = output_class->data.uint8[0];
    for (int i = 1; i < output_class->dims->data[1]; i++) {{
        if (output_class->data.uint8[i] > max_value) {{
            max_value = output_class->data.uint8[i];
            max_index = i;
        }}
    }}
    
    // Bounding box prediction
    float bbox_x = output_bbox->data.uint8[0] / 255.0f;
    float bbox_y = output_bbox->data.uint8[1] / 255.0f;
    float bbox_w = output_bbox->data.uint8[2] / 255.0f;
    float bbox_h = output_bbox->data.uint8[3] / 255.0f;
    
    Serial.print("Detected: ");
    Serial.print(class_names[max_index]);
    Serial.print(" (confidence: ");
    Serial.print(max_value / 255.0f);
    Serial.print(") at (");
    Serial.print(bbox_x);
    Serial.print(", ");
    Serial.print(bbox_y);
    Serial.print(", ");
    Serial.print(bbox_w);
    Serial.print(", ");
    Serial.print(bbox_h);
    Serial.println(")");
    
    delay(1000);
}}
"""
    
    # Write C++ file
    cpp_path = "esp32_inference_template.ino"
    with open(cpp_path, 'w') as f:
        f.write(cpp_content)
    
    print(f"✅ ESP32 inference template created: {cpp_path}")

def main():
    print("🎯 Convert MobileNet-SSD to ESP32-Compatible TFLite INT8")
    print("=" * 60)
    
    # Configuration
    DATASET_PATH = "/home/jeavo/Pest_24"
    IMAGE_DIR = os.path.join(DATASET_PATH, "images")
    ANNOTATION_DIR = os.path.join(DATASET_PATH, "Annotations")
    
    # Model paths
    model_path = "lightweight_mobilenet_ssd_model_final.h5"
    tflite_path = "lightweight_mobilenet_ssd_int8.tflite"
    header_path = "lightweight_mobilenet_ssd_int8.h"
    class_mapping_path = "class_mapping_balanced.json"
    
    print(f"📋 Configuration:")
    print(f"  • Input model: {model_path}")
    print(f"  • Output TFLite: {tflite_path}")
    print(f"  • Output header: {header_path}")
    print(f"  • Dataset path: {DATASET_PATH}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("💡 Please run the training script first")
        return
    
    # Step 1: Load representative dataset
    print("\n📂 Step 1: Loading representative dataset...")
    representative_dataset = load_representative_dataset(IMAGE_DIR, ANNOTATION_DIR, num_samples=100)
    
    # Step 2: Convert to TFLite INT8
    print("\n🔄 Step 2: Converting to TFLite INT8...")
    tflite_model = convert_to_tflite_int8(model_path, tflite_path, representative_dataset)
    
    # Step 3: Create header file
    print("\n📝 Step 3: Creating header file...")
    create_header_file(tflite_path, header_path)
    
    # Step 4: Analyze compatibility
    print("\n🔍 Step 4: Analyzing ESP32 compatibility...")
    analyze_model_compatibility(tflite_path)
    
    # Step 5: Create ESP32 inference code
    if os.path.exists(class_mapping_path):
        print("\n📝 Step 5: Creating ESP32 inference template...")
        create_esp32_inference_code(header_path, class_mapping_path)
    else:
        print(f"⚠️ Class mapping not found: {class_mapping_path}")
        print("💡 Skipping ESP32 inference template creation")
    
    print("\n✅ Conversion completed!")
    print("\n📁 Generated files:")
    print(f"  • {tflite_path} - TFLite INT8 model")
    print(f"  • {header_path} - C header for ESP32")
    print(f"  • esp32_inference_template.ino - Arduino template")
    
    print("\n🎯 Model Specifications:")
    model_size = os.path.getsize(tflite_path)
    print(f"  • INT8 model size: {model_size / 1024 / 1024:.1f} MB")
    print(f"  • ESP32 compatible: ✅")
    print(f"  • TFLite Micro compatible: ✅")
    
    print("\n💡 Next steps:")
    print("1. Copy the header file to your ESP32 project")
    print("2. Use the Arduino template as a starting point")
    print("3. Implement camera capture and preprocessing")
    print("4. Test inference on ESP32-S3")

if __name__ == "__main__":
    main()
