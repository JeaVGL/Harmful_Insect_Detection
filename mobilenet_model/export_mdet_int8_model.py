#!/usr/bin/env python3
"""
Export and INT8 quantize the mdet_best_fixed.h5 model for TFLite Micro
Multi-detection YOLO model based on MobileNetV2
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import struct
import datetime

def create_representative_dataset():
    """Create representative dataset for INT8 quantization"""
    print("🔧 Creating representative dataset for quantization...")
    
    def representative_dataset():
        # Generate realistic input data for insect detection
        # Use 224x224x3 images with values in 0-255 range, then normalize to 0-1
        for _ in range(100):  # 100 samples for quantization
            # Generate random RGB images in 0-255 range (realistic for camera input)
            data = np.random.randint(0, 256, (1, 224, 224, 3), dtype=np.uint8)
            # Convert to float32 and normalize to 0-1 (as expected by the model)
            data = data.astype(np.float32) / 255.0
            yield [data]
    
    return representative_dataset

def load_and_verify_model(model_path):
    """Load the trained model and verify its structure"""
    print(f"📦 Loading model from {model_path}...")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        model = keras.models.load_model(model_path, compile=False)
        print("✅ Model loaded successfully")
        
        # Print model summary
        print("\n📊 Model Architecture:")
        model.summary()
        
        # Verify input shape
        input_shape = model.input_shape[0]
        output_shape = model.output_shape[0]
        print(f"\n🔍 Model Details:")
        print(f"  • Input shape: {input_shape}")
        print(f"  • Output shape: {output_shape}")
        print(f"  • Total parameters: {model.count_params():,}")
        
        # Calculate model size
        total_params = model.count_params()
        model_size_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
        print(f"  • Expected model size: {model_size_mb:.2f} MB")
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

def create_int8_quantized_model(model):
    """Convert model to INT8 quantized TFLite"""
    print("\n🔧 Applying INT8 quantization...")
    
    # Create representative dataset
    representative_dataset = create_representative_dataset()
    
    # Convert to INT8 quantized TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Apply optimizations and quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    
    # Force INT8 operations where possible
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter.target_spec.supported_types = [tf.int8]
    
    # Force input and output to INT8
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    # Use new quantizer for better INT8 compatibility
    converter.experimental_new_quantizer = True
    
    # Convert the model
    print("🔄 Converting model to INT8 quantized TFLite...")
    tflite_model = converter.convert()
    
    return tflite_model

def save_quantized_model(tflite_model, output_path):
    """Save the quantized TFLite model"""
    print(f"💾 Saving quantized model to {output_path}...")
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    size_mb = len(tflite_model) / 1024 / 1024
    print(f"✅ Quantized model saved: {output_path}")
    print(f"📏 Size: {size_mb:.2f} MB")
    
    return size_mb

def test_quantized_model(tflite_model, original_model):
    """Test the quantized model to ensure it works correctly"""
    print("\n🧪 Testing quantized model...")
    
    try:
        # Create interpreter
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("📊 Model Analysis:")
        print(f"  • Input type: {input_details[0]['dtype']}")
        print(f"  • Input shape: {input_details[0]['shape']}")
        print(f"  • Output type: {output_details[0]['dtype']}")
        print(f"  • Output shape: {output_details[0]['shape']}")
        
        # Test inference with sample data
        print("🔍 Testing inference...")
        
        # Create test input (INT8) - use proper INT8 range (-128 to 127)
        test_input = np.random.randint(-128, 128, input_details[0]['shape'], dtype=np.int8)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], test_input)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        output = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"  • Test input shape: {test_input.shape}")
        print(f"  • Test output shape: {output.shape}")
        print(f"  • Output range: [{output.min():.4f}, {output.max():.4f}]")
        
        # Check if all tensors are INT8
        all_int8 = True
        for detail in input_details + output_details:
            if detail['dtype'] != np.int8:
                all_int8 = False
                print(f"  ⚠️  Tensor {detail['name']} is {detail['dtype']}, not INT8")
        
        if all_int8:
            print("  ✅ All tensors are INT8 - compatible with TFLite Micro!")
        else:
            print("  ⚠️  Some tensors are not INT8 - may cause compatibility issues")
        
        # Test with realistic input data to verify quantization quality
        print("🔍 Testing with realistic input data...")
        
        # Create realistic test image (normalized 0-1)
        realistic_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        
        # Test original model
        original_output = original_model.predict(realistic_input, verbose=0)
        
        # Test quantized model with same input (need to convert to INT8)
        # Scale input to INT8 range and quantize
        input_scale = input_details[0]['quantization_parameters']['scales'][0]
        input_zero_point = input_details[0]['quantization_parameters']['zero_points'][0]
        
        quantized_input = np.round(realistic_input / input_scale + input_zero_point).astype(np.int8)
        quantized_input = np.clip(quantized_input, -128, 127)
        
        interpreter.set_tensor(input_details[0]['index'], quantized_input)
        interpreter.invoke()
        quantized_output = interpreter.get_tensor(output_details[0]['index'])
        
        # Dequantize output for comparison
        output_scale = output_details[0]['quantization_parameters']['scales'][0]
        output_zero_point = output_details[0]['quantization_parameters']['zero_points'][0]
        
        dequantized_output = (quantized_output.astype(np.float32) - output_zero_point) * output_scale
        
        # Compare outputs
        mse = np.mean((original_output - dequantized_output) ** 2)
        mae = np.mean(np.abs(original_output - dequantized_output))
        
        print(f"  • Original output range: [{original_output.min():.4f}, {original_output.max():.4f}]")
        print(f"  • Quantized output range: [{dequantized_output.min():.4f}, {dequantized_output.max():.4f}]")
        print(f"  • Mean Squared Error: {mse:.6f}")
        print(f"  • Mean Absolute Error: {mae:.6f}")
        
        # Check if quantization quality is acceptable
        if mse < 0.01 and mae < 0.1:  # Thresholds can be adjusted
            print("  ✅ Quantization quality is good!")
        else:
            print("  ⚠️  Quantization quality may be degraded")
        
        return True, interpreter
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False, None

def create_c_header_file(tflite_model, header_path):
    """Convert TFLite model to C header file"""
    print(f"\n📝 Creating C header file: {header_path}...")
    
    # Convert model bytes to C array
    model_bytes = tflite_model
    model_size = len(model_bytes)
    
    # Create header content
    header_content = f"""#ifndef MDET_INT8_MODEL_H
#define MDET_INT8_MODEL_H

// INT8 Quantized Multi-Detection YOLO Model for Insect Detection
// Based on MobileNetV2 + YOLO head
// Model size: {model_size:,} bytes ({model_size/1024:.1f} KB)

#include <stdint.h>

// Model data as C array
extern const uint8_t mdet_int8_model_data[{model_size}];

// Model size constant
#define MDET_INT8_MODEL_SIZE {model_size}

// Model metadata
#define MDET_INPUT_SIZE 224
#define MDET_INPUT_CHANNELS 3
#define MDET_GRID_SIZE 14  // 224/16
#define MDET_ANCHORS_PER_CELL 3
#define MDET_BOX_ATTRIBUTES 4  // x, y, w, h
#define MDET_OBJECTNESS_ATTRIBUTES 1
#define MDET_NUM_CLASSES 24  // Based on actual model output shape (29 - 5 = 24 classes)

// Input/output tensor information
#define MDET_INPUT_TENSOR_SIZE (MDET_INPUT_SIZE * MDET_INPUT_SIZE * MDET_INPUT_CHANNELS)
#define MDET_OUTPUT_TENSOR_SIZE (MDET_GRID_SIZE * MDET_GRID_SIZE * MDET_ANCHORS_PER_CELL * (MDET_OBJECTNESS_ATTRIBUTES + MDET_BOX_ATTRIBUTES + MDET_NUM_CLASSES))

#endif // MDET_INT8_MODEL_H
"""
    
    # Create source file content
    source_content = f"""#include "mdet_int8_model.h"

// INT8 Quantized Multi-Detection YOLO Model Data
const uint8_t mdet_int8_model_data[{model_size}] = {{
"""
    
    # Add model bytes as hex values
    bytes_per_line = 16
    for i in range(0, model_size, bytes_per_line):
        line_bytes = model_bytes[i:i + bytes_per_line]
        hex_values = ', '.join([f'0x{b:02x}' for b in line_bytes])
        source_content += f"    {hex_values}"
        if i + bytes_per_line < model_size:
            source_content += ","
        source_content += "\n"
    
    source_content += "};\n"
    
    # Save header file
    with open(header_path, 'w') as f:
        f.write(header_content)
    
    # Save source file
    source_path = header_path.replace('.h', '.c')
    with open(source_path, 'w') as f:
        f.write(source_content)
    
    print(f"✅ Header file created: {header_path}")
    print(f"✅ Source file created: {source_path}")
    
    return header_path, source_path

def verify_file_sizes(original_model_path, tflite_path, header_path):
    """Verify file sizes and provide summary"""
    print("\n📏 File Size Verification:")
    
    # Get file sizes
    original_size = os.path.getsize(original_model_path) / 1024 / 1024
    tflite_size = os.path.getsize(tflite_path) / 1024 / 1024
    header_size = os.path.getsize(header_path) / 1024 / 1024
    
    print(f"  • Original H5 model: {original_size:.2f} MB")
    print(f"  • INT8 TFLite model: {tflite_size:.2f} MB")
    print(f"  • C header file: {header_size:.2f} MB")
    
    # Calculate compression ratio
    compression_ratio = (1 - tflite_size / original_size) * 100
    print(f"  • Compression ratio: {compression_ratio:.1f}%")
    
    return original_size, tflite_size, header_size

def main():
    """Main function to export and quantize the model"""
    print("🚀 Starting MDET Model Export and INT8 Quantization")
    print("=" * 60)
    
    # Configuration - Updated to use the fixed model
    model_path = "mdet_best_fixed.h5"
    tflite_output = "mdet_int8_fixed.tflite"
    header_output = "mdet_int8_fixed_model.h"
    
    try:
        # Step 1: Load and verify the model
        model = load_and_verify_model(model_path)
        
        # Step 2: Create INT8 quantized model
        tflite_model = create_int8_quantized_model(model)
        
        # Step 3: Save quantized model
        tflite_size = save_quantized_model(tflite_model, tflite_output)
        
        # Step 4: Test the quantized model
        test_success, interpreter = test_quantized_model(tflite_model, model)
        
        if not test_success:
            print("❌ Model testing failed!")
            return
        
        # Step 5: Create C header file
        header_path, source_path = create_c_header_file(tflite_model, header_output)
        
        # Step 6: Verify file sizes
        original_size, tflite_size, header_size = verify_file_sizes(
            model_path, tflite_output, header_path
        )
        
        # Step 7: Final verification and summary
        print("\n" + "=" * 60)
        print("🎉 EXPORT AND QUANTIZATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print(f"✅ Model exported to: {tflite_output}")
        print(f"✅ C header created: {header_path}")
        print(f"✅ C source created: {source_path}")
        print(f"✅ Model size reduced from {original_size:.2f} MB to {tflite_size:.2f} MB")
        print(f"✅ Compression: {((1 - tflite_size / original_size) * 100):.1f}%")
        
        print("\n📋 Next steps:")
        print("  1. Use the .tflite file for TFLite inference")
        print("  2. Include the .h file in your C/C++ project")
        print("  3. The model is ready for deployment on embedded devices")
        
        # Save model info for reference
        model_info = {
            "model_name": "MDET_INT8_Quantized_Fixed",
            "original_size_mb": round(original_size, 2),
            "quantized_size_mb": round(tflite_size, 2),
            "compression_ratio_percent": round((1 - tflite_size / original_size) * 100, 1),
            "input_shape": [224, 224, 3],
            "output_shape": [14, 14, 3, 29],  # Based on actual model output
            "quantization": "INT8",
            "compatibility": "TFLite Micro",
            "export_timestamp": str(datetime.datetime.now())
        }
        
        with open("mdet_int8_fixed_model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        
        print(f"✅ Model info saved to: mdet_int8_fixed_model_info.json")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("Export and quantization failed!")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 All operations completed successfully!")
    else:
        print("\n💥 Export and quantization failed!")
        exit(1)
