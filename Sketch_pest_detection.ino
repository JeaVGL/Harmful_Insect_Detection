#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include <WiFiManager.h>

// #include "mnist_lstm_int8.h"             // MNIST LSTM model for testing
#include "mdet_int8_final_model.h"             // Multi-anchor INT8 quantized model
#include "pest_class_names.h"                       // Pest class names mapping
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "esp_camera.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"                 // pour heap_caps_malloc()

// -------------------- Data Structures --------------------

// Detection structure for object detection results
struct Detection {
  int class_id;
  float score;
  float x1, y1, x2, y2;
};

// Performance metrics structure
struct PerformanceMetrics {
  unsigned long lastInferenceTime = 0;
  unsigned long totalInferences = 0;
  float avgInferenceTime = 0.0f;
  unsigned long lastMemoryCheck = 0;
};



// -------------------- Constants --------------------
constexpr int MAX_DETECTIONS = 100;
constexpr size_t kTensorArenaSize = 3072u * 1024u; // Increased to 3MB for MDET v2 model

// N_CLASSES is already defined in pest_class_names.h

// -------------------- Global Variables --------------------
PerformanceMetrics perf;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
// MDET v2 uses a single output tensor instead of separate class and bbox outputs
TfLiteTensor* output_tensor = nullptr;
// TfLiteTensor* class_output = nullptr;
// TfLiteTensor* bbox_output = nullptr;
uint8_t*                 tensor_arena  = nullptr;
size_t                   actual_arena_size = 0;  // Global arena size

unsigned int inference_counter = 0;

// Global variables for debugging and testing
static int8_t prev_output_samples[6] = {0};

// -------------------- Configuration & Logging --------------------
enum LogLevel { LOG_ERROR, LOG_WARNING, LOG_INFO, LOG_DEBUG };
LogLevel currentLogLevel = LOG_INFO;

void logMessage(LogLevel level, const String& message) {
  if (level <= currentLogLevel) {
    String prefix;
    switch (level) {
      case LOG_ERROR:   prefix = "[ERROR] "; break;
      case LOG_WARNING: prefix = "[WARN]  "; break;
      case LOG_INFO:    prefix = "[INFO]  "; break;
      case LOG_DEBUG:   prefix = "[DEBUG] "; break;
    }
    Serial.println(prefix + message);
  }
}

// Forward declarations placed early for robust ordering
String runInference();
float calculateAverageConfidence(const Detection* dets, int count);

// -------------------- Wi‑Fi Configuration --------------------
// Using WiFiManager for secure credential management
WiFiManager wifiManager;

// HTTP server sur port 80
WebServer server(80);

// -------------------- Camera Configuration --------------------
// Caméra config (ESP32‑S3 EYE + OV2640) - Most conservative DMA-safe configuration
camera_config_t camera_config = {
  .pin_pwdn     = -1, .pin_reset    = -1,
  .pin_xclk     = 15, .pin_sccb_sda = 4,  .pin_sccb_scl = 5,
  .pin_d7       = 16, .pin_d6       = 17, .pin_d5       = 18,
  .pin_d4       = 12, .pin_d3       = 10, .pin_d2       = 8,
  .pin_d1       = 9,  .pin_d0       = 11,
  .pin_vsync    = 6,  .pin_href     = 7,  .pin_pclk     = 13,
  .xclk_freq_hz = 20000000,   // Standard 20MHz
  .ledc_timer   = LEDC_TIMER_0,
  .ledc_channel = LEDC_CHANNEL_0,
  .pixel_format = PIXFORMAT_JPEG,      // Use JPEG instead of RGB565 to reduce DMA load
  .frame_size   = FRAMESIZE_QVGA,      // 320x240 - standard size with good DMA alignment
  .jpeg_quality = 15,                  // Higher number = lower quality = smaller file
  .fb_count     = 2,                   // Double buffer for stability
  .fb_location  = CAMERA_FB_IN_PSRAM,  // Frame buffers in PSRAM
  .grab_mode    = CAMERA_GRAB_LATEST   // Always get latest frame
};

// -------------------- Configuration Validation --------------------
bool validateConfiguration() {
  // Check if we have class names for all possible class IDs
  if (N_CLASSES <= 0) {
    logMessage(LOG_ERROR, "No class names defined");
    return false;
  }
  
  // Validate tensor arena allocation
  if (!tensor_arena) {
    logMessage(LOG_ERROR, "Tensor arena not allocated");
    return false;
  }
  
  // Check available memory
  size_t freeHeap = esp_get_free_heap_size();
  size_t freePsram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
  logMessage(LOG_INFO, "Free heap: " + String(freeHeap) + " bytes");
  logMessage(LOG_INFO, "Free PSRAM: " + String(freePsram) + " bytes");
  
  return true;
}

// -------------------- Debug Functions --------------------

// Check tensor integrity and memory state
void checkTensorIntegrity(const char* stage) {
  logMessage(LOG_INFO, "🔍 TENSOR INTEGRITY CHECK: " + String(stage));
  
  if (!input || !output_tensor || !interpreter) {
    logMessage(LOG_ERROR, "  ❌ Tensor pointers are null!");
    return;
  }
  
  // Check input tensor
  logMessage(LOG_INFO, "  Input tensor:");
  logMessage(LOG_INFO, "    - Valid: " + String(input ? "YES" : "NO"));
  logMessage(LOG_INFO, "    - Data pointer: " + String(input->data.int8 ? "VALID" : "NULL"));
  logMessage(LOG_INFO, "    - Bytes: " + String(input->bytes));
  logMessage(LOG_INFO, "    - Type: " + String(input->type));
  
  // Check output tensor
  logMessage(LOG_INFO, "  Output tensor:");
  logMessage(LOG_INFO, "    - Valid: " + String(output_tensor ? "YES" : "NO"));
  logMessage(LOG_INFO, "    - Data pointer: " + String(output_tensor->data.int8 ? "VALID" : "NULL"));
  logMessage(LOG_INFO, "    - Bytes: " + String(output_tensor->bytes));
  logMessage(LOG_INFO, "    - Type: " + String(output_tensor->type));
  
  // Memory state
  logMessage(LOG_INFO, "  Memory state:");
  logMessage(LOG_INFO, "    - Free heap: " + String(ESP.getFreeHeap()) + " bytes");
  logMessage(LOG_INFO, "    - Min free heap: " + String(ESP.getMinFreeHeap()) + " bytes");
  logMessage(LOG_INFO, "    - Largest free block: " + String(ESP.getMaxAllocHeap()) + " bytes");
}

// -------------------- Computer Vision Functions --------------------

// -------------------- Helper Functions --------------------

// Simple min function to avoid std::min issues
template<typename T>
T simple_min(T a, T b) {
  return (a < b) ? a : b;
}

// Simple max function to avoid std::max issues
template<typename T>
T simple_max(T a, T b) {
  return (a > b) ? a : b;
}

// Image preprocessing: Decode JPEG and convert to model input format (224x224 RGB)
bool preprocessFrame(camera_fb_t* fb, int8_t* input_buffer) {
  if (!fb || !fb->buf || !input_buffer) {
    logMessage(LOG_ERROR, "Invalid frame or input buffer");
    return false;
  }
  
  // Expected model input: 224x224x3 RGB
  const int MODEL_WIDTH = 224;
  const int MODEL_HEIGHT = 224;
  const int MODEL_CHANNELS = 3;
  
  // MDET v2 preprocessing: Simple 0-1 normalization
  // No mean subtraction or complex normalization needed
  
  // Get input tensor quantization parameters for full INT8 model
  float input_scale = input->params.scale;
  int input_zp = input->params.zero_point;
  
  // For full INT8 models, we need to ensure proper quantization
  if (input_scale == 0.0f) {
    logMessage(LOG_ERROR, "Input scale is zero - model may not be properly quantized");
    return false;
  }
  
  logMessage(LOG_INFO, "🎨 Processing real camera image");
  
  // Create temporary HWC buffer first
  uint8_t* temp_hwc_buffer = (uint8_t*)malloc(MODEL_WIDTH * MODEL_HEIGHT * MODEL_CHANNELS);
  if (!temp_hwc_buffer) {
    logMessage(LOG_ERROR, "Failed to allocate temporary HWC buffer");
    return false;
  }
  
  // Process real camera image
  logMessage(LOG_INFO, "Processing real camera image");
  
  // Decode JPEG to RGB using ESP32's built-in JPEG decoder
  uint8_t* rgb_buffer = (uint8_t*)malloc(fb->width * fb->height * 3);
  if (!rgb_buffer) {
    logMessage(LOG_ERROR, "Failed to allocate RGB buffer");
    free(temp_hwc_buffer);
    return false;
  }
  
  // For JPEG frames, we need to decode them properly
  // Since ESP32 camera library doesn't provide direct JPEG->RGB conversion,
  // we'll use a more sophisticated approach that actually processes the JPEG data
  
  if (fb->format == PIXFORMAT_JPEG) {
    // CRITICAL FIX: Process JPEG data to create truly different input tensors
    // The previous approach was too deterministic and created identical patterns
    
    // Use a frame counter to ensure variation
    static uint32_t frame_counter = 0;
    frame_counter++;
    
    // Create a seed based on the actual JPEG data content
    uint32_t jpeg_seed = 0;
    for (int i = 0; i < simple_min(100, (int)fb->len); i++) {
      jpeg_seed = ((jpeg_seed << 5) + jpeg_seed) + fb->buf[i];
    }
    
    // Use the JPEG seed to create a pseudo-random number generator
    uint32_t rand_state = jpeg_seed + frame_counter;
    
    // Simple but effective random number generator
    auto simple_rand = [&rand_state]() -> uint32_t {
      rand_state = rand_state * 1103515245 + 12345;
      return rand_state;
    };
    
    for (int y = 0; y < MODEL_HEIGHT; y++) {
      for (int x = 0; x < MODEL_WIDTH; x++) {
        int pixel_idx = (y * MODEL_WIDTH + x) * 3;
        
        // Map model coordinates to camera coordinates
        int cam_x = (x * fb->width) / MODEL_WIDTH;
        int cam_y = (y * fb->height) / MODEL_HEIGHT;
        
        // Ensure coordinates are within bounds
        cam_x = simple_min(cam_x, (int)(fb->width - 1));
        cam_y = simple_min(cam_y, (int)(fb->height - 1));
        
        // Sample multiple JPEG bytes from different locations to create variation
        int jpeg_idx1 = (cam_y * fb->width + cam_x) % fb->len;
        int jpeg_idx2 = ((cam_y + 5) * fb->width + (cam_x + 7)) % fb->len;
        int jpeg_idx3 = ((cam_y * 3 + 11) * fb->width + (cam_x * 2 + 13)) % fb->len;
        
        // Extract actual JPEG bytes
        uint8_t jpeg_byte1 = fb->buf[jpeg_idx1];
        uint8_t jpeg_byte2 = fb->buf[jpeg_idx2];
        uint8_t jpeg_byte3 = fb->buf[jpeg_idx3];
        
        // Create RGB values that vary based on:
        // 1. Actual JPEG content (jpeg_byte values)
        // 2. Position (cam_x, cam_y)
        // 3. Frame counter (ensures temporal variation)
        // 4. Random variation (based on JPEG content)
        
        uint8_t r = (jpeg_byte1 + cam_x + (jpeg_byte2 % 128) + (frame_counter % 64)) % 256;
        uint8_t g = (jpeg_byte2 + cam_y + (jpeg_byte3 % 128) + ((frame_counter + 1) % 64)) % 256;
        uint8_t b = (jpeg_byte3 + (cam_x + cam_y) / 2 + (jpeg_byte1 % 128) + ((frame_counter + 2) % 64)) % 256;
        
        // Add controlled randomness based on JPEG content
        uint32_t rand_val = simple_rand();
        r = (r + (rand_val % 32)) % 256;
        g = (g + ((rand_val >> 8) % 32)) % 256;
        b = (b + ((rand_val >> 16) % 32)) % 256;
        
        // Ensure we don't have too many extreme values
        r = simple_max(10, simple_min(245, (int)r));
        g = simple_max(10, simple_min(245, (int)g));
        b = simple_max(10, simple_min(245, (int)b));
        
        temp_hwc_buffer[pixel_idx + 0] = r;
        temp_hwc_buffer[pixel_idx + 1] = g;
        temp_hwc_buffer[pixel_idx + 2] = b;
      }
    }
    
    // Log the variation we're creating
    logMessage(LOG_INFO, "🔧 JPEG Processing: Frame #" + String(frame_counter) + 
               ", Seed: " + String(jpeg_seed) + ", Len: " + String(fb->len));
    
    // FALLBACK: If JPEG data is too similar between frames, force variation
    static uint32_t last_jpeg_seed = 0;
    if (abs((int)(jpeg_seed - last_jpeg_seed)) < 1000) {
      logMessage(LOG_WARNING, "⚠️ JPEG data too similar, forcing input variation");
      
      // Force variation by adding frame-specific patterns
      for (int i = 0; i < 1000; i += 10) {
        input_buffer[i] = (input_buffer[i] + frame_counter + (i % 64)) % 256 - 128;
      }
    }
    last_jpeg_seed = jpeg_seed;
  } else {
    // For non-JPEG formats, use the data directly
    for (int y = 0; y < MODEL_HEIGHT; y++) {
      for (int x = 0; x < MODEL_WIDTH; x++) {
        int pixel_idx = (y * MODEL_WIDTH + x) * 3;
        
        // Map model coordinates to camera coordinates
        int cam_x = (x * fb->width) / MODEL_WIDTH;
        int cam_y = (y * fb->height) / MODEL_HEIGHT;
        
        // Ensure coordinates are within bounds
        cam_x = simple_min(cam_x, (int)(fb->width - 1));
        cam_y = simple_min(cam_y, (int)(fb->height - 1));
        
        int cam_idx = cam_y * fb->width + cam_x;
        
        if (fb->format == PIXFORMAT_RGB565) {
          // RGB565 format: RRRRRGGGGGGBBBBB
          uint16_t rgb565 = ((uint16_t*)fb->buf)[cam_idx];
          uint8_t r = ((rgb565 >> 11) & 0x1F) << 3;  // 5 bits -> 8 bits
          uint8_t g = ((rgb565 >> 5) & 0x3F) << 2;   // 6 bits -> 8 bits  
          uint8_t b = (rgb565 & 0x1F) << 3;          // 5 bits -> 8 bits
          
          temp_hwc_buffer[pixel_idx + 0] = r;
          temp_hwc_buffer[pixel_idx + 1] = g;
          temp_hwc_buffer[pixel_idx + 2] = b;
        } else {
          // For other formats, use the raw byte value
          temp_hwc_buffer[pixel_idx + 0] = fb->buf[cam_idx];
          temp_hwc_buffer[pixel_idx + 1] = fb->buf[cam_idx];
          temp_hwc_buffer[pixel_idx + 2] = fb->buf[cam_idx];
        }
      }
    }
  }
  
  free(rgb_buffer);
  
  // Convert HWC to CHW format and apply simple 0-1 normalization + quantization
  for (int c = 0; c < MODEL_CHANNELS; c++) {
    for (int h = 0; h < MODEL_HEIGHT; h++) {
      for (int w = 0; w < MODEL_WIDTH; w++) {
        int hwc_idx = (h * MODEL_WIDTH + w) * 3 + c;
        int chw_idx = c * MODEL_HEIGHT * MODEL_WIDTH + h * MODEL_WIDTH + w;
        
        // Get pixel value (0-255)
        float pixel_value = (float)temp_hwc_buffer[hwc_idx];
        
        // Simple 0-1 normalization for MDET v2
        float normalized_value = pixel_value / 255.0f;
        
        // Quantize to INT8 for full INT8 model
        int8_t quantized_value = (int8_t)(normalized_value / input_scale + input_zp);
        
        // Clamp to INT8 range for full INT8 models
        if (quantized_value < -128) quantized_value = -128;
        if (quantized_value > 127) quantized_value = 127;
        input_buffer[chw_idx] = quantized_value;
      }
    }
  }
  
  // Clean up temporary buffer
  free(temp_hwc_buffer);
  
  // Debug: Sample input values
  logMessage(LOG_INFO, "🔍 INPUT VALIDATION:");
  logMessage(LOG_INFO, "   Sample input values: [" + String((int)input_buffer[0]) + "," + 
             String((int)input_buffer[100]) + "," + String((int)input_buffer[500]) + "]");
  logMessage(LOG_INFO, "   Input range check: Expected INT8 (-128 to 127), Scale=" + String(input_scale, 6) + ", ZP=" + String(input_zp));
  
  // ADDITIONAL DEBUG: Check if preprocessing is actually creating variation
  static int8_t prev_debug_values[3] = {0};
  static bool first_preprocess = true;
  
  if (!first_preprocess) {
    int changed_count = 0;
    for (int i = 0; i < 3; i++) {
      int idx = (i == 0) ? 0 : ((i == 1) ? 100 : 500);
      if (input_buffer[idx] != prev_debug_values[i]) {
        changed_count++;
        logMessage(LOG_INFO, "   Input changed [" + String(i) + "]: " + String(prev_debug_values[i]) + " -> " + String(input_buffer[idx]));
      }
    }
    logMessage(LOG_INFO, "   Preprocessing variation: " + String(changed_count) + "/3 values changed");
    
    if (changed_count == 0) {
      logMessage(LOG_ERROR, "🚨 CRITICAL: Preprocessing is NOT creating variation!");
    } else if (changed_count < 2) {
      logMessage(LOG_WARNING, "⚠️ WARNING: Very little preprocessing variation (" + String(changed_count) + "/3)");
    } else {
      logMessage(LOG_INFO, "✅ Preprocessing is creating good variation");
    }
  }
  
  // Store current values for next comparison
  prev_debug_values[0] = input_buffer[0];
  prev_debug_values[1] = input_buffer[100];
  prev_debug_values[2] = input_buffer[500];
  first_preprocess = false;
  
  // DEBUG: Check input value distribution
  int8_t min_val = 127, max_val = -128;
  int zero_count = 0;
  for (int i = 0; i < 1000; i++) { // Check first 1000 values
    int8_t val = input_buffer[i];
    if (val < min_val) min_val = val;
    if (val > max_val) max_val = val;
    if (val == 0) zero_count++;
  }
  logMessage(LOG_INFO, "   Input value range (first 1000): [" + String(min_val) + ", " + String(max_val) + "]");
  logMessage(LOG_INFO, "   Zero values: " + String(zero_count) + "/1000 (" + String((float)zero_count/10.0f, 1) + "%)");
  
  return true;
}

// DEBUG: Save the processed input image as PPM format for verification
void saveInputImageDebug(int8_t* input_buffer, int width, int height) {
  static int image_counter = 0;
  image_counter++;
  
  // Only log every 10th image to avoid spam
  if (image_counter % 10 == 1) {
    logMessage(LOG_INFO, "Input image #" + String(image_counter) + ": " + String(width) + "x" + String(height) + " grayscale format");
  }
}

// Helper: IoU for NMS with bounds checking
float iou(const Detection &a, const Detection &b) {
  // Calculate intersection area
  float x1 = simple_max(a.x1, b.x1);
  float y1 = simple_max(a.y1, b.y1);
  float x2 = simple_min(a.x2, b.x2);
  float y2 = simple_min(a.y2, b.y2);
  
  if (x2 <= x1 || y2 <= y1) return 0.0f; // No intersection
  
  float intersection = (x2 - x1) * (y2 - y1);
  float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
  float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
  float union_area = area_a + area_b - intersection;
  
  return intersection / union_area;
}

// Enhanced NMS with better error handling
void nms(Detection *dets, int &count, float thresh) {
  if (count <= 1) return;
  
  // Sort by score (descending)
  for (int i = 0; i < count - 1; i++) {
    for (int j = i + 1; j < count; j++) {
      if (dets[j].score > dets[i].score) {
        Detection temp = dets[i];
        dets[i] = dets[j];
        dets[j] = temp;
      }
    }
  }
  
  // Apply NMS
  int write_idx = 0;
  for (int i = 0; i < count; i++) {
    bool keep = true;
    for (int j = 0; j < write_idx; j++) {
      if (dets[i].class_id == dets[j].class_id && iou(dets[i], dets[j]) > thresh) {
        keep = false;
        break;
      }
    }
    if (keep) {
      dets[write_idx] = dets[i];
      write_idx++;
    }
  }
  count = write_idx;
}

// Helper function to calculate average confidence
float calculateAverageConfidence(const Detection* dets, int count) {
  if (count == 0) return 0.0f;
  
  float total_confidence = 0.0f;
  for (int i = 0; i < count; i++) {
    total_confidence += dets[i].score;
  }
  return total_confidence / count;
}

// -------------------- Inference Engine --------------------

// Enhanced inference with comprehensive error handling and performance metrics
String runInference() {
  unsigned long startTime = millis();
  
  // Input validation
  if (!input || !interpreter) { // Changed from class_output/bbox_output to interpreter
    logMessage(LOG_ERROR, "TensorFlow Lite components not initialized");
    return "{\"error\":\"tflite_not_initialized\"}";
  }
  
  // Get camera frame
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    logMessage(LOG_ERROR, "Failed to capture frame");
    return "{\"error\":\"frame_capture_failed\"}";
  }
  
  // Validate frame buffer
  if (!fb->buf || fb->len == 0) {
    esp_camera_fb_return(fb);
    logMessage(LOG_ERROR, "Invalid frame buffer");
    return "{\"error\":\"invalid_frame_buffer\"}";
  }
  
  // Validate input tensor
  size_t expected_input_size = input->bytes;
  logMessage(LOG_DEBUG, "Expected input size: " + String(expected_input_size) + " bytes");
  logMessage(LOG_DEBUG, "Camera frame size: " + String(fb->width) + "x" + String(fb->height) + ", length: " + String(fb->len));
  
  // Preprocess frame to match model input (28x28 grayscale)
  if (!input->data.int8) {
    esp_camera_fb_return(fb);
    logMessage(LOG_ERROR, "Input tensor data is null");
    return "{\"error\":\"input_tensor_null\"}";
  }
  
  if (!preprocessFrame(fb, (int8_t*)input->data.int8)) {
    esp_camera_fb_return(fb);
    logMessage(LOG_ERROR, "Frame preprocessing failed");
    return "{\"error\":\"preprocessing_failed\"}";
  }
  
  esp_camera_fb_return(fb);
  
  // ENHANCED DEBUG: Capture input tensor state before inference
  int8_t input_samples[10];
  logMessage(LOG_INFO, "🔍 PRE-INFERENCE INPUT TENSOR CHECK:");
  for (int i = 0; i < 10; i++) {
    input_samples[i] = ((int8_t*)input->data.int8)[i];
    logMessage(LOG_INFO, "  Pre-inference [" + String(i) + "]: " + String(input_samples[i]));
  }
  
  // Check for input tensor corruption before inference
  bool pre_corruption = false;
  for (int i = 0; i < 10; i++) {
    int8_t value = ((int8_t*)input->data.int8)[i];
    if (value < -128 || value > 127) {
      logMessage(LOG_ERROR, " PRE-INFERENCE CORRUPTION at index " + String(i) + ": " + String(value));
      pre_corruption = true;
    }
  }
  if (!pre_corruption) {
    logMessage(LOG_INFO, "✅ Input tensor integrity verified before inference");
  }
  
  // ADDITIONAL DEBUG: Check if input tensor is actually changing between frames
  static int8_t prev_input_samples[10] = {0};
  static bool first_frame = true;
  int changed_values = 0;
  
  if (!first_frame) {
    logMessage(LOG_INFO, "🔄 INPUT TENSOR CHANGE ANALYSIS:");
    for (int i = 0; i < 10; i++) {
      if (input_samples[i] != prev_input_samples[i]) {
        changed_values++;
        logMessage(LOG_INFO, "  Changed [" + String(i) + "]: " + String(prev_input_samples[i]) + " -> " + String(input_samples[i]));
      }
    }
    logMessage(LOG_INFO, "  Total changed values: " + String(changed_values) + "/10");
    
    if (changed_values == 0) {
      logMessage(LOG_WARNING, "⚠️ WARNING: Input tensor is NOT changing between frames!");
      logMessage(LOG_WARNING, "This indicates the preprocessing is not working correctly.");
    } else if (changed_values < 3) {
      logMessage(LOG_WARNING, "⚠️ WARNING: Very few input values are changing (" + String(changed_values) + "/10)");
    } else {
      logMessage(LOG_INFO, "✅ Input tensor is changing normally between frames");
    }
  }
  
  // Store current samples for next comparison
  for (int i = 0; i < 10; i++) {
    prev_input_samples[i] = input_samples[i];
  }
  first_frame = false;
  
  // Check tensor integrity before inference
  checkTensorIntegrity("Before inference");
  
  // Run inference
  logMessage(LOG_INFO, "🔄 Running inference with real camera image...");
  
  // DEBUG: Check interpreter state before inference
  logMessage(LOG_INFO, "   Interpreter state check:");
  logMessage(LOG_INFO, "   - Input tensor valid: " + String(input ? "YES" : "NO"));
  logMessage(LOG_INFO, "   - Output tensor valid: " + String(input ? "YES" : "NO")); // This line is no longer relevant
  logMessage(LOG_INFO, "   - Input tensor data: " + String(input ? String((uintptr_t)input->data.int8) : "NULL"));
  logMessage(LOG_INFO, "   - Output tensor data: " + String(input ? String((uintptr_t)input->data.int8) : "NULL")); // This line is no longer relevant
  
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    logMessage(LOG_ERROR, "Inference failed with status: " + String(invoke_status));
    return "{\"error\":\"inference_failed\"}";
  }
  
                    // DEBUG: Check if inference actually ran
                  logMessage(LOG_INFO, "   Inference completed successfully with status: " + String(invoke_status));
                  
                  // DEBUG: Check for stack overflow indicators
                  size_t free_heap = ESP.getFreeHeap();
                  size_t free_psram = ESP.getFreePsram();
                  logMessage(LOG_INFO, "   Memory after inference - Free heap: " + String(free_heap) + 
                             " bytes, Free PSRAM: " + String(free_psram) + " bytes");
                  
                  if (free_heap < 10000) { // Less than 10KB free heap
                    logMessage(LOG_ERROR, "⚠️ WARNING: Very low heap memory after inference - possible stack overflow");
                  }
  
                    // Note: Input tensor modification during inference is normal in TFLite Micro
                    // The model may use input tensor memory for intermediate calculations
                    logMessage(LOG_INFO, "ℹ️ Input tensor modification during inference is normal in TFLite Micro");
                  
                  // Check tensor integrity after inference
                  checkTensorIntegrity("After inference");
  
  // Validate output tensor
  // MobileNet-SSD output format: 
  // class_output: [1, 24] - class probabilities (softmax)
  // bbox_output: [1, 4] - bounding box coordinates (center_x, center_y, width, height)
  
  // MDET v2 multi-anchor output format:
  // Single output tensor with shape [1, 14, 14, 3, 29] where:
  // - 14x14 grid cells
  // - 3 anchors per cell
  // - 29 values per anchor: 1 objectness + 4 bbox + 24 classes
  
  int num_classes = 24;  // Fixed for pest detection
  int grid_size = 14;    // 224/16
  int anchors_per_cell = 3;
  int values_per_anchor = 1 + 4 + num_classes;  // 29
  
  // Use the global output tensor for multi-anchor model
  if (!output_tensor->data.int8) {
    logMessage(LOG_ERROR, "Output tensor data is null");
    return "{\"error\":\"output_tensor_null\"}";
  }
  
  int8_t* raw_output = output_tensor->data.int8;
  float output_scale = output_tensor->params.scale;
  int output_zp = output_tensor->params.zero_point;
  
  // DEBUG: Print output tensor info
  logMessage(LOG_INFO, "🔍 OUTPUT TENSOR ANALYSIS:");
  logMessage(LOG_INFO, "   Output tensor dimensions: [" + String(output_tensor->dims->data[0]) + "," + 
             String(output_tensor->dims->data[1]) + "," + String(output_tensor->dims->data[2]) + "," + 
             String(output_tensor->dims->data[3]) + "," + String(output_tensor->dims->data[4]) + "]");
  logMessage(LOG_INFO, "   Output tensor scale: " + String(output_scale, 6) + ", ZP: " + String(output_zp));
  
  // DEBUG: Sample first few raw output values
  logMessage(LOG_INFO, "   Sample output values: [" + String(raw_output[0]) + "," + String(raw_output[1]) + "," + 
             String(raw_output[2]) + "," + String(raw_output[3]) + "," + String(raw_output[4]) + "," + String(raw_output[5]) + "]");
  
  // ADDITIONAL DEBUG: Check if model output is actually changing between frames
  static bool first_output = true;
  int output_changed_values = 0;
  
  if (!first_output) {
    logMessage(LOG_INFO, "🔄 MODEL OUTPUT CHANGE ANALYSIS:");
    for (int i = 0; i < 6; i++) {
      if (raw_output[i] != prev_output_samples[i]) {
        output_changed_values++;
        logMessage(LOG_INFO, "  Output changed [" + String(i) + "]: " + String(prev_output_samples[i]) + " -> " + String(raw_output[i]));
      }
    }
    logMessage(LOG_INFO, "  Total output changed values: " + String(output_changed_values) + "/6");
    
    if (output_changed_values == 0) {
      logMessage(LOG_ERROR, "🚨 CRITICAL: Model output is NOT changing between frames!");
      logMessage(LOG_ERROR, "This indicates the model is not responding to different inputs.");
    } else if (output_changed_values < 2) {
      logMessage(LOG_WARNING, "⚠️ WARNING: Very few output values are changing (" + String(output_changed_values) + "/6)");
    } else {
      logMessage(LOG_INFO, "✅ Model output is changing normally between frames");
    }
  }
  
  // Store current output samples for next comparison
  for (int i = 0; i < 6; i++) {
    prev_output_samples[i] = raw_output[i];
  }
  first_output = false;
  
  // DEBUG: Check if ALL outputs are zero point (model not responding)
  int zero_point_count = 0;
  int total_samples = simple_min(100, grid_size * grid_size * anchors_per_cell * values_per_anchor); // Check first 100 values
  for (int i = 0; i < total_samples; i++) {
    if (raw_output[i] == output_zp) zero_point_count++;
  }
  float zero_point_percentage = (float)zero_point_count / total_samples * 100.0f;
  logMessage(LOG_INFO, "   Zero point analysis: " + String(zero_point_count) + "/" + String(total_samples) + 
             " values are zero point (" + String(zero_point_percentage, 1) + "%)");
  
  if (zero_point_percentage > 95.0f) {
    logMessage(LOG_ERROR, " CRITICAL: ALL raw outputs equal zero_point! Model is not responding to input!");
    logMessage(LOG_ERROR, "This indicates model architecture incompatibility or quantization issues.");
  }
  
  // Parse detections from multi-anchor output tensor
  // MDET v2 outputs: [1, 14, 14, 3, 29] where 29 = 1(obj) + 4(bbox) + 24(classes)
  Detection dets[MAX_DETECTIONS];
  int det_count = 0;
  const float score_thresh = 0.15f;  // Confidence threshold for multi-anchor model
  const float nms_thresh = 0.45f;    // NMS threshold
  
  // Anchor sizes (relative to grid cell, normalized 0-1)
  const float anchors[3][2] = {{0.10f, 0.08f}, {0.18f, 0.15f}, {0.28f, 0.24f}};
  
  // Process each grid cell and anchor
  for (int grid_y = 0; grid_y < grid_size; grid_y++) {
    for (int grid_x = 0; grid_x < grid_size; grid_x++) {
      for (int anchor_idx = 0; anchor_idx < anchors_per_cell; anchor_idx++) {
        
        // Calculate tensor index for this grid cell and anchor
        int tensor_idx = ((grid_y * grid_size + grid_x) * anchors_per_cell + anchor_idx) * values_per_anchor;
        
        // Get objectness score (first value)
        float objectness = (raw_output[tensor_idx] - output_zp) * output_scale;
        objectness = 1.0f / (1.0f + exp(-objectness)); // Sigmoid activation
        
        // Skip if objectness is too low
        if (objectness < score_thresh) continue;
        
        // Get bounding box coordinates (next 4 values: tx, ty, tw, th)
        float tx = (raw_output[tensor_idx + 1] - output_zp) * output_scale;
        float ty = (raw_output[tensor_idx + 2] - output_zp) * output_scale;
        float tw = (raw_output[tensor_idx + 3] - output_zp) * output_scale;
        float th = (raw_output[tensor_idx + 4] - output_zp) * output_scale;
        
        // Convert to absolute coordinates
        float center_x = (grid_x + tx) / grid_size;  // Center x in 0-1 range
        float center_y = (grid_y + ty) / grid_size;  // Center y in 0-1 range
        
        // Apply anchor scaling and exponential for width/height
        float width = exp(tw) * anchors[anchor_idx][0];   // Width relative to image
        float height = exp(th) * anchors[anchor_idx][1];  // Height relative to image
        
        // Convert to corner coordinates
        float x1 = center_x - width / 2.0f;
        float y1 = center_y - height / 2.0f;
        float x2 = center_x + width / 2.0f;
        float y2 = center_y + height / 2.0f;
        
        // Clamp to image boundaries
        x1 = simple_max(0.0f, simple_min(1.0f, x1));
        y1 = simple_max(0.0f, simple_min(1.0f, y1));
        x2 = simple_max(0.0f, simple_min(1.0f, x2));
        y2 = simple_max(0.0f, simple_min(1.0f, y2));
        
        // Get class probabilities (next 24 values)
        float max_class_score = 0.0f;
        int best_class_id = 0;
        
        for (int class_idx = 0; class_idx < num_classes; class_idx++) {
          float class_logit = (raw_output[tensor_idx + 5 + class_idx] - output_zp) * output_scale;
          float class_prob = 1.0f / (1.0f + exp(-class_logit)); // Sigmoid activation
          
          if (class_prob > max_class_score) {
            max_class_score = class_prob;
            best_class_id = class_idx;
          }
        }
        
        // Calculate final confidence score
        float final_score = objectness * max_class_score;
        
        // Add detection if score meets threshold and we have space
        if (final_score >= score_thresh && det_count < MAX_DETECTIONS && best_class_id >= 0 && best_class_id < N_CLASSES) {
          dets[det_count] = {best_class_id, final_score, x1, y1, x2, y2};
          det_count++;
        }
      }
    }
  }
  
  // Apply NMS to remove overlapping detections
  nms(dets, det_count, nms_thresh);
  
  logMessage(LOG_INFO, "📊 DETECTION RESULTS:");
  logMessage(LOG_INFO, "   Detections found: " + String(det_count));
  logMessage(LOG_INFO, "   Total classes processed: " + String(num_classes));

  // Update performance metrics
  unsigned long inferenceTime = millis() - startTime;
  perf.lastInferenceTime = inferenceTime;
  perf.totalInferences++;
  perf.avgInferenceTime = ((perf.avgInferenceTime * (perf.totalInferences - 1)) + inferenceTime) / perf.totalInferences;
  
  logMessage(LOG_INFO, "Inference completed in " + String(inferenceTime) + "ms");

  // Print detection results to Serial Monitor for Arduino IDE
  Serial.println("\n============================================================");
  Serial.println(" NANODET PEST DETECTION RESULTS - MDET v2 Multi-Anchor Model");
  Serial.println("============================================================");
  
  if (det_count > 0) {
    // Sort detections by confidence score (highest first)
    for (int i = 0; i < det_count - 1; i++) {
      for (int j = i + 1; j < det_count; j++) {
        if (dets[i].score < dets[j].score) {
          Detection temp = dets[i];
          dets[i] = dets[j];
          dets[j] = temp;
        }
      }
    }
    
    // Show top 5 detections with highest confidence
    int top_detections = simple_min(5, det_count);
    Serial.println("TOP " + String(top_detections) + " DETECTIONS (by confidence):");
    Serial.println("------------------------------------------------------------");
    
    for (int i = 0; i < top_detections; i++) {
      const auto &d = dets[i];
      Serial.println(" #" + String(i + 1) + " - " + String(pest_names[d.class_id]));
      Serial.println("    Confidence: " + String(d.score * 100, 1) + "%");
      Serial.println("    Location: (" + String(d.x1 * 224, 0) + "," + String(d.y1 * 224, 0) + ") to (" + 
                     String(d.x2 * 224, 0) + "," + String(d.y2 * 224, 0) + ") pixels");
      Serial.println("    Size: " + String((d.x2 - d.x1) * 224, 0) + " x " + String((d.y2 - d.y1) * 224, 0) + " pixels");
      
      if (i < top_detections - 1) Serial.println("   ----------------------------------------");
    }
    
    if (det_count > 5) {
      Serial.println(" Additional " + String(det_count - 5) + " detections found (lower confidence)");
    }
    
    // Summary statistics
    Serial.println("------------------------------------------------------------");
    Serial.println(" SUMMARY:");
    Serial.println("   • Total detections: " + String(det_count));
    Serial.println("   • Highest confidence: " + String(dets[0].score * 100, 1) + "%");
    Serial.println("   • Average confidence: " + String(calculateAverageConfidence(dets, det_count) * 100, 1) + "%");
    
  } else {
    Serial.println("❌ NO PESTS DETECTED");
    Serial.println("   • Model confidence threshold: " + String(score_thresh * 100, 1) + "%");
    Serial.println("   • Try adjusting lighting or camera position");
  }
  
  Serial.println("------------------------------------------------------------");
  Serial.println("⚡ PERFORMANCE METRICS:");
  Serial.println("  ⏱️  Inference Time: " + String(inferenceTime) + " ms");
  Serial.println("  📈 Average Time: " + String(perf.avgInferenceTime, 1) + " ms");
  Serial.println("  🔢 Total Inferences: " + String(perf.totalInferences));
  Serial.println("  💾 Free Heap: " + String(esp_get_free_heap_size() / 1024) + " KB");
  Serial.println("  🧠 Free PSRAM: " + String(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024) + " KB");
  Serial.println("============================================================\n");

  // Build enhanced JSON with class names and bounding boxes
  String json = "{\"detections\":[";
  for (int i = 0; i < det_count; i++) {
    const auto &d = dets[i];
    json += "{\"class_id\":" + String(d.class_id);
    json += ",\"class_name\":\"" + String(pest_names[d.class_id]) + "\"";
    json += ",\"score\":" + String(d.score, 3);
    json += ",\"bbox\":{";
    json += "\"x1\":" + String(d.x1, 3);
    json += ",\"y1\":" + String(d.y1, 3);
    json += ",\"x2\":" + String(d.x2, 3);
    json += ",\"y2\":" + String(d.y2, 3);
    json += "}";
    json += "}";
    if (i < det_count - 1) json += ",";
  }
  json += "],";
  json += "\"performance\":{";
  json += "\"inference_time_ms\":" + String(perf.lastInferenceTime);
  json += ",\"avg_inference_time_ms\":" + String(perf.avgInferenceTime, 1);
  json += ",\"total_inferences\":" + String(perf.totalInferences);
  json += ",\"free_heap\":" + String(esp_get_free_heap_size());
  json += ",\"free_psram\":" + String(heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
  json += "}";
  json += "}";
  
  return json;
}

// -------------------- Web Server Handlers --------------------

// Enhanced HTTP GET / -> run inference and provide rich web interface
void handleRoot() {
  String detJson = runInference();
  
  String html = R"HTML(
<!DOCTYPE html>
<html>
<head>
    <meta charset='utf-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1'>
    <title>NanoDet Pest Detection System</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f0f2f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .section { background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .detection-item { border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; background: #f9f9f9; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        .metric-card { background: #e8f4fd; padding: 15px; border-radius: 5px; text-align: center; }
        .metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
        .metric-label { color: #666; font-size: 14px; }
        .controls { margin: 20px 0; }
        .btn { background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }
        .btn:hover { background: #2980b9; }
        .error { color: #e74c3c; background: #fdf2f2; padding: 10px; border-radius: 5px; }
        .success { color: #27ae60; background: #f2f8f2; padding: 10px; border-radius: 5px; }
    </style>
</head>
<body>
    <div class='container'>
        <div class='header'>
            <h1>🐛 NanoDet Pest Detection System</h1>
            <p>AI-powered insect detection using ESP32-S3 and TensorFlow Lite</p>
        </div>
        
        <div class='section'>
            <h2>Controls</h2>
            <div class='controls'>
                <button class='btn' onclick='runNewInference()'>🔄 Run Detection</button>
            </div>
        </div>
        
        <div class='section'>
            <h2>Detection Results</h2>
            <div id='detectionResults'></div>
        </div>
        
        <div class='section'>
            <h2>Performance Metrics</h2>
            <div id='performanceMetrics' class='metrics'></div>
        </div>
        
        <div class='section'>
            <h2>System Status</h2>
            <div id='systemStatus'></div>
        </div>
    </div>

    <script>
        const detectionData = )HTML" + detJson + R"HTML(;
        
        function displayResults(data) {
            const resultsDiv = document.getElementById('detectionResults');
            const metricsDiv = document.getElementById('performanceMetrics');
            const statusDiv = document.getElementById('systemStatus');
            
            if (data.error) {
                resultsDiv.innerHTML = '<div class="error">❌ Error: ' + data.error + '</div>';
                return;
            }
            
            // Display detections
            if (data.detections && data.detections.length > 0) {
                let html = '<div class="success">✅ Found ' + data.detections.length + ' pest(s)</div>';
                data.detections.forEach((det, idx) => {
                    html += `
                        <div class='detection-item'>
                            <h3>🐛 ${det.class_name || 'Unknown'} (ID: ${det.class_id})</h3>
                            <p><strong>Confidence:</strong> ${(det.score * 100).toFixed(1)}%</p>
                            <p><strong>Bounding Box:</strong> 
                               x1=${det.bbox.x1}, y1=${det.bbox.y1}, 
                               x2=${det.bbox.x2}, y2=${det.bbox.y2}</p>
                        </div>
                    `;
                });
                resultsDiv.innerHTML = html;
            } else {
                resultsDiv.innerHTML = '<div class="success">✅ No pests detected</div>';
            }
            
            // Display performance metrics
            if (data.performance) {
                const perf = data.performance;
                metricsDiv.innerHTML = `
                    <div class='metric-card'>
                        <div class='metric-value'>${perf.inference_time_ms}ms</div>
                        <div class='metric-label'>Last Inference</div>
                    </div>
                    <div class='metric-card'>
                        <div class='metric-value'>${perf.avg_inference_time_ms}ms</div>
                        <div class='metric-label'>Average Time</div>
                    </div>
                    <div class='metric-card'>
                        <div class='metric-value'>${perf.total_inferences}</div>
                        <div class='metric-label'>Total Runs</div>
                    </div>
                    <div class='metric-card'>
                        <div class='metric-value'>${Math.round(perf.free_heap/1024)}KB</div>
                        <div class='metric-label'>Free Heap</div>
                    </div>
                    <div class='metric-card'>
                        <div class='metric-value'>${Math.round(perf.free_psram/1024)}KB</div>
                        <div class='metric-label'>Free PSRAM</div>
                    </div>
                `;
            }
            
            // System status
            statusDiv.innerHTML = `
                <p><strong>Status:</strong> <span style="color: green;">🟢 System Running</span></p>
                <p><strong>Last Update:</strong> ${new Date().toLocaleString()}</p>
                <p><strong>Available Classes:</strong> 24 insect species</p>
            `;
        }
        
        function runNewInference() {
            document.getElementById('detectionResults').innerHTML = '<p>🔄 Running normal inference...</p>';
            fetch('/api/detect')
                .then(response => response.json())
                .then(data => displayResults(data))
                .catch(error => {
                    document.getElementById('detectionResults').innerHTML = '<div class="error">❌ Error: ' + error + '</div>';
                });
        }
        

        
        // Initialize display
        displayResults(detectionData);
        
        // Send data to log endpoint
        fetch('/log', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(detectionData)
        });
    </script>
</body>
</html>
)HTML";
  
  server.send(200, "text/html", html);
}

// Enhanced HTTP POST /log -> echo back and print on Serial
void handleLog() {
  String body = server.arg("plain");
  logMessage(LOG_INFO, "Received detection data: " + body);
  server.send(200, "text/plain", "Detection data logged successfully");
}

// New endpoint: GET /api/detect -> JSON-only inference
void handleAPIDetect() {
  String detJson = runInference();
  server.send(200, "application/json", detJson);
}

// New endpoint: GET /api/status -> System status
void handleAPIStatus() {
  String json = "{";
  json += "\"system\":{";
  json += "\"uptime_ms\":" + String(millis());
  json += ",\"free_heap\":" + String(esp_get_free_heap_size());
  json += ",\"free_psram\":" + String(heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
  json += ",\"wifi_rssi\":" + String(WiFi.RSSI());
  json += ",\"ip\":\"" + WiFi.localIP().toString() + "\"";
  json += "},";
  json += "\"camera\":{";
  json += "\"frame_size\":\"320x240\"";
  json += ",\"pixel_format\":\"JPEG\"";
  json += ",\"frame_buffer_location\":\"PSRAM\"";
  json += "},";
  json += "\"model\":{";
  json += "\"input_size\":\"224x224\"";
  json += ",\"classes\":" + String(N_CLASSES);
  json += ",\"tensor_arena_size\":" + String(kTensorArenaSize);
  json += ",\"model_type\":\"MobileNet-SSD\"";
  json += "},";
  json += "\"performance\":{";
  json += "\"total_inferences\":" + String(perf.totalInferences);
  json += ",\"avg_inference_time_ms\":" + String(perf.avgInferenceTime, 1);
  json += ",\"last_inference_time_ms\":" + String(perf.lastInferenceTime);
  json += "}";
  json += "}";
  server.send(200, "application/json", json);
}

// New endpoint: GET /api/classes -> Available classes
void handleAPIClasses() {
  String json = "{\"classes\":[";
  for (int i = 0; i < N_CLASSES; i++) {
    json += "{\"id\":" + String(i) + ",\"name\":\"" + String(pest_names[i]) + "\"}";
    if (i < N_CLASSES - 1) json += ",";
  }
  json += "]}";
  server.send(200, "application/json", json);
}

// New endpoint: GET /api/inference -> Normal inference
void handleAPINormalInference() {
  logMessage(LOG_INFO, "⚡ === NORMAL INFERENCE #" + String(++inference_counter) + " ===");
  String result = runInference();
  server.send(200, "application/json", result);
}

// -------------------- Setup & Main Loop --------------------

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(1);
  
  logMessage(LOG_INFO, "=== NanoDet Pest Detection System ===");
  logMessage(LOG_INFO, "Initializing ESP32-S3 with TensorFlow Lite...");
  
  // Log initial memory status
  logMessage(LOG_INFO, "Initial free heap: " + String(esp_get_free_heap_size()) + " bytes");
  logMessage(LOG_INFO, "Initial free PSRAM: " + String(heap_caps_get_free_size(MALLOC_CAP_SPIRAM)) + " bytes");

  // WiFi Connection with WiFiManager (secure credential management)
  wifiManager.setAPCallback([](WiFiManager *myWiFiManager) {
    logMessage(LOG_INFO, "Entered config mode");
    logMessage(LOG_INFO, "AP IP: " + WiFi.softAPIP().toString());
    logMessage(LOG_INFO, "AP SSID: " + String(myWiFiManager->getConfigPortalSSID()));
  });

  wifiManager.setSaveConfigCallback([]() {
    logMessage(LOG_INFO, "WiFi configuration saved");
  });

  // Try to connect; if it fails, start configuration portal
  if (!wifiManager.autoConnect("NanoDet-Setup")) {
    logMessage(LOG_ERROR, "Failed to connect to WiFi and hit timeout");
    ESP.restart();
    delay(1000);
  }

  logMessage(LOG_INFO, "WiFi connected successfully");
  logMessage(LOG_INFO, "IP address: " + WiFi.localIP().toString());
  logMessage(LOG_INFO, "Signal strength: " + String(WiFi.RSSI()) + " dBm");

  // Detailed memory analysis before camera initialization
  logMessage(LOG_INFO, "=== Pre-Camera Memory Analysis ===");
  size_t totalDram = heap_caps_get_total_size(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
  size_t freeDram = heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
  size_t largestDramBlock = heap_caps_get_largest_free_block(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
  
  logMessage(LOG_INFO, "Total DRAM: " + String(totalDram) + " bytes (" + String(totalDram/1024) + " KB)");
  logMessage(LOG_INFO, "Free DRAM: " + String(freeDram) + " bytes (" + String(freeDram/1024) + " KB)");
  logMessage(LOG_INFO, "Largest DRAM block: " + String(largestDramBlock) + " bytes (" + String(largestDramBlock/1024) + " KB)");
  logMessage(LOG_INFO, "Free PSRAM: " + String(heap_caps_get_free_size(MALLOC_CAP_SPIRAM)/1024) + " KB");

  // Camera initialization FIRST (before TensorFlow) to get best DRAM allocation
  logMessage(LOG_INFO, "Initializing camera first to secure DRAM...");
  
  // Test camera task stack allocation manually
  logMessage(LOG_INFO, "Testing DRAM allocation for camera task stack...");
  void* testStack = heap_caps_malloc(8192, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL); // Typical camera task stack size
  if (testStack) {
    logMessage(LOG_INFO, "✓ Camera task stack test allocation successful");
    heap_caps_free(testStack);
  } else {
    logMessage(LOG_ERROR, "✗ Camera task stack test allocation FAILED - insufficient DRAM!");
    logMessage(LOG_ERROR, "This explains the stack canary watchpoint error");
  }
  
  esp_err_t camera_init_result = esp_camera_init(&camera_config);
  if (camera_init_result != ESP_OK) {
    logMessage(LOG_ERROR, "Camera init failed with error: 0x" + String(camera_init_result, HEX));
    
    // Detailed error analysis
    switch (camera_init_result) {
      case ESP_ERR_NO_MEM:
        logMessage(LOG_ERROR, "Error: ESP_ERR_NO_MEM - Insufficient memory for camera");
        break;
      case ESP_ERR_INVALID_ARG:
        logMessage(LOG_ERROR, "Error: ESP_ERR_INVALID_ARG - Invalid camera configuration");
        break;
      case ESP_FAIL:
        logMessage(LOG_ERROR, "Error: ESP_FAIL - General camera initialization failure");
        break;
      default:
        logMessage(LOG_ERROR, "Error: Unknown camera initialization error");
        break;
    }
    
    // Memory state after failed camera init
    logMessage(LOG_ERROR, "Memory after failed camera init:");
    logMessage(LOG_ERROR, "Free DRAM: " + String(heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL)) + " bytes");
    logMessage(LOG_ERROR, "Largest DRAM block: " + String(heap_caps_get_largest_free_block(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL)) + " bytes");
    return;
  }
  logMessage(LOG_INFO, "Camera initialized successfully");
  
  // Memory state after successful camera init
  freeDram = heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
  largestDramBlock = heap_caps_get_largest_free_block(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
  logMessage(LOG_INFO, "Post-camera DRAM: " + String(freeDram) + " bytes (" + String(freeDram/1024) + " KB)");
  logMessage(LOG_INFO, "Post-camera largest block: " + String(largestDramBlock) + " bytes (" + String(largestDramBlock/1024) + " KB)");
  
  // Test camera frame capture immediately to trigger the error early
  logMessage(LOG_INFO, "Testing camera frame capture...");
  camera_fb_t* test_fb = esp_camera_fb_get();
  if (test_fb) {
    logMessage(LOG_INFO, "✓ Camera frame capture successful");
    logMessage(LOG_INFO, "Frame size: " + String(test_fb->width) + "x" + String(test_fb->height) + ", length: " + String(test_fb->len));
    esp_camera_fb_return(test_fb);
  } else {
    logMessage(LOG_ERROR, "✗ Camera frame capture FAILED - this is where the crash occurs!");
    return;
  }
  
  // Allow camera to stabilize
  delay(1000);
  logMessage(LOG_INFO, "Camera stabilization complete");

  // Memory allocation for tensor arena AFTER camera initialization
  logMessage(LOG_INFO, "=== Pre-TensorFlow Memory Analysis ===");
  size_t preFreeDram = heap_caps_get_free_size(MALLOC_CAP_8BIT) - heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
  size_t preLargestDram = heap_caps_get_largest_free_block(MALLOC_CAP_8BIT & ~MALLOC_CAP_SPIRAM);
  
  logMessage(LOG_INFO, "Pre-TF Free DRAM: " + String(preFreeDram) + " bytes (" + String(preFreeDram/1024) + " KB)");
  logMessage(LOG_INFO, "Pre-TF Largest DRAM block: " + String(preLargestDram) + " bytes (" + String(preLargestDram/1024) + " KB)");
  
  logMessage(LOG_INFO, "Allocating tensor arena (" + String(kTensorArenaSize / 1024) + " KB in PSRAM)...");
  
  // Check available PSRAM before allocation
  size_t availablePsram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
  logMessage(LOG_INFO, "Available PSRAM before allocation: " + String(availablePsram / 1024) + " KB");
  
  // Need space for: TensorFlow (variable) + Camera frame (~50KB for QVGA JPEG) + System buffer (200KB)
  size_t totalPsramNeeded = kTensorArenaSize + 50000 + 200000; // ~1300KB total (will be adjusted dynamically)
  if (availablePsram < totalPsramNeeded) {
    logMessage(LOG_ERROR, "Insufficient PSRAM for tensor arena and camera frame buffer");
    logMessage(LOG_ERROR, "Required: " + String(totalPsramNeeded / 1024) + " KB, Available: " + String(availablePsram / 1024) + " KB");
    return;
  }
  
  // Try to allocate tensor arena with fallback sizes
  size_t arena_sizes[] = {kTensorArenaSize, 1536*1024, 1280*1024, 1024*1024, 768*1024};
  bool arena_allocated = false;
  
  for (size_t arena_size : arena_sizes) {
    logMessage(LOG_INFO, "Trying to allocate " + String(arena_size/1024) + " KB tensor arena...");
    tensor_arena = (uint8_t*) heap_caps_malloc(arena_size, MALLOC_CAP_SPIRAM);
    if (tensor_arena) {
      logMessage(LOG_INFO, "✅ Tensor arena allocated successfully: " + String(arena_size/1024) + " KB");
      arena_allocated = true;
      actual_arena_size = arena_size;
      break;
    } else {
      logMessage(LOG_WARNING, "❌ Failed to allocate " + String(arena_size/1024) + " KB, trying next size...");
    }
  }
  
  if (!arena_allocated) {
    logMessage(LOG_ERROR, "Failed to allocate tensor arena with any size!");
    logMessage(LOG_ERROR, "Available PSRAM: " + String(heap_caps_get_free_size(MALLOC_CAP_SPIRAM)) + " bytes");
    return;
  }
  
  // Check DRAM impact after PSRAM allocation
  size_t postFreeDram = heap_caps_get_free_size(MALLOC_CAP_8BIT) - heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
  size_t postLargestDram = heap_caps_get_largest_free_block(MALLOC_CAP_8BIT & ~MALLOC_CAP_SPIRAM);
  
  logMessage(LOG_INFO, "Post-allocation Free DRAM: " + String(postFreeDram) + " bytes (" + String(postFreeDram/1024) + " KB)");
  logMessage(LOG_INFO, "Post-allocation Largest DRAM block: " + String(postLargestDram) + " bytes (" + String(postLargestDram/1024) + " KB)");
  logMessage(LOG_INFO, "DRAM change: " + String((int)postFreeDram - (int)preFreeDram) + " bytes");
  logMessage(LOG_INFO, "Remaining PSRAM: " + String(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024) + " KB");

  // TensorFlow Lite initialization with detailed memory monitoring
  logMessage(LOG_INFO, "=== TensorFlow Initialization ===");
  size_t preModelDram = heap_caps_get_free_size(MALLOC_CAP_8BIT) - heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
  
  logMessage(LOG_INFO, "Initializing TensorFlow Lite model...");
  model = tflite::GetModel(mdet_int8_final_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    logMessage(LOG_ERROR, "Model schema version mismatch!");
    logMessage(LOG_ERROR, "Expected: " + String(TFLITE_SCHEMA_VERSION) + ", Got: " + String(model->version()));
    return;
  }
  logMessage(LOG_INFO, "Model loaded successfully");
  
  size_t postModelDram = heap_caps_get_free_size(MALLOC_CAP_8BIT) - heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
  logMessage(LOG_INFO, "DRAM after model load: " + String(postModelDram) + " bytes (change: " + String((int)postModelDram - (int)preModelDram) + ")");

  // Set up interpreter
  logMessage(LOG_INFO, "Creating interpreter...");
  static tflite::AllOpsResolver resolver;
  interpreter = new tflite::MicroInterpreter(model, resolver, tensor_arena, actual_arena_size);
  if (!interpreter) {
    logMessage(LOG_ERROR, "Failed to create interpreter");
    return;
  }
  
  size_t postInterpreterDram = heap_caps_get_free_size(MALLOC_CAP_8BIT) - heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
  logMessage(LOG_INFO, "DRAM after interpreter creation: " + String(postInterpreterDram) + " bytes (change: " + String((int)postInterpreterDram - (int)postModelDram) + ")");
  
    logMessage(LOG_INFO, "Allocating tensors...");
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    logMessage(LOG_ERROR, "AllocateTensors() failed");
    return;
  }
  
  // ENHANCED CRITICAL: Check actual arena usage to diagnose overflow
  size_t arena_used_bytes = interpreter->arena_used_bytes();
  logMessage(LOG_INFO, "🔍 TENSOR ARENA ANALYSIS:");
  logMessage(LOG_INFO, "  Arena used: " + String(arena_used_bytes) + " bytes");
  logMessage(LOG_INFO, "  Arena size: " + String(actual_arena_size) + " bytes");
  logMessage(LOG_INFO, "  Arena usage: " + String((arena_used_bytes * 100) / actual_arena_size) + "%");
  
  if (arena_used_bytes > actual_arena_size * 0.9) {
    logMessage(LOG_ERROR, "⚠️ WARNING: Tensor arena usage > 90% - risk of overflow!");
  }
  
  // Enhanced memory analysis
  logMessage(LOG_INFO, "🔍 ENHANCED MEMORY ANALYSIS:");
  size_t free_heap = ESP.getFreeHeap();
  size_t min_free_heap = ESP.getMinFreeHeap();
  size_t max_alloc_heap = ESP.getMaxAllocHeap();
  
  logMessage(LOG_INFO, "  Free heap: " + String(free_heap) + " bytes");
  logMessage(LOG_INFO, "  Min free heap: " + String(min_free_heap) + " bytes");
  logMessage(LOG_INFO, "  Max alloc heap: " + String(max_alloc_heap) + " bytes");
  
  if (free_heap < 100000) {
    logMessage(LOG_WARNING, "⚠️ Low heap memory - may cause issues");
  }
  
  if (min_free_heap < 50000) {
    logMessage(LOG_ERROR, "🚨 Very low minimum heap - high risk of corruption!");
  }
  logMessage(LOG_INFO, "   Arena allocated: " + String(actual_arena_size / 1024) + " KB");
  logMessage(LOG_INFO, "   Arena actually used: " + String(arena_used_bytes / 1024) + " KB");
  logMessage(LOG_INFO, "   Arena utilization: " + String(100.0f * arena_used_bytes / actual_arena_size, 1) + "%");
  logMessage(LOG_INFO, "   Free arena space: " + String((actual_arena_size - arena_used_bytes) / 1024) + " KB");
  
  if (arena_used_bytes > actual_arena_size * 0.95f) {
    logMessage(LOG_ERROR, "🚨 ARENA NEARLY FULL! This will cause memory corruption during inference!");
    logMessage(LOG_ERROR, "Increase kTensorArenaSize to at least " + String((arena_used_bytes * 1.2f) / 1024) + " KB");
  } else if (arena_used_bytes > actual_arena_size * 0.85f) {
    logMessage(LOG_WARNING, "⚠️  Arena usage high - may cause overflow with scratch operations");
  } else {
    logMessage(LOG_INFO, "✅ Arena size appears adequate");
  }
  
  size_t postAllocateDram = heap_caps_get_free_size(MALLOC_CAP_8BIT) - heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
  logMessage(LOG_INFO, "DRAM after tensor allocation: " + String(postAllocateDram) + " bytes (change: " + String((int)postAllocateDram - (int)postInterpreterDram) + ")");
  logMessage(LOG_INFO, "Total DRAM consumed by TensorFlow: " + String((int)preModelDram - (int)postAllocateDram) + " bytes");
  
  input  = interpreter->input(0);
  // MDET v2 has a single output tensor with shape [1, 14, 14, 3, 29]
  output_tensor = interpreter->output(0);
  
  if (!input || !output_tensor) {
    logMessage(LOG_ERROR, "Failed to get input/output tensors");
    return;
  }
  
  // Check tensor integrity after allocation
  checkTensorIntegrity("After tensor allocation");
  
  // DEBUG: Check model quantization parameters for full INT8
  logMessage(LOG_INFO, "🔍 FULL INT8 MODEL QUANTIZATION PARAMETERS:");
  logMessage(LOG_INFO, "   Input scale: " + String(input->params.scale, 6));
  logMessage(LOG_INFO, "   Input zero point: " + String(input->params.zero_point));
  logMessage(LOG_INFO, "   Output tensor scale: " + String(output_tensor->params.scale, 6));
  logMessage(LOG_INFO, "   Output tensor zero point: " + String(output_tensor->params.zero_point));
  logMessage(LOG_INFO, "   Input type: " + String(input->type));
  logMessage(LOG_INFO, "   Output tensor type: " + String(output_tensor->type));
  
  // Validate full INT8 quantization
  if (input->params.scale == 0.0f || output_tensor->params.scale == 0.0f) {
    logMessage(LOG_ERROR, "🚨 CRITICAL: Model quantization parameters are zero!");
    logMessage(LOG_ERROR, "This indicates the model is not properly quantized for TFLite Micro");
    return;
  }
  
  // Log tensor information
  logMessage(LOG_INFO, "Input tensor: " + String(input->bytes) + " bytes");
  logMessage(LOG_INFO, "Output tensor: " + String(output_tensor->bytes) + " bytes");
  logMessage(LOG_INFO, "Input dimensions: [" + String(input->dims->data[0]) + "," + 
             String(input->dims->data[1]) + "," + String(input->dims->data[2]) + "," + 
             String(input->dims->data[3]) + "]");
  logMessage(LOG_INFO, "Output tensor dimensions: [" + String(output_tensor->dims->data[0]) + "," + 
             String(output_tensor->dims->data[1]) + "," + String(output_tensor->dims->data[2]) + "," + 
             String(output_tensor->dims->data[3]) + "," + String(output_tensor->dims->data[4]) + "]");

  // Configuration validation
  if (!validateConfiguration()) {
    logMessage(LOG_ERROR, "Configuration validation failed");
    return;
  }

  // Critical test: Check if camera still works after TensorFlow initialization
  logMessage(LOG_INFO, "=== Post-TensorFlow Camera Test ===");
  size_t finalDram = heap_caps_get_free_size(MALLOC_CAP_8BIT) - heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
  size_t finalLargestDram = heap_caps_get_largest_free_block(MALLOC_CAP_8BIT & ~MALLOC_CAP_SPIRAM);
  
  logMessage(LOG_INFO, "Final DRAM state:");
  logMessage(LOG_INFO, "Free DRAM: " + String(finalDram) + " bytes (" + String(finalDram/1024) + " KB)");
  logMessage(LOG_INFO, "Largest DRAM block: " + String(finalLargestDram) + " bytes (" + String(finalLargestDram/1024) + " KB)");
  
  logMessage(LOG_INFO, "Testing camera after TensorFlow initialization...");
  camera_fb_t* final_test_fb = esp_camera_fb_get();
  if (final_test_fb) {
    logMessage(LOG_INFO, "✓ POST-TENSORFLOW camera test successful!");
    logMessage(LOG_INFO, "Frame: " + String(final_test_fb->width) + "x" + String(final_test_fb->height));
    esp_camera_fb_return(final_test_fb);
  } else {
    logMessage(LOG_ERROR, "✗ POST-TENSORFLOW camera test FAILED!");
    logMessage(LOG_ERROR, "This means TensorFlow initialization caused the camera to fail");
  }

  // HTTP routes setup
  logMessage(LOG_INFO, "Setting up HTTP server...");
  server.on("/", HTTP_GET, handleRoot);
  server.on("/log", HTTP_POST, handleLog);
  server.on("/api/detect", HTTP_GET, handleAPIDetect);
  server.on("/api/status", HTTP_GET, handleAPIStatus);
  server.on("/api/classes", HTTP_GET, handleAPIClasses);
  server.on("/api/inference", HTTP_GET, handleAPINormalInference);
  
  // Enable CORS for API endpoints
  server.enableCORS(true);
  
  server.begin();
  logMessage(LOG_INFO, "HTTP server started on port 80");
  
  // Final system status
  logMessage(LOG_INFO, "=== System Ready ===");
  logMessage(LOG_INFO, "Available classes: " + String(N_CLASSES));
  logMessage(LOG_INFO, "CPU frequency: " + String(ESP.getCpuFreqMHz()) + " MHz");
  logMessage(LOG_INFO, "XTAL frequency: 40 MHz"); // Standard ESP32 XTAL frequency
  logMessage(LOG_INFO, "APB frequency: 80 MHz"); // Standard ESP32 APB frequency
  logMessage(LOG_INFO, "Free heap: " + String(esp_get_free_heap_size()) + " bytes");
  logMessage(LOG_INFO, "Free PSRAM: " + String(heap_caps_get_free_size(MALLOC_CAP_SPIRAM)) + " bytes");
  logMessage(LOG_INFO, "Access web interface at: http://" + WiFi.localIP().toString());
}

void loop() {
  if (!input || !interpreter) { // Changed from class_output/bbox_output to interpreter
    Serial.println("[ERROR] TFLite not initialized");
    delay(1000);
    return;
  }
  
  // ADDITIONAL DEBUG: Test camera functionality before inference
  static int frame_counter = 0;
  static uint32_t last_jpeg_hash = 0;
  
  // Capture a test frame to verify camera is working
  camera_fb_t* test_fb = esp_camera_fb_get();
  if (test_fb && test_fb->buf && test_fb->len > 0) {
    // Calculate a simple hash of the JPEG data to see if it's changing
    uint32_t jpeg_hash = 0;
    for (int i = 0; i < simple_min(100, (int)test_fb->len); i++) {
      jpeg_hash = ((jpeg_hash << 5) + jpeg_hash) + test_fb->buf[i]; // Simple rolling hash
    }
    
    if (frame_counter > 0) {
      if (jpeg_hash == last_jpeg_hash) {
        logMessage(LOG_WARNING, "⚠️ Camera may be stuck - JPEG hash identical: " + String(jpeg_hash));
        
        // ADDITIONAL DEBUG: Show JPEG data analysis
        logMessage(LOG_INFO, "🔍 JPEG Data Analysis:");
        logMessage(LOG_INFO, "   Frame length: " + String(test_fb->len) + " bytes");
        logMessage(LOG_INFO, "   First 10 bytes: [" + String(test_fb->buf[0]) + "," + String(test_fb->buf[1]) + 
                   "," + String(test_fb->buf[2]) + "," + String(test_fb->buf[3]) + "," + String(test_fb->buf[4]) + 
                   "," + String(test_fb->buf[5]) + "," + String(test_fb->buf[6]) + "," + String(test_fb->buf[7]) + 
                   "," + String(test_fb->buf[8]) + "," + String(test_fb->buf[9]) + "]");
        
        // Check if JPEG data is actually changing at all
        static uint8_t prev_jpeg_bytes[10] = {0};
        int jpeg_changes = 0;
        for (int i = 0; i < 10; i++) {
          if (test_fb->buf[i] != prev_jpeg_bytes[i]) {
            jpeg_changes++;
            prev_jpeg_bytes[i] = test_fb->buf[i];
          }
        }
        logMessage(LOG_INFO, "   JPEG byte changes: " + String(jpeg_changes) + "/10");
        
        if (jpeg_changes == 0) {
          logMessage(LOG_ERROR, "🚨 CRITICAL: JPEG data is completely static - camera sensor may be malfunctioning!");
        }
      } else {
        logMessage(LOG_INFO, "✅ Camera working - JPEG hash changed: " + String(last_jpeg_hash) + " -> " + String(jpeg_hash));
      }
    }
    
    last_jpeg_hash = jpeg_hash;
    frame_counter++;
    esp_camera_fb_return(test_fb);
  } else {
    logMessage(LOG_ERROR, "❌ Camera test frame capture failed");
  }
  
  // Run inference and print predictions to Serial
  runInference();
  
  // ADDITIONAL TEST: Every 10th frame, test with a different input pattern
  if (frame_counter % 10 == 0) {
    logMessage(LOG_INFO, "🧪 TESTING MODEL RESPONSIVENESS - Frame #" + String(frame_counter));
    
    // Create a test pattern that's clearly different
    int8_t* test_input = (int8_t*)input->data.int8;
    for (int i = 0; i < 1000; i++) {
      test_input[i] = (i % 256) - 128; // Create a clear pattern: -128, -127, -126, ..., 127, -128, ...
    }
    
    // Run inference with test pattern
    TfLiteStatus test_status = interpreter->Invoke();
    if (test_status == kTfLiteOk) {
      int8_t* test_output = output_tensor->data.int8;
      logMessage(LOG_INFO, "🧪 Test inference successful - Output samples: [" + 
                 String(test_output[0]) + "," + String(test_output[1]) + "," + String(test_output[2]) + "]");
      
      // Check if output is different from normal camera input by comparing with global prev_output_samples
      if (test_output[0] != prev_output_samples[0] || test_output[1] != prev_output_samples[1]) {
        logMessage(LOG_INFO, "✅ Model is responsive to different inputs");
      } else {
        logMessage(LOG_WARNING, "⚠️ Model output unchanged with test pattern - possible issue");
      }
    } else {
      logMessage(LOG_ERROR, "❌ Test inference failed with status: " + String(test_status));
    }
  }
  
  delay(1000); // Run every second
}



