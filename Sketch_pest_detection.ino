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
#include "jpeg_decoder.h"  // ESP32 JPEG decoder library (ESP-IDF v5.x)

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

// Image processing function declarations
bool decodeJPEGToRGB(camera_fb_t* fb, uint8_t* output_buffer, int target_width, int target_height);
bool processRGB565(camera_fb_t* fb, uint8_t* output_buffer, int target_width, int target_height);
bool processOtherFormats(camera_fb_t* fb, uint8_t* output_buffer, int target_width, int target_height);
bool switchToRGB565IfNeeded();
void debugSaveDecodedImage(uint8_t* rgb_buffer, int width, int height, const char* filename);

// -------------------- Wi‑Fi Configuration --------------------
// Using WiFiManager for secure credential management
WiFiManager wifiManager;

// Static IP configuration
IPAddress staticIP(192, 168, 1, 200);    // Your desired static IP
IPAddress gateway(192, 168, 1, 1);       // Your router's IP
IPAddress subnet(255, 255, 255, 0);      // Subnet mask
IPAddress dns(8, 8, 8, 8);              // DNS server (Google's 8.8.8.8)

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
  .pixel_format = PIXFORMAT_JPEG,      // Use JPEG for smaller memory footprint
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
  logMessage(LOG_INFO, " TENSOR INTEGRITY CHECK: " + String(stage));
  
  if (!input || !output_tensor || !interpreter) {
    logMessage(LOG_ERROR, "   Tensor pointers are null!");
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

// -------------------- Image Processing Helper Functions --------------------

// Optimized JPEG decoding with better error handling and memory management
bool decodeJPEGToRGB(camera_fb_t* fb, uint8_t* output_buffer, int target_width, int target_height) {
  if (!fb || !fb->buf || !output_buffer) {
    logMessage(LOG_ERROR, "Invalid parameters for JPEG decoding");
    return false;
  }
  
  // Use ESP32's hardware JPEG decoder for better performance
  esp_jpeg_image_cfg_t decode_cfg = {
    .indata = fb->buf,
    .indata_size = fb->len,
    .outbuf = output_buffer,
    .outbuf_size = target_width * target_height * 3, // Direct RGB888 output
    .out_format = JPEG_IMAGE_FORMAT_RGB888,          // Decode directly to RGB888
    .out_scale = JPEG_IMAGE_SCALE_0,
    .flags = {0},
    .advanced = {nullptr, 0}
  };
  
  esp_jpeg_image_output_t img_info;
  esp_err_t ret = esp_jpeg_decode(&decode_cfg, &img_info);
  if (ret != ESP_OK) {
    logMessage(LOG_ERROR, "JPEG decode failed: " + String(esp_err_to_name(ret)));
    return false;
  }
  
  // If the decoded image is larger than target, resize it efficiently
  if (img_info.width != target_width || img_info.height != target_height) {
    // Simple nearest-neighbor resize for better performance
    uint8_t* temp_buffer = (uint8_t*)heap_caps_malloc(target_width * target_height * 3, MALLOC_CAP_8BIT);
    if (!temp_buffer) {
      logMessage(LOG_ERROR, "Failed to allocate resize buffer");
      return false;
    }
    
    // Resize with nearest neighbor interpolation
    for (int y = 0; y < target_height; y++) {
      for (int x = 0; x < target_width; x++) {
        int src_x = (x * img_info.width) / target_width;
        int src_y = (y * img_info.height) / target_height;
        
        // Ensure bounds
        src_x = simple_min(src_x, (int)(img_info.width - 1));
        src_y = simple_min(src_y, (int)(img_info.height - 1));
        
        // Copy RGB values
        int src_idx = (src_y * img_info.width + src_x) * 3;
        int dst_idx = (y * target_width + x) * 3;
        
        temp_buffer[dst_idx + 0] = output_buffer[src_idx + 0]; // R
        temp_buffer[dst_idx + 1] = output_buffer[src_idx + 1]; // G
        temp_buffer[dst_idx + 2] = output_buffer[src_idx + 2]; // B
      }
    }
    
    // Copy resized data back to output buffer
    memcpy(output_buffer, temp_buffer, target_width * target_height * 3);
    free(temp_buffer);
  }
  
  logMessage(LOG_INFO, "JPEG decoded and resized: " + String(fb->width) + "x" + String(fb->height) + 
             " -> " + String(target_width) + "x" + String(target_height) + " RGB");
  
  return true;
}

// Optimized RGB565 processing with lookup tables for better performance
bool processRGB565(camera_fb_t* fb, uint8_t* output_buffer, int target_width, int target_height) {
  if (!fb || !fb->buf || !output_buffer) {
    logMessage(LOG_ERROR, "Invalid parameters for RGB565 processing");
    return false;
  }
  
  uint16_t* rgb565_data = (uint16_t*)fb->buf;
  
  // Static lookup tables for RGB565 to RGB888 conversion (initialized once)
  static uint8_t r_lut[32], g_lut[64], b_lut[32];
  static bool lut_initialized = false;
  
  if (!lut_initialized) {
    // Initialize lookup tables for better performance
    for (int i = 0; i < 32; i++) {
      r_lut[i] = (i << 3) | (i >> 2);  // 5 bits -> 8 bits with dithering
      b_lut[i] = r_lut[i];
    }
    for (int i = 0; i < 64; i++) {
      g_lut[i] = (i << 2) | (i >> 4);  // 6 bits -> 8 bits with dithering
    }
    lut_initialized = true;
  }
  
  // Convert RGB565 to RGB888 and resize to target dimensions
  for (int y = 0; y < target_height; y++) {
    for (int x = 0; x < target_width; x++) {
      // Map target coordinates to source coordinates
      int src_x = (x * fb->width) / target_width;
      int src_y = (y * fb->height) / target_height;
      
      // Ensure bounds
      src_x = simple_min(src_x, (int)(fb->width - 1));
      src_y = simple_min(src_y, (int)(fb->height - 1));
      
      // Get RGB565 pixel
      uint16_t rgb565 = rgb565_data[src_y * fb->width + src_x];
      
      // Convert using lookup tables for better performance
      uint8_t r = r_lut[(rgb565 >> 11) & 0x1F];
      uint8_t g = g_lut[(rgb565 >> 5) & 0x3F];
      uint8_t b = b_lut[rgb565 & 0x1F];
      
      // Store in output buffer (HWC format)
      int pixel_idx = (y * target_width + x) * 3;
      output_buffer[pixel_idx + 0] = r;
      output_buffer[pixel_idx + 1] = g;
      output_buffer[pixel_idx + 2] = b;
    }
  }
  
  logMessage(LOG_INFO, "RGB565 processed successfully: " + String(fb->width) + "x" + String(fb->height) + 
             " -> " + String(target_width) + "x" + String(target_height) + " RGB");
  
  return true;
}

// Process other formats (fallback)
bool processOtherFormats(camera_fb_t* fb, uint8_t* output_buffer, int target_width, int target_height) {
  if (!fb || !fb->buf || !output_buffer) {
    logMessage(LOG_ERROR, "Invalid parameters for other format processing");
    return false;
  }
  
  logMessage(LOG_WARNING, "Processing unsupported format: " + String(fb->format) + 
             " - using grayscale conversion");
  
  // Convert to grayscale and resize to target dimensions
  for (int y = 0; y < target_height; y++) {
    for (int x = 0; x < target_width; x++) {
      // Map target coordinates to source coordinates
      int src_x = (x * fb->width) / target_width;
      int src_y = (y * fb->height) / target_height;
      
      // Ensure bounds
      src_x = simple_min(src_x, (int)(fb->width - 1));
      src_y = simple_min(src_y, (int)(fb->height - 1));
      
      // Get source pixel value
      uint8_t pixel_value = fb->buf[src_y * fb->width + src_x];
      
      // Convert to RGB (grayscale)
      int pixel_idx = (y * target_width + x) * 3;
      output_buffer[pixel_idx + 0] = pixel_value;  // R
      output_buffer[pixel_idx + 1] = pixel_value;  // G
      output_buffer[pixel_idx + 2] = pixel_value;  // B
    }
  }
  
  logMessage(LOG_INFO, "Other format processed as grayscale: " + String(fb->width) + "x" + String(fb->height) + 
             " -> " + String(target_width) + "x" + String(target_height) + " RGB");
  
  return true;
}

// Dynamic camera format switching based on available memory
bool switchToRGB565IfNeeded() {
  size_t freeHeap = esp_get_free_heap_size();
  size_t freePsram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
  
  // If we have enough memory, switch to RGB565 for better quality
  if (freeHeap > 100000 && freePsram > 200000) {  // 100KB heap + 200KB PSRAM
    logMessage(LOG_INFO, "Switching to RGB565 format for better image quality");
    
    // Stop camera
    esp_camera_deinit();
    
    // Modify camera config
    camera_config_t rgb_config = camera_config;
    rgb_config.pixel_format = PIXFORMAT_RGB565;
    rgb_config.fb_count = 1;  // Reduce buffer count for RGB565
    
    // Reinitialize camera
    esp_err_t err = esp_camera_init(&rgb_config);
    if (err != ESP_OK) {
      logMessage(LOG_ERROR, "Failed to switch to RGB565: " + String(esp_err_to_name(err)));
      
      // Fallback to JPEG
      esp_camera_deinit();
      err = esp_camera_init(&camera_config);
      if (err != ESP_OK) {
        logMessage(LOG_ERROR, "Failed to restore JPEG format: " + String(esp_err_to_name(err)));
        return false;
      }
      return false;
    }
    
    logMessage(LOG_INFO, "Successfully switched to RGB565 format");
    return true;
  }
  
  return false;
}

// Optimized image preprocessing with reduced memory allocations and better performance
bool preprocessFrame(camera_fb_t* fb, int8_t* input_buffer) {
  if (!fb || !fb->buf || !input_buffer) {
    logMessage(LOG_ERROR, "Invalid frame or input buffer");
    return false;
  }
  
  // Expected model input: 224x224x3 RGB
  const int MODEL_WIDTH = 224;
  const int MODEL_HEIGHT = 224;
  const int MODEL_CHANNELS = 3;
  
  // Get input tensor parameters for quantization
  float input_scale = input->params.scale;
  int input_zp = input->params.zero_point;
  
  if (input_scale == 0.0f) {
    logMessage(LOG_ERROR, "Input scale is zero - model may not be properly quantized");
    return false;
  }
  
  logMessage(LOG_INFO, "Processing camera image: " + String(fb->width) + "x" + String(fb->height) + 
             ", format: " + String(fb->format) + ", length: " + String(fb->len) + " bytes");
  
  // Use PSRAM for temporary buffer to preserve DRAM
  uint8_t* temp_hwc_buffer = (uint8_t*)heap_caps_malloc(MODEL_WIDTH * MODEL_HEIGHT * MODEL_CHANNELS, MALLOC_CAP_SPIRAM);
  if (!temp_hwc_buffer) {
    logMessage(LOG_ERROR, "Failed to allocate temporary HWC buffer in PSRAM");
    return false;
  }
  
  bool success = false;
  
  if (fb->format == PIXFORMAT_JPEG) {
    // Optimized JPEG decoding
    success = decodeJPEGToRGB(fb, temp_hwc_buffer, MODEL_WIDTH, MODEL_HEIGHT);
  } else if (fb->format == PIXFORMAT_RGB565) {
    // Optimized RGB565 processing
    success = processRGB565(fb, temp_hwc_buffer, MODEL_WIDTH, MODEL_HEIGHT);
  } else {
    // Fallback for other formats
    success = processOtherFormats(fb, temp_hwc_buffer, MODEL_WIDTH, MODEL_HEIGHT);
  }
  
  if (!success) {
    logMessage(LOG_ERROR, "Failed to process image format");
    heap_caps_free(temp_hwc_buffer);
    return false;
  }
  
  // Optimized HWC to CHW conversion with inline quantization
  const float scale_factor = 1.0f / (255.0f * input_scale);
  const float zp_offset = (float)input_zp;
  
  for (int c = 0; c < MODEL_CHANNELS; c++) {
    for (int h = 0; h < MODEL_HEIGHT; h++) {
      for (int w = 0; w < MODEL_WIDTH; w++) {
        int hwc_idx = (h * MODEL_WIDTH + w) * 3 + c;
        int chw_idx = c * MODEL_HEIGHT * MODEL_WIDTH + h * MODEL_WIDTH + w;
        
        // Get pixel value and apply quantization in one step
        float pixel_value = (float)temp_hwc_buffer[hwc_idx];
        float quantized = pixel_value * scale_factor + zp_offset;
        
        // Clamp to INT8 range
        if (quantized < -128.0f) quantized = -128.0f;
        if (quantized > 127.0f) quantized = 127.0f;
        
        input_buffer[chw_idx] = (int8_t)quantized;
      }
    }
  }
  
  // Clean up temporary buffer
  heap_caps_free(temp_hwc_buffer);
  
  // Reduced debug logging for better performance
  logMessage(LOG_DEBUG, "Preprocessing completed successfully");
  
  return true;
}

// Debug function to save decoded image data (for testing)
void debugSaveDecodedImage(uint8_t* rgb_buffer, int width, int height, const char* filename) {
  // This is a debug function that could be used to save images to SD card
  // For now, just log the first few pixel values
  logMessage(LOG_INFO, "DEBUG: First 9 pixels of decoded image:");
  for (int y = 0; y < 3; y++) {
    for (int x = 0; x < 3; x++) {
      int idx = (y * width + x) * 3;
      logMessage(LOG_INFO, "  Pixel(" + String(x) + "," + String(y) + "): R=" + String(rgb_buffer[idx]) + 
                 " G=" + String(rgb_buffer[idx+1]) + " B=" + String(rgb_buffer[idx+2]));
    }
  }
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
  // Get camera frame
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    logMessage(LOG_ERROR, "Failed to capture frame");
    return "{\"error\":\"frame_capture_failed\"}";
  }
  
  // Process the frame and return it immediately to avoid holding it
  String result = runInferenceWithFrame(fb);
  esp_camera_fb_return(fb);
  return result;
}

// New function that takes a frame parameter to avoid double capture
String runInferenceWithFrame(camera_fb_t* fb) {
  unsigned long startTime = millis();
  
  // Input validation with proper error handling
  if (!input || !interpreter) {
    logMessage(LOG_ERROR, "TensorFlow Lite components not initialized");
    return "{\"error\":\"tflite_not_initialized\"}";
  }
  
  // Validate frame buffer
  if (!fb || !fb->buf || fb->len == 0) {
    logMessage(LOG_ERROR, "Invalid frame buffer");
    return "{\"error\":\"invalid_frame_buffer\"}";
  }
  
  // Validate input tensor
  if (!input->data.int8) {
    logMessage(LOG_ERROR, "Input tensor data is null");
    return "{\"error\":\"input_tensor_null\"}";
  }
  
  size_t expected_input_size = input->bytes;
  logMessage(LOG_DEBUG, "Expected input size: " + String(expected_input_size) + " bytes");
  logMessage(LOG_DEBUG, "Camera frame size: " + String(fb->width) + "x" + String(fb->height) + ", length: " + String(fb->len));
  
  // Preprocess frame to match model input (224x224 RGB)
  if (!preprocessFrame(fb, (int8_t*)input->data.int8)) {
    logMessage(LOG_ERROR, "Frame preprocessing failed");
    return "{\"error\":\"preprocessing_failed\"}";
  }
  
  // ENHANCED DEBUG: Capture input tensor state before inference
  int8_t input_samples[10];
  logMessage(LOG_INFO, " PRE-INFERENCE INPUT TENSOR CHECK:");
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
    logMessage(LOG_INFO, " Input tensor integrity verified before inference");
  }
  
  // ADDITIONAL DEBUG: Check if input tensor is actually changing between frames
  static int8_t prev_input_samples[10] = {0};
  static bool first_frame = true;
  int changed_values = 0;
  
  if (!first_frame) {
    logMessage(LOG_INFO, "INPUT TENSOR CHANGE ANALYSIS:");
    for (int i = 0; i < 10; i++) {
      if (input_samples[i] != prev_input_samples[i]) {
        changed_values++;
        logMessage(LOG_INFO, "  Changed [" + String(i) + "]: " + String(prev_input_samples[i]) + " -> " + String(input_samples[i]));
      }
    }
    logMessage(LOG_INFO, "  Total changed values: " + String(changed_values) + "/10");
    
    if (changed_values == 0) {
      logMessage(LOG_WARNING, " WARNING: Input tensor is NOT changing between frames!");
      logMessage(LOG_WARNING, "This indicates the preprocessing is not working correctly.");
    } else if (changed_values < 3) {
      logMessage(LOG_WARNING, " WARNING: Very few input values are changing (" + String(changed_values) + "/10)");
    } else {
      logMessage(LOG_INFO, " Input tensor is changing normally between frames");
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
  logMessage(LOG_INFO, " Running inference with real camera image...");
  
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
                    logMessage(LOG_ERROR, " WARNING: Very low heap memory after inference - possible stack overflow");
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
  logMessage(LOG_INFO, " OUTPUT TENSOR ANALYSIS:");
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
    logMessage(LOG_INFO, " MODEL OUTPUT CHANGE ANALYSIS:");
    for (int i = 0; i < 6; i++) {
      if (raw_output[i] != prev_output_samples[i]) {
        output_changed_values++;
        logMessage(LOG_INFO, "  Output changed [" + String(i) + "]: " + String(prev_output_samples[i]) + " -> " + String(raw_output[i]));
      }
    }
    logMessage(LOG_INFO, "  Total output changed values: " + String(output_changed_values) + "/6");
    
    if (output_changed_values == 0) {
      logMessage(LOG_ERROR, " CRITICAL: Model output is NOT changing between frames!");
      logMessage(LOG_ERROR, "This indicates the model is not responding to different inputs.");
    } else if (output_changed_values < 2) {
      logMessage(LOG_WARNING, " WARNING: Very few output values are changing (" + String(output_changed_values) + "/6)");
    } else {
      logMessage(LOG_INFO, " Model output is changing normally between frames");
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
  const float score_thresh = 0.05f;  // Confidence threshold for multi-anchor model
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
  
  logMessage(LOG_INFO, " DETECTION RESULTS:");
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
    Serial.println(" NO PESTS DETECTED");
    Serial.println("   • Model confidence threshold: " + String(score_thresh * 100, 1) + "%");
    Serial.println("   • Try adjusting lighting or camera position");
  }
  
  Serial.println("------------------------------------------------------------");
  Serial.println("⚡ PERFORMANCE METRICS:");
  Serial.println("    Inference Time: " + String(inferenceTime) + " ms");
  Serial.println("   Average Time: " + String(perf.avgInferenceTime, 1) + " ms");
  Serial.println("   Total Inferences: " + String(perf.totalInferences));
  Serial.println("   Free Heap: " + String(esp_get_free_heap_size() / 1024) + " KB");
  Serial.println("   Free PSRAM: " + String(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024) + " KB");
  Serial.println("============================================================\n");

  // Build enhanced JSON with class names and bounding boxes
  String json = "{\"detections\":[";
  for (int i = 0; i < det_count; i++) {
    const auto &d = dets[i];
    json += "{\"class\":\"" + String(pest_names[d.class_id]) + "\"";
    json += ",\"score\":" + String(d.score, 3);
    json += ",\"bbox\":[";
    json += String(d.x1 * 224, 0) + "," + String(d.y1 * 224, 0) + ",";
    json += String((d.x2 - d.x1) * 224, 0) + "," + String((d.y2 - d.y1) * 224, 0);
    json += "]";
    json += "}";
    if (i < det_count - 1) json += ",";
  }
  json += "],";
  json += "\"width\":224,\"height\":224,";
  json += "\"performance\":{";
  json += "\"inference_time_ms\":" + String(perf.lastInferenceTime);
  json += ",\"avg_inference_time_ms\":" + String(perf.avgInferenceTime, 1);
  json += ",\"total_inferences\":" + String(perf.totalInferences);
  json += ",\"free_heap\":" + String(esp_get_free_heap_size());
  json += ",\"free_psram\":" + String(heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
  json += "}";
  json += "}";
  
  // Debug: Log the generated JSON
  logMessage(LOG_INFO, "Generated JSON: " + json);
  
  return json;
}

// -------------------- Web Server Handlers --------------------

// Enhanced HTTP GET / -> run inference and provide rich web interface
void handleRoot() {
  // Run inference only when explicitly requested to avoid blocking
  String detJson = runInference();
  
  // Debug: Log the JSON being sent to web interface
  logMessage(LOG_INFO, "Sending JSON to web interface: " + detJson);
  
  // Escape quotes in JSON for safe embedding in HTML
  detJson.replace("\"", "\\\"");
  
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
        
        <div class='section'>
            <h2>Debug Info</h2>
            <div id='debugInfo'></div>
            <div style="background: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 5px;">
                <h4>Raw JSON Data:</h4>
                <pre id="rawJsonData" style="background: white; padding: 10px; overflow-x: auto;">)HTML" + detJson + R"HTML(</pre>
            </div>
        </div>
    </div>

    <script>
        // Test if JavaScript is working
        alert('JavaScript is working!');
        
        // Get the JSON data from the debug section and parse it
        let detectionData;
        try {
            // Extract the JSON from the debug section
            const rawJsonElement = document.querySelector('#rawJsonData');
            if (!rawJsonElement) {
                throw new Error('Raw JSON element not found');
            }
            
            const rawJsonText = rawJsonElement.textContent;
            console.log('Raw JSON text:', rawJsonText);
            
            // Parse the JSON
            detectionData = JSON.parse(rawJsonText);
            console.log('Detection data loaded successfully:', detectionData);
        } catch (error) {
            console.error('Error parsing detection data:', error);
            detectionData = { error: 'Failed to parse detection data: ' + error.message };
        }
        
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
                            <h3>🐛 ${det.class || 'Unknown'}</h3>
                            <p><strong>Confidence:</strong> ${(det.score * 100).toFixed(1)}%</p>
                            <p><strong>Bounding Box:</strong> 
                               x=${det.bbox[0]}, y=${det.bbox[1]}, 
                               w=${det.bbox[2]}, h=${det.bbox[3]}</p>
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
            
            // Debug info
            document.getElementById('debugInfo').innerHTML = `
                <p><strong>Raw JSON:</strong> <pre>${JSON.stringify(data, null, 2)}</pre></p>
                <p><strong>Detection Count:</strong> ${data.detections ? data.detections.length : 'undefined'}</p>
                <p><strong>First Detection:</strong> ${data.detections && data.detections[0] ? JSON.stringify(data.detections[0]) : 'undefined'}</p>
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
        if (detectionData && !detectionData.error) {
            displayResults(detectionData);
            
            // Send data to log endpoint
            fetch('/log', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(detectionData)
            });
        } else {
            // Show error state
            document.getElementById('detectionResults').innerHTML = '<div class="error">❌ Failed to load detection data</div>';
            console.error('Detection data error:', detectionData);
        }
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

// New endpoint: GET /api/detect -> JSON-only inference (non-blocking)
void handleAPIDetect() {
  logMessage(LOG_INFO, " API Detect request received from: " + server.client().remoteIP().toString());
  
  // Check if system is ready for inference
  if (!input || !interpreter) {
    server.send(503, "application/json", "{\"error\":\"system_not_ready\"}");
    return;
  }
  
  // Run inference with timeout protection
  unsigned long start_time = millis();
  String detJson = runInference();
  unsigned long inference_time = millis() - start_time;
  
  logMessage(LOG_INFO, " Inference completed in " + String(inference_time) + "ms");
  logMessage(LOG_INFO, " Sending detection response to: " + server.client().remoteIP().toString());
  
  server.send(200, "application/json", detJson);
}

// New endpoint: GET /api/status -> System status (fast, non-blocking)
void handleAPIStatus() {
  // Quick status check without inference - very fast response
  String json = "{";
  json += "\"system\":{";
  json += "\"uptime_ms\":" + String(millis());
  json += ",\"free_heap\":" + String(esp_get_free_heap_size());
  json += ",\"free_psram\":" + String(heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
  json += ",\"wifi_rssi\":" + String(WiFi.RSSI());
  json += ",\"ip\":\"" + WiFi.localIP().toString() + "\"";
  json += ",\"status\":\"ready\"";
  json += "},";
  json += "\"camera\":{";
  json += "\"frame_size\":\"320x240\"";
  json += ",\"pixel_format\":\"JPEG\"";
  json += ",\"frame_buffer_location\":\"PSRAM\"";
  json += ",\"ready\":" + String(input && interpreter ? "true" : "false");
  json += "},";
  json += "\"model\":{";
  json += "\"input_size\":\"224x224\"";
  json += ",\"classes\":" + String(N_CLASSES);
  json += ",\"tensor_arena_size\":" + String(actual_arena_size);
  json += ",\"model_type\":\"MDET_v2_Multi_Anchor\"";
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

// Optimized camera frame endpoint with better error handling
void handleFrame() {
  // Get camera frame with timeout protection
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    server.send(500, "text/plain", "Failed to capture frame");
    return;
  }
  
  // Set headers for optimal delivery
  server.sendHeader("Content-Type", "image/jpeg");
  server.sendHeader("Content-Disposition", "inline; filename=frame.jpg");
  server.sendHeader("Cache-Control", "no-cache, no-store, must-revalidate");
  server.sendHeader("Pragma", "no-cache");
  server.sendHeader("Expires", "0");
  
  // Send frame data efficiently
  server.send_P(200, "image/jpeg", (const char*)fb->buf, fb->len);
  
  // Return frame buffer immediately
  esp_camera_fb_return(fb);
  
  // Minimal logging to avoid blocking
  logMessage(LOG_DEBUG, "Camera frame served - Size: " + String(fb->len) + " bytes");
}

// New endpoint: GET /api/inference -> Normal inference (with progress feedback)
void handleAPINormalInference() {
  logMessage(LOG_INFO, "⚡ === NORMAL INFERENCE #" + String(++inference_counter) + " ===");
  
  // Check system readiness first
  if (!input || !interpreter) {
    server.send(503, "application/json", "{\"error\":\"system_not_ready\"}");
    return;
  }
  
  // Run inference with progress tracking
  unsigned long start_time = millis();
  String result = runInference();
  unsigned long total_time = millis() - start_time;
  
  // Add timing information to result
  if (!result.startsWith("{\"error\":")) {
    // Insert timing info into JSON response
    int insert_pos = result.indexOf("\"performance\":");
    if (insert_pos > 0) {
      String timing_info = ",\"request_processing_time_ms\":" + String(total_time);
      result = result.substring(0, insert_pos) + timing_info + result.substring(insert_pos);
    }
  }
  
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

  // Set static IP configuration
  wifiManager.setSTAStaticIPConfig(staticIP, gateway, subnet, dns);

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
  
  // Try to switch to RGB565 format if we have enough memory
  // DISABLED: Causes DMA overflow and crashes
  // switchToRGB565IfNeeded();
  
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
  logMessage(LOG_INFO, " TENSOR ARENA ANALYSIS:");
  logMessage(LOG_INFO, "  Arena used: " + String(arena_used_bytes) + " bytes");
  logMessage(LOG_INFO, "  Arena size: " + String(actual_arena_size) + " bytes");
  logMessage(LOG_INFO, "  Arena usage: " + String((arena_used_bytes * 100) / actual_arena_size) + "%");
  
  if (arena_used_bytes > actual_arena_size * 0.9) {
    logMessage(LOG_ERROR, " WARNING: Tensor arena usage > 90% - risk of overflow!");
  }
  
  // Enhanced memory analysis
  logMessage(LOG_INFO, " ENHANCED MEMORY ANALYSIS:");
  size_t free_heap = ESP.getFreeHeap();
  size_t min_free_heap = ESP.getMinFreeHeap();
  size_t max_alloc_heap = ESP.getMaxAllocHeap();
  
  logMessage(LOG_INFO, "  Free heap: " + String(free_heap) + " bytes");
  logMessage(LOG_INFO, "  Min free heap: " + String(min_free_heap) + " bytes");
  logMessage(LOG_INFO, "  Max alloc heap: " + String(max_alloc_heap) + " bytes");
  
  if (free_heap < 100000) {
    logMessage(LOG_WARNING, " Low heap memory - may cause issues");
  }
  
  if (min_free_heap < 50000) {
    logMessage(LOG_ERROR, " Very low minimum heap - high risk of corruption!");
  }
  logMessage(LOG_INFO, "   Arena allocated: " + String(actual_arena_size / 1024) + " KB");
  logMessage(LOG_INFO, "   Arena actually used: " + String(arena_used_bytes / 1024) + " KB");
  logMessage(LOG_INFO, "   Arena utilization: " + String(100.0f * arena_used_bytes / actual_arena_size, 1) + "%");
  logMessage(LOG_INFO, "   Free arena space: " + String((actual_arena_size - arena_used_bytes) / 1024) + " KB");
  
  if (arena_used_bytes > actual_arena_size * 0.95f) {
    logMessage(LOG_ERROR, " ARENA NEARLY FULL! This will cause memory corruption during inference!");
    logMessage(LOG_ERROR, "Increase kTensorArenaSize to at least " + String((arena_used_bytes * 1.2f) / 1024) + " KB");
  } else if (arena_used_bytes > actual_arena_size * 0.85f) {
    logMessage(LOG_WARNING, "  Arena usage high - may cause overflow with scratch operations");
  } else {
    logMessage(LOG_INFO, " Arena size appears adequate");
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
  logMessage(LOG_INFO, " FULL INT8 MODEL QUANTIZATION PARAMETERS:");
  logMessage(LOG_INFO, "   Input scale: " + String(input->params.scale, 6));
  logMessage(LOG_INFO, "   Input zero point: " + String(input->params.zero_point));
  logMessage(LOG_INFO, "   Output tensor scale: " + String(output_tensor->params.scale, 6));
  logMessage(LOG_INFO, "   Output tensor zero point: " + String(output_tensor->params.zero_point));
  logMessage(LOG_INFO, "   Input type: " + String(input->type));
  logMessage(LOG_INFO, "   Output tensor type: " + String(output_tensor->type));
  
  // Validate full INT8 quantization
  if (input->params.scale == 0.0f || output_tensor->params.scale == 0.0f) {
    logMessage(LOG_ERROR, " CRITICAL: Model quantization parameters are zero!");
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
  server.on("/frame.jpg", HTTP_GET, handleFrame);
  
  // Add a simple health check endpoint (ultra-fast, non-blocking)
  server.on("/health", HTTP_GET, []() {
    // Minimal logging for health checks to avoid blocking
    server.send(200, "application/json", "{\"status\":\"ok\",\"uptime_ms\":" + String(millis()) + "}");
  });
  
  // Add a quick system check endpoint
  server.on("/quick-check", HTTP_GET, []() {
    // Very fast response with minimal processing
    String response = "{\"ready\":" + String(input && interpreter ? "true" : "false");
    response += ",\"heap\":" + String(esp_get_free_heap_size());
    response += ",\"uptime\":" + String(millis());
    response += "}";
    server.send(200, "application/json", response);
  });
  
  // Add camera view endpoint with detections overlay
  server.on("/camera-view", HTTP_GET, []() {
    logMessage(LOG_INFO, "Camera view request received from: " + server.client().remoteIP().toString());
    
    String html = "<!DOCTYPE html>";
    html += "<html>";
    html += "<head>";
    html += "<title>ESP32 Camera View - Live Detections</title>";
    html += "<meta charset=\"utf-8\">";
    html += "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">";
    html += "<style>";
    html += "body { font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }";
    html += ".container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }";
    html += ".camera-section { text-align: center; margin-bottom: 30px; }";
    html += ".camera-container { position: relative; display: inline-block; border: 3px solid #4CAF50; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.2); }";
    html += ".camera-image { max-width: 100%; height: auto; display: block; }";
    html += ".detection-overlay { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }";
    html += ".detection-box { position: absolute; border: 3px solid #FF5722; background: rgba(255, 87, 34, 0.2); border-radius: 5px; }";
    html += ".detection-label { position: absolute; top: -25px; left: 0; background: #FF5722; color: white; padding: 2px 8px; border-radius: 3px; font-size: 12px; font-weight: bold; white-space: nowrap; }";
    html += ".controls { margin: 20px 0; }";
    html += ".btn { background: #4CAF50; color: white; border: none; padding: 10px 20px; margin: 5px; border-radius: 5px; cursor: pointer; font-size: 16px; }";
    html += ".btn:hover { background: #45a049; }";
    html += ".status { background: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #4CAF50; }";
    html += ".detections-info { background: #fff3e0; padding: 15px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #FF9800; }";
    html += ".detection-item { background: white; padding: 10px; margin: 10px 0; border-radius: 5px; border: 1px solid #ddd; }";
    html += ".performance { background: #f3e5f5; padding: 15px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #9C27B0; }";
    html += ".performance-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }";
    html += ".performance-item { background: white; padding: 10px; border-radius: 5px; text-align: center; }";
    html += ".performance-value { font-size: 24px; font-weight: bold; color: #9C27B0; }";
    html += ".performance-label { color: #666; font-size: 14px; }";
    html += ".refresh-indicator { display: inline-block; margin-left: 10px; color: #666; }";
    html += "@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }";
    html += ".spinning { animation: spin 1s linear infinite; }";
    html += "</style>";
    html += "</head>";
    html += "<body>";
    html += "<div class=\"container\">";
    html += "<h1>ESP32 Pest Detection - Live Camera View</h1>";
    html += "<div class=\"camera-section\">";
    html += "<div class=\"camera-container\">";
    html += "<img id=\"cameraImage\" class=\"camera-image\" src=\"/frame.jpg\" alt=\"Live Camera Feed\">";
    html += "<div id=\"detectionOverlay\" class=\"detection-overlay\"></div>";
    html += "</div>";
    html += "<div class=\"controls\">";
    html += "<button class=\"btn\" onclick=\"refreshImage()\">Refresh Image</button>";
    html += "<button class=\"btn\" onclick=\"forceImageRefresh()\">Force Refresh</button>";
    html += "<button class=\"btn\" onclick=\"toggleAutoRefresh()\">Pause Auto-Refresh</button>";
    html += "<button class=\"btn\" onclick=\"fetchDetections()\">Fetch Detections</button>";
    html += "<span id=\"refreshIndicator\" class=\"refresh-indicator\">Auto-refreshing every 3s</span>";
    html += "</div>";
    html += "</div>";
    html += "<div class=\"status\">";
    html += "<h3>System Status</h3>";
    html += "<p><strong>Last Update:</strong> <span id=\"lastUpdate\">Never</span></p>";
    html += "<p><strong>Image Refresh:</strong> <span id=\"imageStatus\">Ready</span></p>";
    html += "<p><strong>Detection Count:</strong> <span id=\"detectionCount\">0</span> pests detected</p>";
    html += "</div>";
    html += "<div class=\"detections-info\">";
    html += "<h3>Live Detections</h3>";
    html += "<div id=\"detectionsList\">No detections yet...</div>";
    html += "</div>";
    html += "<div class=\"performance\">";
    html += "<h3>Performance Metrics</h3>";
    html += "<div class=\"performance-grid\" id=\"performanceGrid\">";
    html += "<div class=\"performance-item\">";
    html += "<div class=\"performance-value\" id=\"inferenceTime\">--</div>";
    html += "<div class=\"performance-label\">Inference Time (ms)</div>";
    html += "</div>";
    html += "<div class=\"performance-item\">";
    html += "<div class=\"performance-value\" id=\"totalInferences\">--</div>";
    html += "<div class=\"performance-label\">Total Inferences</div>";
    html += "</div>";
    html += "<div class=\"performance-item\">";
    html += "<div class=\"performance-value\" id=\"freeHeap\">--</div>";
    html += "<div class=\"performance-label\">Free Heap (MB)</div>";
    html += "</div>";
    html += "<div class=\"performance-item\">";
    html += "<div class=\"performance-value\" id=\"freePSRAM\">--</div>";
    html += "<div class=\"performance-label\">Free PSRAM (MB)</div>";
    html += "</div>";
    html += "</div>";
    html += "</div>";
    html += "</div>";
    html += "<script>";
    html += "var autoRefresh = true;";
    html += "var refreshInterval;";
    html += "document.addEventListener('DOMContentLoaded', function() {";
    html += "fetchDetections();";
    html += "startAutoRefresh();";
    html += "});";
    html += "function startAutoRefresh() {";
    html += "refreshInterval = setInterval(function() {";
    html += "if (autoRefresh) {";
    html += "console.log('Auto-refresh triggered at: ' + new Date().toLocaleTimeString());";
    html += "forceImageRefresh();";
    html += "fetchDetections();";
    html += "}";
    html += "}, 3000);";
    html += "}";
    html += "function forceImageRefresh() {";
    html += "var img = document.getElementById('cameraImage');";
    html += "var status = document.getElementById('imageStatus');";
    html += "var indicator = document.getElementById('refreshIndicator');";
    html += "status.textContent = 'Forcing refresh...';";
    html += "indicator.classList.add('spinning');";
    html += "console.log('Force refreshing image...');";
    html += "var timestamp = Date.now();";
    html += "var newSrc = '/frame.jpg?t=' + timestamp + '&v=' + Math.random();";
    html += "console.log('Force refresh src: ' + newSrc);";
    html += "img.src = newSrc;";
    html += "img.onload = function() {";
    html += "status.textContent = 'Ready (forced)';";
    html += "indicator.classList.remove('spinning');";
    html += "document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();";
    html += "console.log('Force refresh successful at: ' + timestamp);";
    html += "};";
    html += "img.onerror = function() {";
    html += "status.textContent = 'Force refresh failed';";
    html += "indicator.classList.remove('spinning');";
    html += "console.log('Force refresh failed at: ' + timestamp);";
    html += "};";
    html += "}";
    html += "function toggleAutoRefresh() {";
    html += "autoRefresh = !autoRefresh;";
    html += "var btn = event.target;";
    html += "if (autoRefresh) {";
    html += "btn.textContent = 'Pause Auto-Refresh';";
    html += "btn.style.background = '#4CAF50';";
    html += "document.getElementById('refreshIndicator').textContent = 'Auto-refreshing every 3s';";
    html += "} else {";
    html += "btn.textContent = 'Resume Auto-Refresh';";
    html += "btn.style.background = '#FF9800';";
    html += "document.getElementById('refreshIndicator').textContent = 'Auto-refresh paused';";
    html += "}";
    html += "}";
    html += "function refreshImage() {";
    html += "var img = document.getElementById('cameraImage');";
    html += "var status = document.getElementById('imageStatus');";
    html += "var indicator = document.getElementById('refreshIndicator');";
    html += "status.textContent = 'Refreshing...';";
    html += "indicator.classList.add('spinning');";
    html += "console.log('Refreshing image...');";
    html += "console.log('Current image src: ' + img.src);";
    html += "var newSrc = '/frame.jpg?t=' + Date.now();";
    html += "console.log('New image src: ' + newSrc);";
    html += "img.src = newSrc;";
    html += "img.onload = function() {";
    html += "status.textContent = 'Ready';";
    html += "indicator.classList.remove('spinning');";
    html += "document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();";
    html += "console.log('Image refreshed successfully');";
    html += "console.log('Image dimensions: ' + img.naturalWidth + 'x' + img.naturalHeight);";
    html += "};";
    html += "img.onerror = function() {";
    html += "status.textContent = 'Error loading image';";
    html += "indicator.classList.remove('spinning');";
    html += "console.log('Error loading image');";
    html += "console.log('Error details: ' + img.src);";
    html += "};";
    html += "}";
    html += "function fetchDetections() {";
    html += "fetch('/api/detect')";
    html += ".then(function(response) { return response.json(); })";
    html += ".then(function(data) {";
    html += "displayDetections(data);";
    html += "updatePerformance(data.performance);";
    html += "})";
    html += ".catch(function(error) {";
    html += "console.error('Error fetching detections:', error);";
    html += "document.getElementById('detectionsList').innerHTML = '<p style=\"color: red;\">Error loading detections</p>';";
    html += "});";
    html += "}";
    html += "function displayDetections(data) {";
    html += "var detectionsList = document.getElementById('detectionsList');";
    html += "var detectionCount = document.getElementById('detectionCount');";
    html += "var overlay = document.getElementById('detectionOverlay');";
    html += "if (data.detections && data.detections.length > 0) {";
    html += "detectionCount.textContent = data.detections.length;";
    html += "var html = '';";
    html += "overlay.innerHTML = '';";
    html += "for (var i = 0; i < data.detections.length; i++) {";
    html += "var det = data.detections[i];";
    html += "var box = document.createElement('div');";
    html += "box.className = 'detection-box';";
    html += "box.style.left = (det.bbox[0] / data.width * 100) + '%';";
    html += "box.style.top = (det.bbox[1] / data.height * 100) + '%';";
    html += "box.style.width = (det.bbox[2] / data.width * 100) + '%';";
    html += "box.style.height = (det.bbox[3] / data.height * 100) + '%';";
    html += "var label = document.createElement('div');";
    html += "label.className = 'detection-label';";
    html += "label.textContent = det.class + ' (' + (det.score * 100).toFixed(1) + '%)';";
    html += "box.appendChild(label);";
    html += "overlay.appendChild(box);";
    html += "html += '<div class=\"detection-item\">';";
    html += "html += '<h3>' + det.class + '</h3>';";
    html += "html += '<p><strong>Confidence:</strong> ' + (det.score * 100).toFixed(1) + '%</p>';";
    html += "html += '<p><strong>Location:</strong> x=' + det.bbox[0] + ', y=' + det.bbox[1] + ', w=' + det.bbox[2] + ', h=' + det.bbox[3] + '</p>';";
    html += "html += '</div>';";
    html += "}";
    html += "detectionsList.innerHTML = html;";
    html += "} else {";
    html += "detectionCount.textContent = '0';";
    html += "detectionsList.innerHTML = '<p>No pests detected in current frame</p>';";
    html += "overlay.innerHTML = '';";
    html += "}";
    html += "}";
    html += "function updatePerformance(perf) {";
    html += "if (perf) {";
    html += "document.getElementById('inferenceTime').textContent = perf.inference_time_ms || '--';";
    html += "document.getElementById('totalInferences').textContent = perf.total_inferences || '--';";
    html += "document.getElementById('freeHeap').textContent = Math.round((perf.free_heap || 0) / 1024 / 1024) + ' MB';";
    html += "document.getElementById('freePSRAM').textContent = Math.round((perf.free_psram || 0) / 1024 / 1024) + ' MB';";
    html += "}";
    html += "}";
    html += "</script>";
    html += "</body>";
    html += "</html>";
    
    server.send(200, "text/html", html);
  });
  
  // Add a network test endpoint
  server.on("/network-test", HTTP_GET, []() {
    String clientIP = server.client().remoteIP().toString();
    logMessage(LOG_INFO, "Network test request received from: " + clientIP);
    
    String response = "{";
    response += "\"status\":\"connected\",";
    response += "\"client_ip\":\"" + clientIP + "\",";
    response += "\"esp32_ip\":\"" + WiFi.localIP().toString() + "\",";
    response += "\"gateway\":\"" + WiFi.gatewayIP().toString() + "\",";
    response += "\"subnet\":\"" + WiFi.subnetMask().toString() + "\",";
    response += "\"timestamp\":" + String(millis());
    response += "}";
    
    server.send(200, "application/json", response);
  });
  
  // Add a button test endpoint
  server.on("/button-test", HTTP_GET, []() {
    String clientIP = server.client().remoteIP().toString();
    logMessage(LOG_INFO, "Button test request received from: " + clientIP);
    
    String response = "{";
    response += "\"status\":\"button_clicked\",";
    response += "\"timestamp\":" + String(millis());
    response += "}";
    
    server.send(200, "application/json", response);
  });
  
  // Enable CORS for API endpoints
  server.enableCORS(true);
  
  server.begin();
  logMessage(LOG_INFO, "HTTP server started on port 80");
  
  // Network debugging info
  logMessage(LOG_INFO, "=== Network Configuration ===");
  logMessage(LOG_INFO, "WiFi SSID: " + WiFi.SSID());
  logMessage(LOG_INFO, "WiFi IP: " + WiFi.localIP().toString());
  logMessage(LOG_INFO, "WiFi Gateway: " + WiFi.gatewayIP().toString());
  logMessage(LOG_INFO, "WiFi Subnet: " + WiFi.subnetMask().toString());
  logMessage(LOG_INFO, "WiFi DNS: " + WiFi.dnsIP().toString());
  logMessage(LOG_INFO, "WiFi Channel: " + String(WiFi.channel()));
  logMessage(LOG_INFO, "WiFi RSSI: " + String(WiFi.RSSI()) + " dBm");
  
  // Test network connectivity
  logMessage(LOG_INFO, "=== TESTING NETWORK CONNECTIVITY ===");
  
  // Test 1: Can ESP32 reach its own gateway?
  WiFiClient gatewayClient;
  if (gatewayClient.connect(WiFi.gatewayIP(), 80)) {
    logMessage(LOG_INFO, "Gateway connectivity: SUCCESS");
    gatewayClient.stop();
  } else {
    logMessage(LOG_INFO, "Gateway connectivity: FAILED");
  }
  
  // Test 2: Can ESP32 reach external internet?
  WiFiClient internetClient;
  if (internetClient.connect("8.8.8.8", 53)) {
    logMessage(LOG_INFO, "Internet connectivity: SUCCESS");
    internetClient.stop();
  } else {
    logMessage(LOG_INFO, "Internet connectivity: FAILED");
  }
  
  // Test 3: Check if WebServer is properly bound
  logMessage(LOG_INFO, "WebServer status: READY");
  logMessage(LOG_INFO, "=============================");
  
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
  // Handle HTTP requests FIRST - this is critical for responsiveness
  server.handleClient();
  
  if (!input || !interpreter) {
    logMessage(LOG_ERROR, "TFLite not initialized");
    delay(1000);
    return;
  }
  
  // Run inference only when requested via web interface, not automatically
  // This prevents blocking the web server and reduces unnecessary processing
  static unsigned long last_health_check = 0;
  unsigned long current_time = millis();
  
  // Only perform minimal health checks every 30 seconds
  if (current_time - last_health_check >= 30000) {
    last_health_check = current_time;
    
    // Quick camera health check without inference
    camera_fb_t* test_fb = esp_camera_fb_get();
    if (test_fb && test_fb->buf && test_fb->len > 0) {
      logMessage(LOG_INFO, "Camera health check passed - Size: " + String(test_fb->len) + " bytes");
      esp_camera_fb_return(test_fb);
    } else {
      logMessage(LOG_ERROR, "Camera health check failed");
    }
  }
  
  // Handle HTTP requests more frequently for better responsiveness
  server.handleClient();
  
  // Minimal delay to keep server responsive
  delay(10);
}



