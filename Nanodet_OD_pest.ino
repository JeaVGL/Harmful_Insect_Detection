#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include <WiFiManager.h>

#include "nanodet_intermediate_int8.h"             // CORRECTED model with proper ONNX export
#include "insect_class_names.h"                       // Class names mapping
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

// State of a tensor for debugging
struct TensorState {
  int8_t samples[10];
  int8_t min_val;
  int8_t max_val;
  int unique_count;
  uint32_t checksum;
};


// -------------------- Constants --------------------
constexpr int MAX_DETECTIONS = 100;
constexpr size_t kTensorArenaSize = 2048u * 1024u; // INCREASED to 2MB to prevent tensor arena overflow

// -------------------- Global Variables --------------------
PerformanceMetrics perf;
const tflite::Model*      model        = nullptr;
tflite::MicroInterpreter* interpreter  = nullptr;
TfLiteTensor*            input        = nullptr;
TfLiteTensor*            output       = nullptr;
uint8_t*                 tensor_arena  = nullptr;

// Global pattern tracking for debugging
static int current_test_pattern = -1;

// Debug mode flag
bool debug_mode = true;
unsigned int inference_counter = 0;

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

// -------------------- Computer Vision Functions --------------------

// Image preprocessing: Decode JPEG and convert to model input format (288x288 RGB)
bool preprocessFrame(camera_fb_t* fb, int8_t* input_buffer) {
  if (!fb || !fb->buf || !input_buffer) {
    logMessage(LOG_ERROR, "Invalid frame or input buffer");
    return false;
  }
  
  // Expected model input: 288x288x3 RGB
  const int MODEL_WIDTH = 288;
  const int MODEL_HEIGHT = 288;
  const int MODEL_CHANNELS = 3;
  
  // CORRECTED preprocessing based on training config analysis
  const float NANODET_MEAN_R = 103.53f;
  const float NANODET_MEAN_G = 116.28f;
  const float NANODET_MEAN_B = 123.675f;
  const float NANODET_STD_R = 57.375f;
  const float NANODET_STD_G = 57.12f;
  const float NANODET_STD_B = 58.395f;
  
  // Get input tensor quantization parameters
  float input_scale = input->params.scale;
  int input_zp = input->params.zero_point;
  
  current_test_pattern = (current_test_pattern + 1) % 4; // Test 4 different input patterns
  logMessage(LOG_INFO, "🎨 Using test pattern " + String(current_test_pattern));
  
  // Create temporary HWC buffer first
  uint8_t* temp_hwc_buffer = (uint8_t*)malloc(MODEL_WIDTH * MODEL_HEIGHT * MODEL_CHANNELS);
  if (!temp_hwc_buffer) {
    logMessage(LOG_ERROR, "Failed to allocate temporary HWC buffer");
    return false;
  }
  
  switch (current_test_pattern) {
    case 0: {
      // Pattern 1: VARIED GRADIENT - creates spatial variation for model to detect
      logMessage(LOG_INFO, "Creating VARIED gradient pattern (not uniform!)");
      for (int y = 0; y < MODEL_HEIGHT; y++) {
        for (int x = 0; x < MODEL_WIDTH; x++) {
          int pixel_idx = y * MODEL_WIDTH + x;
          int buffer_idx = pixel_idx * MODEL_CHANNELS;
          
          // Create spatial variation - gradient from dark to light
          float base_intensity = 80.0f + (170.0f * x / MODEL_WIDTH); // 80 to 250 across width
          float noise = (float)(random(-20, 21)); // Add random noise ±20
          
          float pixel_r = base_intensity + noise;
          float pixel_g = base_intensity + noise * 0.8f;
          float pixel_b = base_intensity + noise * 0.6f;
          
          // Clamp to valid range
          pixel_r = constrain(pixel_r, 0.0f, 255.0f);
          pixel_g = constrain(pixel_g, 0.0f, 255.0f);
          pixel_b = constrain(pixel_b, 0.0f, 255.0f);
          
          // Apply NanoDet normalization
          float normalized_r = (pixel_r - NANODET_MEAN_R) / NANODET_STD_R;
          float normalized_g = (pixel_g - NANODET_MEAN_G) / NANODET_STD_G;
          float normalized_b = (pixel_b - NANODET_MEAN_B) / NANODET_STD_B;
          
          // Quantize
          float input_scale = input->params.scale;
          int input_zp = input->params.zero_point;
          
          int quantized_r = (int)(normalized_r / input_scale) + input_zp;
          int quantized_g = (int)(normalized_g / input_scale) + input_zp;
          int quantized_b = (int)(normalized_b / input_scale) + input_zp;
          
          temp_hwc_buffer[buffer_idx + 0] = constrain(quantized_r, -128, 127);
          temp_hwc_buffer[buffer_idx + 1] = constrain(quantized_g, -128, 127);
          temp_hwc_buffer[buffer_idx + 2] = constrain(quantized_b, -128, 127);
        }
      }
      break;
    }
    
    case 1: {
      // Pattern 2: ENHANCED insect-like pattern with realistic background variation
      int center_x = MODEL_WIDTH / 2;
      int center_y = MODEL_HEIGHT / 2;
      int radius = 40; // Larger radius for better detection
      int insect_pixels = 0;
      
      logMessage(LOG_INFO, "Creating ENHANCED insect pattern with varied background");
      
      // Fill each pixel individually, checking if it's part of the insect blob
      for (int y = 0; y < MODEL_HEIGHT; y++) {
        for (int x = 0; x < MODEL_WIDTH; x++) {
          int pixel_idx = y * MODEL_WIDTH + x;
          int buffer_idx = pixel_idx * MODEL_CHANNELS;
          
          // Check if this pixel is inside the insect blob
          int dx = x - center_x;
          int dy = y - center_y;
          float distance = sqrt(dx*dx + dy*dy);
          bool is_insect = (distance < radius);
          
          float pixel_r, pixel_g, pixel_b;
          if (is_insect) {
            // Dark brown insect color with slight variation
            float variation = (float)(random(-10, 11)) / 10.0f; // ±1.0 variation
            pixel_r = 60.0f + variation * 20.0f;  // Dark brown range
            pixel_g = 40.0f + variation * 15.0f;
            pixel_b = 25.0f + variation * 10.0f;
            insect_pixels++;
          } else {
            // Varied background - simulate natural leaf/soil texture
            float base = 140.0f + 60.0f * sin(x * 0.1f) * cos(y * 0.1f); // Natural texture
            float noise = (float)(random(-30, 31)); // Significant texture variation
            pixel_r = base + noise + random(-10, 11);
            pixel_g = base + noise * 0.9f + random(-8, 9);
            pixel_b = base + noise * 0.7f + random(-6, 7);
          }
          
          // Clamp to valid range
          pixel_r = constrain(pixel_r, 20.0f, 240.0f);
          pixel_g = constrain(pixel_g, 20.0f, 240.0f);
          pixel_b = constrain(pixel_b, 20.0f, 240.0f);
          
          float normalized_r = (pixel_r - NANODET_MEAN_R) / NANODET_STD_R;
          float normalized_g = (pixel_g - NANODET_MEAN_G) / NANODET_STD_G;
          float normalized_b = (pixel_b - NANODET_MEAN_B) / NANODET_STD_B;
          
          float input_scale = input->params.scale;
          int input_zp = input->params.zero_point;
          
          int quantized_r = (int)(normalized_r / input_scale) + input_zp;
          int quantized_g = (int)(normalized_g / input_scale) + input_zp;
          int quantized_b = (int)(normalized_b / input_scale) + input_zp;
          
          // Store in HWC format temporarily
          temp_hwc_buffer[buffer_idx + 0] = constrain(quantized_r, -128, 127);
          temp_hwc_buffer[buffer_idx + 1] = constrain(quantized_g, -128, 127);
          temp_hwc_buffer[buffer_idx + 2] = constrain(quantized_b, -128, 127);
        }
      }
      logMessage(LOG_INFO, "Insect pattern created: " + String(insect_pixels) + " insect pixels");
      break;
    }
    
    case 2: {
      // Pattern 3: IMPROVED checkerboard with edge variation for feature detection
      logMessage(LOG_INFO, "Creating IMPROVED checkerboard with feature-rich edges");
      for (int y = 0; y < MODEL_HEIGHT; y++) {
        for (int x = 0; x < MODEL_WIDTH; x++) {
          int pixel_idx = y * MODEL_WIDTH + x;
          int buffer_idx = pixel_idx * MODEL_CHANNELS;
          
          // Smaller checkerboard squares for more features
          bool is_white = ((x / 16) + (y / 16)) % 2 == 0;
          
          // Add edge enhancement - brighter/darker at square boundaries
          bool is_edge = ((x % 16) < 2) || ((y % 16) < 2) || ((x % 16) > 13) || ((y % 16) > 13);
          
          float base_val = is_white ? 200.0f : 60.0f;
          float edge_modifier = is_edge ? (is_white ? 30.0f : -30.0f) : 0.0f;
          float noise = (float)(random(-10, 11));
          
          float pixel_r = base_val + edge_modifier + noise;
          float pixel_g = base_val + edge_modifier * 0.8f + noise * 0.9f;
          float pixel_b = base_val + edge_modifier * 0.6f + noise * 0.7f;
          
          // Clamp values
          pixel_r = constrain(pixel_r, 20.0f, 240.0f);
          pixel_g = constrain(pixel_g, 20.0f, 240.0f);
          pixel_b = constrain(pixel_b, 20.0f, 240.0f);
          
          float normalized_r = (pixel_r - NANODET_MEAN_R) / NANODET_STD_R;
          float normalized_g = (pixel_g - NANODET_MEAN_G) / NANODET_STD_G;
          float normalized_b = (pixel_b - NANODET_MEAN_B) / NANODET_STD_B;
          
          float input_scale = input->params.scale;
          int input_zp = input->params.zero_point;
          
          int quantized_r = (int)(normalized_r / input_scale) + input_zp;
          int quantized_g = (int)(normalized_g / input_scale) + input_zp;
          int quantized_b = (int)(normalized_b / input_scale) + input_zp;
          
          // Store in HWC format temporarily
          temp_hwc_buffer[buffer_idx + 0] = constrain(quantized_r, -128, 127);
          temp_hwc_buffer[buffer_idx + 1] = constrain(quantized_g, -128, 127);
          temp_hwc_buffer[buffer_idx + 2] = constrain(quantized_b, -128, 127);
        }
      }
      break;
    }
    
    case 3: {
      // Pattern 4: TEST - Raw values without normalization (to test if normalization is wrong)
      logMessage(LOG_INFO, "Pattern 4: RAW TEST - Direct quantization without normalization");
      float input_scale = input->params.scale;
      int input_zp = input->params.zero_point;
      
      for (int y = 0; y < MODEL_HEIGHT; y++) {
        for (int x = 0; x < MODEL_WIDTH; x++) {
          int pixel_idx = y * MODEL_WIDTH + x;
          int buffer_idx = pixel_idx * MODEL_CHANNELS;
          
          // Simple gradient pattern: raw RGB values 0-255
          float raw_r = (x % 256);
          float raw_g = (y % 256); 
          float raw_b = ((x + y) % 256);
          
          // Direct quantization: scale [0,255] to quantized range
          int quantized_r = (int)((raw_r / 255.0f - 0.5f) / input_scale) + input_zp;
          int quantized_g = (int)((raw_g / 255.0f - 0.5f) / input_scale) + input_zp;  
          int quantized_b = (int)((raw_b / 255.0f - 0.5f) / input_scale) + input_zp;
          
          temp_hwc_buffer[buffer_idx + 0] = constrain(quantized_r, -128, 127);
          temp_hwc_buffer[buffer_idx + 1] = constrain(quantized_g, -128, 127);
          temp_hwc_buffer[buffer_idx + 2] = constrain(quantized_b, -128, 127);
        }
      }
      break;
    }
  }
  
  // Convert from HWC to CHW format for model input
  // HWC: [R1,G1,B1, R2,G2,B2, ...]  →  CHW: [R1,R2,R3,..., G1,G2,G3,..., B1,B2,B3,...]
  int pixels_per_channel = MODEL_WIDTH * MODEL_HEIGHT;
  
  // Sample MULTIPLE DIFFERENT values to verify pattern variation
  logMessage(LOG_INFO, "Input quantization sample (checking pattern variation):");
  for (int sample = 0; sample < 5; sample++) {
    // Sample different areas: corners and center
    int sample_positions[] = {0, MODEL_WIDTH/4, MODEL_WIDTH/2, MODEL_WIDTH*3/4, MODEL_WIDTH-1};
    int idx = sample_positions[sample] * MODEL_WIDTH * MODEL_CHANNELS;
    if (idx < MODEL_WIDTH * MODEL_HEIGHT * MODEL_CHANNELS) {
      logMessage(LOG_INFO, "HWC[pos=" + String(sample_positions[sample]) + "]: R=" + String((int)temp_hwc_buffer[idx]) + 
                 " G=" + String((int)temp_hwc_buffer[idx + 1]) + 
                 " B=" + String((int)temp_hwc_buffer[idx + 2]));
    }
  }
  
  // R channel: copy all R values to start of buffer
  for (int i = 0; i < pixels_per_channel; i++) {
    input_buffer[i] = temp_hwc_buffer[i * 3 + 0];  // R channel
  }
  
  // G channel: copy all G values to middle of buffer
  for (int i = 0; i < pixels_per_channel; i++) {
    input_buffer[pixels_per_channel + i] = temp_hwc_buffer[i * 3 + 1];  // G channel
  }
  
  // B channel: copy all B values to end of buffer
  for (int i = 0; i < pixels_per_channel; i++) {
    input_buffer[pixels_per_channel * 2 + i] = temp_hwc_buffer[i * 3 + 2];  // B channel
  }
  
  // Clean up temporary buffer
  free(temp_hwc_buffer);
  
  // Debug: Sample CHW values to verify conversion
  logMessage(LOG_INFO, "CHW tensor sample:");
  int8_t* input_data = (int8_t*)input_buffer;
  for (int ch = 0; ch < 3; ch++) {
    int base_idx = ch * pixels_per_channel;
    logMessage(LOG_INFO, "Channel " + String(ch) + "[0-2]: " + String((int)input_data[base_idx]) + 
               ", " + String((int)input_data[base_idx + 1]) + 
               ", " + String((int)input_data[base_idx + 2]));
  }
  
  // DEBUG: Save processed input image for verification
  saveInputImageDebug(input_buffer, MODEL_WIDTH, MODEL_HEIGHT);
  
  // CRITICAL: Check if input values are in expected range for this model
  logMessage(LOG_INFO, "🔍 INPUT VALIDATION:");
  logMessage(LOG_INFO, "   Sample input values: [" + String((int8_t)input_buffer[0]) + "," + 
             String((int8_t)input_buffer[1000]) + "," + String((int8_t)input_buffer[50000]) + "]");
  logMessage(LOG_INFO, "   Input range check: Expected INT8 (-128 to 127), Scale=" + String(input_scale, 6) + ", ZP=" + String(input_zp));
  
  // Check if all values are exactly zero_point (which would be problematic)
  int zp_count = 0;
  int min_val = 127, max_val = -128;
  for (int i = 0; i < 1000; i += 100) {
    int8_t val = (int8_t)input_buffer[i];
    if (val == input_zp) zp_count++;
    if (val < min_val) min_val = val;
    if (val > max_val) max_val = val;
  }
  logMessage(LOG_INFO, "   Input statistics: min=" + String(min_val) + ", max=" + String(max_val) + ", zp_count=" + String(zp_count) + "/10");
  
  if (zp_count > 8) {
    logMessage(LOG_ERROR, "⚠️  Too many input values equal zero_point (" + String(zp_count) + "/10) - input may be incorrectly formatted!");
  }
  
  // ENHANCED: Test if the model responds to extreme input values
  if (current_test_pattern == 0) {
    logMessage(LOG_INFO, "🧪 TESTING: Injecting extreme test values to check model response...");
    // Temporarily set a few input values to extreme ranges to test model sensitivity
    input_buffer[0] = -128;  // Minimum INT8
    input_buffer[1] = 127;   // Maximum INT8  
    input_buffer[2] = input_zp;  // Zero point
    input_buffer[10000] = -100;  // Other extreme
    input_buffer[20000] = 100;   // Other extreme
  }
  
  return true;
}

// DEBUG: Save the processed input image as PPM format for verification
void saveInputImageDebug(int8_t* input_buffer, int width, int height) {
  static int image_counter = 0;
  image_counter++;
  
  // Only log every 10th image to avoid spam
  if (image_counter % 10 == 1) {
    logMessage(LOG_INFO, "Input image #" + String(image_counter) + ": " + String(width) + "x" + String(height) + " CHW format");
  }
}

// Helper: IoU for NMS with bounds checking
float iou(const Detection &a, const Detection &b) {
  // Validate bounding boxes
  if (a.x1 >= a.x2 || a.y1 >= a.y2 || b.x1 >= b.x2 || b.y1 >= b.y2) {
    logMessage(LOG_WARNING, "Invalid bounding box detected");
    return 0.0f;
  }
  
  float ix1 = fmaxf(a.x1, b.x1);
  float iy1 = fmaxf(a.y1, b.y1);
  float ix2 = fminf(a.x2, b.x2);
  float iy2 = fminf(a.y2, b.y2);
  float iw = fmaxf(ix2 - ix1, 0.0f);
  float ih = fmaxf(iy2 - iy1, 0.0f);
  float inter = iw * ih;
  float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  
  if (areaA + areaB - inter <= 0) return 0.0f;
  return inter / (areaA + areaB - inter);
}

// Enhanced NMS with better error handling
void nms(Detection *dets, int &count, float thresh) {
  if (!dets || count <= 0) {
    logMessage(LOG_WARNING, "Invalid input to NMS");
    return;
  }
  
  // Sort by score (selection sort)
  for (int i = 0; i < count - 1; i++) {
    int best = i;
    for (int j = i + 1; j < count; j++) {
      if (dets[j].score > dets[best].score) best = j;
    }
    if (best != i) {
      Detection tmp = dets[i];
      dets[i] = dets[best];
      dets[best] = tmp;
    }
  }
  
  // Apply NMS
  for (int i = 0; i < count; i++) {
    if (dets[i].score < 0) continue;
    for (int j = i + 1; j < count; j++) {
      if (dets[j].score < 0) continue;
      if (iou(dets[i], dets[j]) > thresh) {
        dets[j].score = -1;
      }
    }
  }
  
  // Compact array
  int w = 0;
  for (int i = 0; i < count; i++) {
    if (dets[i].score >= 0) dets[w++] = dets[i];
  }
  count = w;
  
  logMessage(LOG_DEBUG, "NMS completed, kept " + String(count) + " detections");
}

// -------------------- Inference Engine --------------------

// Enhanced inference with comprehensive error handling and performance metrics
String runInference() {
  unsigned long startTime = millis();
  
  // Input validation
  if (!input || !output || !interpreter) {
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
  
  // Preprocess frame to match model input (288x288 RGB)
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
  
  // Run inference
  logMessage(LOG_INFO, "🔄 Running inference with pattern " + String(current_test_pattern) + "...");
  
  // CRITICAL TEST: Comprehensive pre/post inference analysis
  int8_t pre_samples[5];
  int sample_positions[] = {0, 1000, 50000, 100000, 200000};
  for (int i = 0; i < 5; i++) {
    if (sample_positions[i] < expected_input_size) {
      pre_samples[i] = input->data.int8[sample_positions[i]];
    }
  }
  
  logMessage(LOG_INFO, "🔬 PRE-INFERENCE INPUT ANALYSIS:");
  String pre_analysis = "   Input samples at [0,1k,50k,100k,200k]: [";
  for (int i = 0; i < 5; i++) {
    pre_analysis += String((int)pre_samples[i]);
    if (i < 4) pre_analysis += ",";
  }
  pre_analysis += "]";
  logMessage(LOG_INFO, pre_analysis);
  
  // Check input variation
  int8_t min_input = 127, max_input = -128;
  int unique_values = 0;
  bool seen_values[256] = {false};
  
  for (int i = 0; i < min(10000, (int)expected_input_size); i++) {
    int8_t val = input->data.int8[i];
    if (val < min_input) min_input = val;
    if (val > max_input) max_input = val;
    int idx = val + 128; // Convert to 0-255 range for array index
    if (!seen_values[idx]) {
      seen_values[idx] = true;
      unique_values++;
    }
  }
  
  logMessage(LOG_INFO, "   Input range: " + String(min_input) + " to " + String(max_input) + ", unique values: " + String(unique_values));
  
  if (unique_values < 5) {
    logMessage(LOG_ERROR, "🚨 INPUT TOO UNIFORM! Only " + String(unique_values) + " unique values in 10k samples!");
    logMessage(LOG_ERROR, "This means the gradient pattern is not working correctly.");
  }
  
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    logMessage(LOG_ERROR, "Inference failed with status: " + String(invoke_status));
    return "{\"error\":\"inference_failed\"}";
  }
  
  // POST-INFERENCE ANALYSIS
  logMessage(LOG_INFO, "🔬 POST-INFERENCE INPUT ANALYSIS:");
  String post_analysis = "   Input samples at [0,1k,50k,100k,200k]: [";
  int changed_count = 0;
  for (int i = 0; i < 5; i++) {
    if (sample_positions[i] < expected_input_size) {
      int8_t post_val = input->data.int8[sample_positions[i]];
      post_analysis += String((int)post_val);
      if (post_val != pre_samples[i]) changed_count++;
      if (i < 4) post_analysis += ",";
    }
  }
  post_analysis += "] (changed: " + String(changed_count) + "/5)";
  logMessage(LOG_INFO, post_analysis);
  
  if (changed_count > 0) {
    logMessage(LOG_ERROR, "� CRITICAL: Input tensor modified during inference!");
    logMessage(LOG_ERROR, "This indicates the model is corrupted or incompatible with TFLite Micro!");
    logMessage(LOG_ERROR, "Possible solutions:");
    logMessage(LOG_ERROR, "  1. Export as FLOAT32 model instead of INT8");
    logMessage(LOG_ERROR, "  2. Use a different quantization method");
    logMessage(LOG_ERROR, "  3. Model file corruption during conversion");
  }
  
  // Validate output tensor
  if (!output->data.int8) {
    logMessage(LOG_ERROR, "Output tensor data is null");
    return "{\"error\":\"output_tensor_null\"}";
  }
  
  int num_det  = output->dims->data[1];
  int det_size = output->dims->data[2];
  
  // Validate output format
  if (det_size < 6) {
    logMessage(LOG_ERROR, "Unexpected detection format, det_size=" + String(det_size));
    return "{\"error\":\"invalid_output_format\"}";
  }
  
  int8_t* raw  = output->data.int8;
  float scale  = output->params.scale;
  int zp       = output->params.zero_point;

  Detection dets[MAX_DETECTIONS];
  int det_count = 0;
  const float score_thresh = 0.01f;  // LOWERED from 0.5 to 0.01 for testing

  // ENHANCED DEBUG: Track score statistics
  float min_score = 1.0f, max_score = -1.0f;
  int scores_above_zero = 0;
  int valid_detections_found = 0;

  for (int i = 0; i < num_det && det_count < MAX_DETECTIONS; i++) {
    int base = i * det_size;
    float score = (raw[base + 1] - zp) * scale;
    
    // Track score statistics
    if (score > max_score) max_score = score;
    if (score < min_score) min_score = score;
    if (score > 0.001f) scores_above_zero++;
    
    if (score < score_thresh) continue;
    valid_detections_found++;
    
    int   cls = raw[base + 0] - zp;
    
    // Validate class ID
    if (cls < 0 || cls >= N_CLASSES) {
      logMessage(LOG_WARNING, "Invalid class ID: " + String(cls));
      continue;
    }
    
    float cx  = (raw[base + 2] - zp) * scale;
    float cy  = (raw[base + 3] - zp) * scale;
    float w   = (raw[base + 4] - zp) * scale;
    float h   = (raw[base + 5] - zp) * scale;
    
    // Validate bounding box
    if (w <= 0 || h <= 0) {
      logMessage(LOG_WARNING, "Invalid bounding box dimensions");
      continue;
    }
    
    dets[det_count++] = {
      cls, score,
      cx - w/2, cy - h/2,
      cx + w/2, cy + h/2
    };
    
    if (det_count >= MAX_DETECTIONS) {
      logMessage(LOG_WARNING, "Maximum detections reached");
      break;
    }
  }

  logMessage(LOG_INFO, "📊 SCORE STATISTICS:");
  logMessage(LOG_INFO, "   Score range: " + String(min_score, 6) + " to " + String(max_score, 6));
  logMessage(LOG_INFO, "   Scores > 0.001: " + String(scores_above_zero) + "/" + String(num_det));
  logMessage(LOG_INFO, "   Valid detections found: " + String(valid_detections_found));
  logMessage(LOG_INFO, "Collected " + String(det_count) + " valid candidates");
  
  // Apply NMS
  nms(dets, det_count, 0.45f);
  logMessage(LOG_INFO, "After NMS: " + String(det_count) + " detections");
  
  // Additional debugging for zero detections
  if (det_count == 0) {
    logMessage(LOG_INFO, "🔍 ZERO DETECTIONS DEBUG:");
    logMessage(LOG_INFO, "   • Score threshold: " + String(score_thresh));
    logMessage(LOG_INFO, "   • Raw detections processed: " + String(num_det));
    logMessage(LOG_INFO, "   • Output tensor scale: " + String(scale, 6));
    logMessage(LOG_INFO, "   • Output tensor zero_point: " + String(zp));
    
    // ENHANCED: Check if ALL outputs are exactly zero_point
    int all_zp_count = 0;
    int non_zp_count = 0;
    int8_t min_raw = 127, max_raw = -128;
    
    for (int i = 0; i < min(100, num_det); i++) {
      int base = i * det_size;
      int8_t raw_score = raw[base + 1];
      if (raw_score == zp) {
        all_zp_count++;
      } else {
        non_zp_count++;
      }
      if (raw_score < min_raw) min_raw = raw_score;
      if (raw_score > max_raw) max_raw = raw_score;
    }
    
    logMessage(LOG_INFO, "   • Output analysis (first 100): all_zp=" + String(all_zp_count) + 
               ", non_zp=" + String(non_zp_count) + ", raw_range=[" + String(min_raw) + "," + String(max_raw) + "]");
    
    if (all_zp_count == min(100, num_det)) {
      logMessage(LOG_ERROR, "🚨 CRITICAL: ALL raw outputs equal zero_point! Model is not responding to input!");
      logMessage(LOG_ERROR, "This indicates either:");
      logMessage(LOG_ERROR, "  1. Input preprocessing is fundamentally wrong");
      logMessage(LOG_ERROR, "  2. Model quantization parameters are incorrect");
      logMessage(LOG_ERROR, "  3. Model file is corrupted or incompatible");
    }
    
    // Sample first few raw outputs
    logMessage(LOG_INFO, "   • Raw output samples (first 5):");
    for (int j = 0; j < min(5, num_det); j++) {
      int base = j * det_size;
      int8_t raw_cls = raw[base + 0];
      int8_t raw_score = raw[base + 1];
      float score = (raw_score - zp) * scale;
      int cls = raw_cls - zp;
      logMessage(LOG_INFO, "     [" + String(j) + "] cls=" + String(cls) + " raw_score=" + String(raw_score) + " score=" + String(score, 4));
    }
  }

  // Update performance metrics
  unsigned long inferenceTime = millis() - startTime;
  perf.lastInferenceTime = inferenceTime;
  perf.totalInferences++;
  perf.avgInferenceTime = ((perf.avgInferenceTime * (perf.totalInferences - 1)) + inferenceTime) / perf.totalInferences;
  
  logMessage(LOG_INFO, "Inference completed in " + String(inferenceTime) + "ms");

  // Print detection results to Serial Monitor for Arduino IDE
  Serial.println("\n=====================================");
  Serial.println("🐛 PEST DETECTION RESULTS");
  Serial.println("=====================================");
  
  if (det_count > 0) {
    Serial.println("✅ DETECTIONS FOUND: " + String(det_count));
    Serial.println("-------------------------------------");
    
    for (int i = 0; i < det_count; i++) {
      const auto &d = dets[i];
      Serial.println("Detection #" + String(i + 1) + ":");
      Serial.println("  🐛 Species: " + String(insect_names[d.class_id]) + " (ID: " + String(d.class_id) + ")");
      Serial.println("  📊 Confidence: " + String(d.score * 100, 1) + "%");
      Serial.println("  📍 Bounding Box:");
      Serial.println("     Top-Left: (" + String(d.x1, 1) + ", " + String(d.y1, 1) + ")");
      Serial.println("     Bottom-Right: (" + String(d.x2, 1) + ", " + String(d.y2, 1) + ")");
      Serial.println("     Size: " + String(d.x2 - d.x1, 1) + " x " + String(d.y2 - d.y1, 1));
      if (i < det_count - 1) Serial.println("  ---");
    }
  } else {
    Serial.println("❌ NO PESTS DETECTED");
    Serial.println("All clear! No insects found in this frame.");
  }
  
  Serial.println("-------------------------------------");
  Serial.println("⚡ PERFORMANCE METRICS:");
  Serial.println("  ⏱️  Inference Time: " + String(inferenceTime) + " ms");
  Serial.println("  📈 Average Time: " + String(perf.avgInferenceTime, 1) + " ms");
  Serial.println("  🔢 Total Inferences: " + String(perf.totalInferences));
  Serial.println("  💾 Free Heap: " + String(esp_get_free_heap_size() / 1024) + " KB");
  Serial.println("  🧠 Free PSRAM: " + String(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024) + " KB");
  Serial.println("=====================================\n");

  // Build enhanced JSON with class names
  String json = "{\"detections\":[";
  for (int i = 0; i < det_count; i++) {
    const auto &d = dets[i];
    json += "{\"class_id\":" + String(d.class_id);
    json += ",\"class_name\":\"" + String(insect_names[d.class_id]) + "\"";
    json += ",\"score\":" + String(d.score, 3);
    json += ",\"bbox\":{";
    json += "\"x1\":" + String(d.x1, 1);
    json += ",\"y1\":" + String(d.y1, 1);
    json += ",\"x2\":" + String(d.x2, 1);
    json += ",\"y2\":" + String(d.y2, 1);
    json += "}}";
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
                <button class='btn' onclick='runNewInference()'>🔄 Run Normal Detection</button>
                <button class='btn' onclick='runDebugInference()'>🔬 Run Debug Detection</button>
                <button class='btn' onclick='toggleDebugMode()'>🐛 Toggle Debug Mode</button>
                <button class='btn' onclick='toggleAutoRefresh()'>⏱️ Auto Refresh</button>
                <button class='btn' onclick='downloadResults()'>💾 Download Results</button>
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
        let autoRefresh = false;
        let refreshInterval;
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
        
        function runDebugInference() {
            document.getElementById('detectionResults').innerHTML = '<p>🔬 Running debug inference...</p>';
            fetch('/api/debug')
                .then(response => response.json())
                .then(data => displayResults(data))
                .catch(error => {
                    document.getElementById('detectionResults').innerHTML = '<div class="error">❌ Debug Error: ' + error + '</div>';
                });
        }
        
        function toggleDebugMode() {
            fetch('/api/toggle-debug')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('systemStatus').innerHTML = '<p>🔄 ' + data.message + '</p>';
                })
                .catch(error => {
                    document.getElementById('systemStatus').innerHTML = '<div class="error">❌ Error: ' + error + '</div>';
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
  json += "\"input_size\":\"288x288\"";
  json += ",\"classes\":" + String(N_CLASSES);
  json += ",\"tensor_arena_size\":" + String(kTensorArenaSize);
  json += ",\"model_type\":\"NanoDet-Plus-m_416\"";
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
    json += "{\"id\":" + String(i) + ",\"name\":\"" + String(insect_names[i]) + "\"}";
    if (i < N_CLASSES - 1) json += ",";
  }
  json += "]}";
  server.send(200, "application/json", json);
}

// New endpoint: GET /api/debug/inference -> Layer-by-layer debug inference
void handleAPIDebugInference() {
  logMessage(LOG_INFO, "🔬 === DEBUG INFERENCE #" + String(++inference_counter) + " ===");
  String result = runLayerByLayerInference();
  server.send(200, "application/json", result);
}

// New endpoint: GET /api/inference -> Normal inference
void handleAPINormalInference() {
  logMessage(LOG_INFO, "⚡ === NORMAL INFERENCE #" + String(++inference_counter) + " ===");
  String result = runInference();
  server.send(200, "application/json", result);
}

// New endpoint: POST /api/toggle_debug -> Toggle debug mode
void handleAPIToggleDebug() {
  debug_mode = !debug_mode;
  String status = debug_mode ? "enabled" : "disabled";
  logMessage(LOG_INFO, "🔄 Debug mode " + status);
  String json = "{\"debug_mode\":" + String(debug_mode ? "true" : "false") + ",\"message\":\"Debug mode " + status + "\"}";
  server.send(200, "application/json", json);
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
  
  // Need space for: TensorFlow (2048KB) + Camera frame (~50KB for QVGA JPEG) + System buffer (200KB)
  size_t totalPsramNeeded = kTensorArenaSize + 50000 + 200000; // ~2300KB total
  if (availablePsram < totalPsramNeeded) {
    logMessage(LOG_ERROR, "Insufficient PSRAM for tensor arena and camera frame buffer");
    logMessage(LOG_ERROR, "Required: " + String(totalPsramNeeded / 1024) + " KB, Available: " + String(availablePsram / 1024) + " KB");
    return;
  }
  
  tensor_arena = (uint8_t*) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM);
  if (!tensor_arena) {
    logMessage(LOG_ERROR, "Failed to allocate tensor arena!");
    logMessage(LOG_ERROR, "Available PSRAM: " + String(heap_caps_get_free_size(MALLOC_CAP_SPIRAM)) + " bytes");
    return;
  }
  logMessage(LOG_INFO, "Tensor arena allocated successfully");
  
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
  model = tflite::GetModel(nanodet_intermediate_int8);
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
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  
  size_t postInterpreterDram = heap_caps_get_free_size(MALLOC_CAP_8BIT) - heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
  logMessage(LOG_INFO, "DRAM after interpreter creation: " + String(postInterpreterDram) + " bytes (change: " + String((int)postInterpreterDram - (int)postModelDram) + ")");
  
    logMessage(LOG_INFO, "Allocating tensors...");
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    logMessage(LOG_ERROR, "AllocateTensors() failed");
    return;
  }
  
  // CRITICAL: Check actual arena usage to diagnose overflow
  size_t arena_used_bytes = interpreter->arena_used_bytes();
  logMessage(LOG_INFO, "🔍 TENSOR ARENA ANALYSIS:");
  logMessage(LOG_INFO, "   Arena allocated: " + String(kTensorArenaSize / 1024) + " KB");
  logMessage(LOG_INFO, "   Arena actually used: " + String(arena_used_bytes / 1024) + " KB");
  logMessage(LOG_INFO, "   Arena utilization: " + String(100.0f * arena_used_bytes / kTensorArenaSize, 1) + "%");
  logMessage(LOG_INFO, "   Free arena space: " + String((kTensorArenaSize - arena_used_bytes) / 1024) + " KB");
  
  if (arena_used_bytes > kTensorArenaSize * 0.95f) {
    logMessage(LOG_ERROR, "🚨 ARENA NEARLY FULL! This will cause memory corruption during inference!");
    logMessage(LOG_ERROR, "Increase kTensorArenaSize to at least " + String((arena_used_bytes * 1.2f) / 1024) + " KB");
  } else if (arena_used_bytes > kTensorArenaSize * 0.85f) {
    logMessage(LOG_WARNING, "⚠️  Arena usage high - may cause overflow with scratch operations");
  } else {
    logMessage(LOG_INFO, "✅ Arena size appears adequate");
  }
  
  size_t postAllocateDram = heap_caps_get_free_size(MALLOC_CAP_8BIT) - heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
  logMessage(LOG_INFO, "DRAM after tensor allocation: " + String(postAllocateDram) + " bytes (change: " + String((int)postAllocateDram - (int)postInterpreterDram) + ")");
  logMessage(LOG_INFO, "Total DRAM consumed by TensorFlow: " + String((int)preModelDram - (int)postAllocateDram) + " bytes");
  
  input  = interpreter->input(0);
  output = interpreter->output(0);
  
  if (!input || !output) {
    logMessage(LOG_ERROR, "Failed to get input/output tensors");
    return;
  }
  
  // Log tensor information
  logMessage(LOG_INFO, "Input tensor: " + String(input->bytes) + " bytes");
  logMessage(LOG_INFO, "Output tensor: " + String(output->bytes) + " bytes");
  logMessage(LOG_INFO, "Input dimensions: [" + String(input->dims->data[0]) + "," + 
             String(input->dims->data[1]) + "," + String(input->dims->data[2]) + "," + 
             String(input->dims->data[3]) + "]");
  logMessage(LOG_INFO, "Output dimensions: [" + String(output->dims->data[0]) + "," + 
             String(output->dims->data[1]) + "," + String(output->dims->data[2]) + "]");

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
  server.on("/api/debug/inference", HTTP_GET, handleAPIDebugInference);
  server.on("/api/inference", HTTP_GET, handleAPINormalInference);
  server.on("/api/toggle_debug", HTTP_POST, handleAPIToggleDebug);
  
  // Enable CORS for API endpoints
  server.enableCORS(true);
  
  server.begin();
  logMessage(LOG_INFO, "HTTP server started on port 80");
  
  // Final system status
  logMessage(LOG_INFO, "=== System Ready ===");
  logMessage(LOG_INFO, "Available classes: " + String(N_CLASSES));
  logMessage(LOG_INFO, "CPU frequency: " + String(getCpuFrequencyMhz()) + " MHz");
  logMessage(LOG_INFO, "XTAL frequency: " + String(getXtalFrequencyMhz()) + " MHz");
  logMessage(LOG_INFO, "APB frequency: " + String(getApbFrequency() / 1000000) + " MHz");
  logMessage(LOG_INFO, "Free heap: " + String(esp_get_free_heap_size()) + " bytes");
  logMessage(LOG_INFO, "Free PSRAM: " + String(heap_caps_get_free_size(MALLOC_CAP_SPIRAM)) + " bytes");
  logMessage(LOG_INFO, "Access web interface at: http://" + WiFi.localIP().toString());
}

void loop() {
  // Layer-by-layer debugging: print tensor state before and after inference
  if (!input || !output || !interpreter) {
    Serial.println("[ERROR] TFLite not initialized");
    delay(1000);
    return;
  }
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("[ERROR] Frame capture failed");
    delay(1000);
    return;
  }
  if (!preprocessFrame(fb, (int8_t*)input->data.int8)) {
    esp_camera_fb_return(fb);
    Serial.println("[ERROR] Preprocessing failed");
    delay(1000);
    return;
  }
  esp_camera_fb_return(fb);

  Serial.println("[DEBUG] Starting layer-by-layer tensor validation...");
  TensorState before = captureTensorState(input);
  printTensorState(before, "Input", "BEFORE");
  TfLiteStatus status = interpreter->Invoke();
  if (status != kTfLiteOk) {
    Serial.println("[ERROR] Inference failed");
    delay(1000);
    return;
  }
  TensorState after = captureTensorState(input);
  printTensorState(after, "Input", "AFTER");
  bool corrupted = false;
  for (int i = 0; i < 10; i++) {
    if (before.samples[i] != after.samples[i]) corrupted = true;
  }
  if (before.checksum != after.checksum) corrupted = true;
  if (corrupted) {
    Serial.println("[ERROR] INPUT TENSOR CORRUPTED DURING INFERENCE!");
  } else {
    Serial.println("[DEBUG] Input tensor remained intact during inference.");
  }
  delay(1000); // Run every second for testing
}
