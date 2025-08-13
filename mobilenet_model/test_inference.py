#!/usr/bin/env python3
"""
Insect Detection Model Inference Script
"""

import os
import cv2
import numpy as np
import tensorflow as tf

class InsectDetector:
    def __init__(self, model_path, conf_threshold=0.25):
        self.conf_threshold = conf_threshold
        
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']
        self.img_size = self.input_shape[1]
        
        # Get quantization parameters
        self.input_quantization = self.input_details[0].get('quantization', None)
        if self.input_quantization:
            self.input_scale = self.input_quantization[0]  # scales
            self.input_zero_point = self.input_quantization[1]  # zero_points
        else:
            self.input_scale = 1.0
            self.input_zero_point = 0
        self.input_dtype = self.input_details[0]['dtype']
        
        # Configuration
        self.S = self.img_size // 16
        self.A = 3
        self.ANCHORS = np.array([[0.10, 0.08], [0.18, 0.15], [0.28, 0.24]])
        
        # Class names - actual pest names from the model
        self.class_names = [
            "Agriotes fuscicollis Miwa",
            "Anomala corpulenta", 
            "Armyworm",
            "Athetis lepigone",
            "Bollworm",
            "Gryllotalpa orientalis",
            "Land tiger",
            "Little Gecko",
            "Meadow borer",
            "Melahotus",
            "Nematode trench",
            "Plutella xylostella",
            "Rice Leaf Roller",
            "Rice planthopper",
            "Scotogramma trifolii Rottemberg",
            "Spodoptera cabbage",
            "Spodoptera exigua",
            "Spodoptera litura",
            "Stem borer",
            "Striped rice bore",
            "Yellow tiger",
            "eight-character tiger",
            "holotrichia oblita",
            "holotrichia parallela"
        ]
        
        # Generate colors for visualization
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.class_names), 3), dtype=np.uint8)
        
        print(f"Model loaded: {self.img_size}x{self.img_size} input")
        print(f"Input dtype: {self.input_dtype}")
        print(f"Input shape: {self.input_shape}")
        print(f"Quantization: scale={self.input_scale}, zero_point={self.input_zero_point}")
        print(f"Output details: {self.output_details[0]}")
    
    def preprocess(self, image):
        # Resize image
        resized = cv2.resize(image, (self.img_size, self.img_size))
        
        if self.input_dtype == np.int8:
            # For INT8 quantized models, we need to quantize the input
            # Convert to float32 first, then quantize
            normalized = resized.astype(np.float32) / 255.0
            quantized = np.round(normalized / self.input_scale + self.input_zero_point)
            quantized = np.clip(quantized, -128, 127).astype(np.int8)
            return np.expand_dims(quantized, axis=0)
        else:
            # For float32 models
            normalized = resized.astype(np.float32) / 255.0
            return np.expand_dims(normalized, axis=0)
    
    def detect(self, image):
        try:
            input_tensor = self.preprocess(image)
            print(f"  Input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
            
            self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
            self.interpreter.invoke()
            predictions = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            print(f"  Output tensor shape: {predictions.shape}, dtype: {predictions.dtype}")
            
            # Decode YOLO predictions
            boxes, class_ids, scores = self.decode_predictions(predictions[0])
            
            return boxes, class_ids, scores
        except Exception as e:
            print(f"  Error during detection: {e}")
            return [], [], []
    
    def decode_predictions(self, pred):
        """Decode YOLO predictions to bounding boxes"""
        # Check if output is quantized
        output_quantization = self.output_details[0].get('quantization', None)
        if output_quantization and self.output_details[0]['dtype'] == np.int8:
            output_scale = output_quantization[0]  # scales
            output_zero_point = output_quantization[1]  # zero_points
            # Dequantize
            pred = (pred.astype(np.float32) - output_zero_point) * output_scale
        
        # Reshape to (S, S, A, 1+4+C)
        pred_shape = (self.S, self.S, self.A, 1 + 4 + len(self.class_names))
        pred = pred.reshape(pred_shape)
        
        # Extract components
        obj = tf.sigmoid(pred[..., 0])  # Objectness
        box = pred[..., 1:5]  # Box predictions
        cls = tf.sigmoid(pred[..., 5:])  # Class probabilities
        
        # Build grid
        gy, gx = tf.meshgrid(tf.range(self.S), tf.range(self.S), indexing="ij")
        gx = tf.cast(gx, tf.float32)
        gy = tf.cast(gy, tf.float32)
        
        # Expand grid dimensions
        gx = tf.expand_dims(gx, axis=-1)
        gy = tf.expand_dims(gy, axis=-1)
        
        # Recover boxes (normalized 0..1)
        cell_x = (box[..., 0] + gx) / float(self.S)
        cell_y = (box[..., 1] + gy) / float(self.S)
        
        # Apply anchors
        aw = tf.constant(self.ANCHORS[:, 0], dtype=tf.float32)
        ah = tf.constant(self.ANCHORS[:, 1], dtype=tf.float32)
        bw = tf.exp(box[..., 2]) * aw
        bh = tf.exp(box[..., 3]) * ah
        
        # Convert to corner format
        x1 = tf.clip_by_value(cell_x - bw/2, 0., 1.)
        y1 = tf.clip_by_value(cell_y - bh/2, 0., 1.)
        x2 = tf.clip_by_value(cell_x + bw/2, 0., 1.)
        y2 = tf.clip_by_value(cell_y + bh/2, 0., 1.)
        
        # Flatten
        boxes = tf.stack([y1, x1, y2, x2], axis=-1)
        boxes = tf.reshape(boxes, (-1, 4))
        obj = tf.reshape(obj, (-1,))
        cls = tf.reshape(cls, (-1, cls.shape[-1]))
        
        # Calculate scores
        scores = obj[:, None] * cls
        class_ids = tf.argmax(scores, axis=-1)
        class_scores = tf.reduce_max(scores, axis=-1)
        
        # Filter by confidence
        keep = class_scores > self.conf_threshold
        boxes = boxes[keep]
        class_ids = tf.cast(class_ids[keep], tf.int32)
        class_scores = class_scores[keep]
        
        # NMS
        selected = tf.image.non_max_suppression(
            boxes, class_scores, 
            max_output_size=100, 
            iou_threshold=0.45
        )
        
        return tf.gather(boxes, selected), tf.gather(class_ids, selected), tf.gather(class_scores, selected)
    
    def visualize(self, image, boxes, class_ids, scores, save_path=None):
        result = image.copy()
        h, w = image.shape[:2]
        
        for box, class_id, score in zip(boxes, class_ids, scores):
            if score > self.conf_threshold:
                # Convert normalized coordinates to pixel coordinates
                y1, x1, y2, x2 = box
                x1 = int(x1 * w)
                y1 = int(y1 * h)
                x2 = int(x2 * w)
                y2 = int(y2 * h)
                
                # Get class name
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                
                # Choose color based on class - OpenCV expects BGR format
                color = tuple(map(int, self.colors[class_id % len(self.colors)]))
                
                # Draw bounding box
                cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
                
                # Draw label background
                label = f"{class_name}: {score:.2f}"
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(result, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
                
                # Draw label text
                cv2.putText(result, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if save_path:
            cv2.imwrite(save_path, result)
        
        return result

def get_test_images():
    """Get existing PNG images from the test_img_inference folder"""
    test_folder = "test_img_inference"
    
    if not os.path.exists(test_folder):
        print(f"Test folder not found: {test_folder}")
        print("Please create the test_img_inference folder and add some PNG images")
        return []
    
    # Get all PNG files from the folder
    png_files = []
    for file in os.listdir(test_folder):
        if file.lower().endswith('.png'):
            png_files.append(os.path.join(test_folder, file))
    
    if not png_files:
        print(f"No PNG files found in {test_folder}")
        print("Please add some PNG images to test")
        return []
    
    print(f"Found {len(png_files)} PNG images in {test_folder}:")
    for file in png_files:
        print(f"  - {os.path.basename(file)}")
    
    return png_files

def main():
    print("Insect Detection Tester")
    
    # Check for model
    model_path = "mdet_int8_final.tflite"
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Please place your TFLite model in the current directory")
        return
    
    # Get existing test images
    test_images = get_test_images()
    
    if not test_images:
        print("No test images found. Exiting.")
        return
    
    # Initialize detector
    detector = InsectDetector(model_path)
    
    # Test on images
    for img_path in test_images:
            print(f"\nProcessing: {img_path}")
            image = cv2.imread(img_path)
            boxes, class_ids, scores = detector.detect(image)
            
            print(f"  Detections: {len(boxes)}")
            for i, (box, class_id, score) in enumerate(zip(boxes, class_ids, scores)):
                class_name = detector.class_names[class_id] if class_id < len(detector.class_names) else f"class_{class_id}"
                print(f"    {i+1}. {class_name}: {score:.3f}")
                print(f"       Box: {box} (normalized coordinates)")
                print(f"       Class ID: {class_id}")
            
            result = detector.visualize(image, boxes, class_ids, scores)
            
            # Save result with descriptive filename
            result_filename = f"results_{os.path.basename(img_path)}"
            cv2.imwrite(result_filename, result)
            print(f"  Result saved to: {result_filename}")
            
            # Also save a copy with detection info in filename
            if len(boxes) > 0:
                top_class = detector.class_names[class_ids[0]] if class_ids[0] < len(detector.class_names) else f"class_{class_ids[0]}"
                top_score = scores[0]
                info_filename = f"detected_{top_class}_{top_score:.2f}_{os.path.basename(img_path)}"
                cv2.imwrite(info_filename, result)
                print(f"  Info copy saved to: {info_filename}")

if __name__ == "__main__":
    main()
