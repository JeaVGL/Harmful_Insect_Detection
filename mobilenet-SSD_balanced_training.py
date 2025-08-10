#!/usr/bin/env python3
"""
Train MobileNet-SSD model for ESP32-S3 object detection
Optimized for dataset imbalance and lighter architecture
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import xml.etree.ElementTree as ET
from PIL import Image
import glob
import random
import json
from collections import Counter
import math

def parse_voc_xml(xml_path):
    """
    Parse Pascal VOC XML annotation file
    Returns: list of (class_id, x_min, y_min, x_max, y_max)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    annotations = []
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        x_min = float(bndbox.find('xmin').text) / width
        y_min = float(bndbox.find('ymin').text) / height
        x_max = float(bndbox.find('xmax').text) / width
        y_max = float(bndbox.find('ymax').text) / height
        
        # Convert to center coordinates and width/height
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        w = x_max - x_min
        h = y_max - y_min
        
        annotations.append((name, x_center, y_center, w, h))
    
    return annotations, width, height

def create_class_mapping(annotation_dir):
    """
    Create a mapping from class names to class IDs
    """
    class_names = set()
    
    # Find all XML files
    xml_files = glob.glob(os.path.join(annotation_dir, "*.xml"))
    
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        for obj in root.findall('object'):
            name = obj.find('name').text
            class_names.add(name)
    
    class_names = sorted(list(class_names))
    class_to_id = {name: i for i, name in enumerate(class_names)}
    id_to_class = {i: name for i, name in enumerate(class_names)}
    
    print(f"📋 Found {len(class_names)} classes: {class_names}")
    return class_to_id, id_to_class

def analyze_dataset_balance(dataset_paths, class_to_id):
    """
    Analyze dataset balance and calculate class weights
    """
    class_counts = Counter()
    
    for sample in dataset_paths:
        for class_id in sample['class_ids']:
            class_counts[class_id] += 1
    
    print("\n📊 Dataset Balance Analysis:")
    print("=" * 50)
    
    total_samples = sum(class_counts.values())
    max_count = max(class_counts.values())
    
    class_weights = {}
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        weight = max_count / count if count > 0 else 1.0
        class_weights[class_id] = weight
        
        class_name = list(class_to_id.keys())[list(class_to_id.values()).index(class_id)]
        percentage = (count / total_samples) * 100
        
        print(f"Class {class_id} ({class_name}): {count:,} samples ({percentage:.1f}%) - Weight: {weight:.2f}")
    
    print(f"\n📈 Balance Statistics:")
    print(f"Total samples: {total_samples:,}")
    print(f"Most common class: {max_count:,} samples")
    print(f"Least common class: {min(class_counts.values()):,} samples")
    print(f"Imbalance ratio: {max_count / min(class_counts.values()):.1f}:1")
    
    return class_weights, class_counts

def load_dataset_paths(image_dir, annotation_dir, class_to_id, test_mode=False):
    """
    Load dataset file paths with balanced sampling
    """
    print(f"📂 Loading dataset paths from {image_dir}")
    
    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    
    print(f"📸 Found {len(image_files)} images")
    
    # In test mode, use only first 1000 images
    if test_mode:
        image_files = image_files[:1000]
        print(f"🧪 TEST MODE: Using first {len(image_files)} images")
    
    dataset_paths = []
    processed_count = 0
    
    for img_path in image_files:
        # Get corresponding XML file
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        xml_path = os.path.join(annotation_dir, f"{img_name}.xml")
        
        if os.path.exists(xml_path):
            try:
                # Parse annotations first to check if valid
                annotations, width, height = parse_voc_xml(xml_path)
                
                if annotations:  # Only process images with annotations
                    # Convert annotations to class IDs and bbox format
                    class_ids = []
                    bboxes = []
                    
                    for class_name, x_center, y_center, w, h in annotations:
                        if class_name in class_to_id:
                            class_ids.append(class_to_id[class_name])
                            bboxes.append([x_center, y_center, w, h])
                    
                    if class_ids:  # Only include if we have valid classes
                        dataset_paths.append({
                            'image_path': img_path,
                            'xml_path': xml_path,
                            'class_ids': class_ids,
                            'bboxes': bboxes
                        })
                        
                        processed_count += 1
                        
                        # Progress indicator
                        if processed_count % 100 == 0:
                            print(f"📊 Processed {processed_count} valid samples...")
                        
                        # Memory management: limit dataset size in test mode
                        if test_mode and processed_count >= 500:
                            print(f"🧪 TEST MODE: Limiting to {processed_count} samples")
                            break
                        
            except Exception as e:
                print(f"⚠️ Error processing {img_path}: {e}")
                continue
    
    print(f"✅ Found {len(dataset_paths)} valid samples")
    return dataset_paths

def create_balanced_data_generator(dataset_paths, class_weights, batch_size=16, num_classes=24, input_size=224):
    """
    Create a data generator with balanced sampling based on class weights
    """
    def generator():
        while True:
            batch_images = []
            batch_classes = []
            batch_bboxes = []
            batch_weights = []
            
            # Sample with replacement based on class weights
            for _ in range(batch_size):
                # Weighted random sampling
                weights = []
                for sample in dataset_paths:
                    # Calculate sample weight based on its classes
                    sample_weight = 0
                    for class_id in sample['class_ids']:
                        sample_weight += class_weights.get(class_id, 1.0)
                    weights.append(sample_weight)
                
                # Normalize weights
                total_weight = sum(weights)
                if total_weight > 0:
                    weights = [w / total_weight for w in weights]
                else:
                    weights = [1.0 / len(dataset_paths)] * len(dataset_paths)
                
                # Sample based on weights
                chosen_sample = random.choices(dataset_paths, weights=weights, k=1)[0]
                
                try:
                    # Load image on-demand
                    img = Image.open(chosen_sample['image_path']).convert('RGB')
                    img = img.resize((input_size, input_size))
                    img_array = np.array(img) / 255.0
                    
                    batch_images.append(img_array)
                    
                    # Use the first annotation if multiple exist
                    if chosen_sample['class_ids']:
                        class_id = chosen_sample['class_ids'][0]
                        bbox = chosen_sample['bboxes'][0]
                        weight = class_weights.get(class_id, 1.0)
                        
                        batch_classes.append(class_id)
                        batch_bboxes.append(bbox)
                        batch_weights.append(weight)
                    else:
                        # Add dummy data if no annotations
                        batch_classes.append(0)
                        batch_bboxes.append([0.5, 0.5, 0.1, 0.1])
                        batch_weights.append(1.0)
                        
                except Exception as e:
                    print(f"⚠️ Error loading {chosen_sample['image_path']}: {e}")
                    # Add dummy data for failed images
                    batch_images.append(np.zeros((input_size, input_size, 3)))
                    batch_classes.append(0)
                    batch_bboxes.append([0.5, 0.5, 0.1, 0.1])
                    batch_weights.append(1.0)
            
            # Convert to numpy arrays
            batch_images = np.array(batch_images)
            batch_classes = np.array(batch_classes)
            batch_bboxes = np.array(batch_bboxes)
            batch_weights = np.array(batch_weights)
            
            yield batch_images, [batch_classes, batch_bboxes]
    
    return generator()

def create_validation_generator(dataset_paths, batch_size=16, num_classes=24, input_size=224):
    """
    Create a validation data generator
    """
    def generator():
        while True:
            # Randomly sample batch
            batch_data = random.sample(dataset_paths, min(batch_size, len(dataset_paths)))
            
            batch_images = []
            batch_classes = []
            batch_bboxes = []
            
            for sample in batch_data:
                try:
                    # Load image on-demand
                    img = Image.open(sample['image_path']).convert('RGB')
                    img = img.resize((input_size, input_size))
                    img_array = np.array(img) / 255.0
                    
                    batch_images.append(img_array)
                    
                    # Use the first annotation if multiple exist
                    if sample['class_ids']:
                        batch_classes.append(sample['class_ids'][0])
                        batch_bboxes.append(sample['bboxes'][0])
                    else:
                        # Add dummy data if no annotations
                        batch_classes.append(0)
                        batch_bboxes.append([0.5, 0.5, 0.1, 0.1])
                        
                except Exception as e:
                    print(f"⚠️ Error loading {sample['image_path']}: {e}")
                    # Add dummy data for failed images
                    batch_images.append(np.zeros((input_size, input_size, 3)))
                    batch_classes.append(0)
                    batch_bboxes.append([0.5, 0.5, 0.1, 0.1])
            
            # Convert to numpy arrays
            batch_images = np.array(batch_images)
            batch_classes = np.array(batch_classes)
            batch_bboxes = np.array(batch_bboxes)
            
            yield batch_images, [batch_classes, batch_bboxes]
    
    return generator()

def create_lightweight_mobilenet_ssd_model(num_classes=24, input_size=224):
    """
    Create a lightweight MobileNet-SSD model optimized for ESP32-S3
    """
    print(f"🏗️ Creating lightweight MobileNet-SSD model with {input_size}x{input_size} input...")
    
    # Use MobileNetV2 with smaller alpha for lighter model
    base_model = keras.applications.MobileNetV2(
        input_shape=(input_size, input_size, 3),
        alpha=0.35,  # 35% of original size - lighter than 0.5
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Create lightweight SSD head
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    
    # Lighter detection head
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.1)(x)
    
    # Output layers
    class_output = keras.layers.Dense(num_classes, activation='softmax', name='class_output')(x)
    bbox_output = keras.layers.Dense(4, name='bbox_output')(x)  # [x, y, width, height]
    
    model = keras.Model(inputs=base_model.input, outputs=[class_output, bbox_output])
    
    print(f"✅ Model created with {model.count_params():,} parameters")
    print(f"📊 Model size: {model.count_params() * 4 / 1024 / 1024:.1f} MB (FP32)")
    return model

class WeightedLoss:
    """
    Custom loss function that applies class weights
    """
    def __init__(self, class_weights):
        self.class_weights = class_weights
    
    def weighted_categorical_crossentropy(self, y_true, y_pred):
        # Ensure y_true is the right shape
        y_true = tf.squeeze(y_true, axis=-1)
        
        # Convert sparse labels to one-hot
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])
        
        # Apply class weights
        weights = tf.gather(tf.constant(list(self.class_weights.values()), dtype=tf.float32), 
                           tf.cast(y_true, tf.int32))
        
        # Calculate weighted loss
        ce_loss = tf.keras.losses.categorical_crossentropy(y_true_one_hot, y_pred)
        weighted_loss = ce_loss * weights
        
        return tf.reduce_mean(weighted_loss)

def train_model(image_dir, annotation_dir, num_classes=24, input_size=224, epochs=30, test_mode=False):
    """Train the lightweight MobileNet-SSD model with balanced sampling"""
    print("🚀 Starting lightweight MobileNet-SSD training with balanced sampling...")
    
    # Create class mapping
    class_to_id, id_to_class = create_class_mapping(annotation_dir)
    num_classes = len(class_to_id)
    
    # Load dataset
    dataset_paths = load_dataset_paths(image_dir, annotation_dir, class_to_id, test_mode=test_mode)
    
    if len(dataset_paths) == 0:
        print("❌ No valid samples found! Check your dataset paths and format.")
        return None, None, None
    
    # Analyze dataset balance and calculate weights
    class_weights, class_counts = analyze_dataset_balance(dataset_paths, class_to_id)
    
    # Split dataset into train/validation
    random.shuffle(dataset_paths)
    split_idx = int(0.8 * len(dataset_paths))
    train_dataset_paths = dataset_paths[:split_idx]
    val_dataset_paths = dataset_paths[split_idx:]
    
    print(f"📊 Dataset split: {len(train_dataset_paths)} train, {len(val_dataset_paths)} validation")
    
    # Create model
    model = create_lightweight_mobilenet_ssd_model(num_classes=num_classes, input_size=input_size)
    
    # Create weighted loss
    weighted_loss = WeightedLoss(class_weights)
    
    # Compile model with standard loss and class weights
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'class_output': 'sparse_categorical_crossentropy',
            'bbox_output': 'mse'
        },
        metrics={
            'class_output': ['accuracy', 'sparse_categorical_crossentropy'],
            'bbox_output': ['mae', 'mse']
        },
        loss_weights={
            'class_output': 1.0,
            'bbox_output': 0.5
        }
    )
    
    # Create data generators
    train_generator = create_balanced_data_generator(
        train_dataset_paths, class_weights, batch_size=16, 
        num_classes=num_classes, input_size=input_size
    )
    val_generator = create_validation_generator(
        val_dataset_paths, batch_size=16, 
        num_classes=num_classes, input_size=input_size
    )
    
    # Callbacks for better training
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-6
        ),
        keras.callbacks.ModelCheckpoint(
            'best_lightweight_mobilenet_ssd_model.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False
        )
    ]
    
    # Calculate steps per epoch
    steps_per_epoch = max(1, len(train_dataset_paths) // 16)
    validation_steps = max(1, len(val_dataset_paths) // 16)
    
    # Train the model
    print("📚 Training model with balanced sampling...")
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
        workers=1,  # Reduce memory usage
        use_multiprocessing=False  # Avoid multiprocessing issues
    )
    
    # Save the final model
    model.save('lightweight_mobilenet_ssd_model_final.h5')
    print("✅ Final model saved to lightweight_mobilenet_ssd_model_final.h5")
    
    # Save class mapping and weights
    with open('class_mapping_balanced.json', 'w') as f:
        json.dump({
            'class_to_id': class_to_id, 
            'id_to_class': id_to_class,
            'class_weights': class_weights,
            'class_counts': dict(class_counts)
        }, f, indent=2)
    print("✅ Class mapping and weights saved to class_mapping_balanced.json")
    
    return model, history, val_dataset_paths, class_weights

def evaluate_model(model, val_dataset_paths, class_weights, input_size=224):
    """Evaluate the trained model with balanced metrics"""
    print("🔍 Evaluating model...")
    
    # Create test data generator
    test_generator = create_validation_generator(val_dataset_paths, batch_size=32, num_classes=24, input_size=input_size)
    
    # Evaluate on test data
    test_steps = min(10, len(val_dataset_paths) // 32)
    evaluation = model.evaluate(test_generator, steps=test_steps, verbose=1)
    
    print("\n📊 Model Evaluation Results:")
    print(f"Test Loss: {evaluation[0]:.4f}")
    print(f"Class Accuracy: {evaluation[3]:.4f}")
    print(f"Class Loss: {evaluation[4]:.4f}")
    print(f"Bbox MAE: {evaluation[5]:.4f}")
    print(f"Bbox MSE: {evaluation[6]:.4f}")
    
    # Generate predictions for confusion matrix
    print("\n🔍 Generating predictions for detailed analysis...")
    predictions = []
    true_labels = []
    
    for i in range(test_steps):
        batch_data, batch_labels = next(test_generator)
        batch_pred = model.predict(batch_data, verbose=0)
        
        # Get class predictions
        class_pred = np.argmax(batch_pred[0], axis=1)
        predictions.extend(class_pred)
        true_labels.extend(batch_labels[0])
    
    # Print classification report
    print("\n📋 Classification Report:")
    print(classification_report(true_labels, predictions))
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Balanced Training')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix_balanced.png', dpi=300, bbox_inches='tight')
    print("✅ Confusion matrix saved as confusion_matrix_balanced.png")
    
    return evaluation

def plot_training_history(history):
    """Plot training history"""
    print("📈 Plotting training history...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot losses
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # Plot class accuracy
    axes[0, 1].plot(history.history['class_output_accuracy'], label='Training Accuracy')
    axes[0, 1].plot(history.history['val_class_output_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Classification Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    
    # Plot bbox MAE
    axes[1, 0].plot(history.history['bbox_output_mae'], label='Training MAE')
    axes[1, 0].plot(history.history['val_bbox_output_mae'], label='Validation MAE')
    axes[1, 0].set_title('Bounding Box MAE')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].legend()
    
    # Plot bbox MSE
    axes[1, 1].plot(history.history['bbox_output_mse'], label='Training MSE')
    axes[1, 1].plot(history.history['val_bbox_output_mse'], label='Validation MSE')
    axes[1, 1].set_title('Bounding Box MSE')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MSE')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('training_history_balanced.png', dpi=300, bbox_inches='tight')
    print("✅ Training history saved as training_history_balanced.png")

def main():
    print("🎯 Lightweight MobileNet-SSD Training with Balanced Sampling")
    print("=" * 70)
    
    # Configuration
    DATASET_PATH = "/home/jeavo/Pest_24"
    IMAGE_DIR = os.path.join(DATASET_PATH, "images")
    ANNOTATION_DIR = os.path.join(DATASET_PATH, "Annotations")
    
    # TEST MODE - Set to True for quick testing, False for full training
    TEST_MODE = False  # Full training mode
    
    INPUT_SIZE = 224
    EPOCHS = 30  # Reduced epochs for faster training
    
    print(f"📋 Configuration:")
    print(f"  • Dataset path: {DATASET_PATH}")
    print(f"  • Image directory: {IMAGE_DIR}")
    print(f"  • Annotation directory: {ANNOTATION_DIR}")
    print(f"  • Test mode: {TEST_MODE}")
    print(f"  • Input size: {INPUT_SIZE}x{INPUT_SIZE}")
    print(f"  • Training epochs: {EPOCHS}")
    print(f"  • Model: MobileNetV2 (alpha=0.35) - Lighter than before")
    print(f"  • Balanced sampling: Enabled")
    
    # Check if directories exist
    if not os.path.exists(IMAGE_DIR):
        print(f"❌ Image directory not found: {IMAGE_DIR}")
        print("💡 Please update the IMAGE_DIR path in the script")
        return
    
    if not os.path.exists(ANNOTATION_DIR):
        print(f"❌ Annotation directory not found: {ANNOTATION_DIR}")
        print("💡 Please update the ANNOTATION_DIR path in the script")
        return
    
    # Step 1: Train model
    print("\n📚 Step 1: Training lightweight MobileNet-SSD model...")
    model, history, val_dataset_paths, class_weights = train_model(
        image_dir=IMAGE_DIR,
        annotation_dir=ANNOTATION_DIR,
        input_size=INPUT_SIZE, 
        epochs=EPOCHS,
        test_mode=TEST_MODE
    )
    
    if model is None:
        print("❌ Training failed!")
        return
    
    # Step 2: Evaluate model
    print("\n🔍 Step 2: Evaluating model...")
    evaluation = evaluate_model(model, val_dataset_paths, class_weights, input_size=INPUT_SIZE)
    
    # Step 3: Plot training history
    print("\n📈 Step 3: Plotting training history...")
    plot_training_history(history)
    
    print("\n✅ Training pipeline completed!")
    print("\n📁 Generated files:")
    print("  • best_lightweight_mobilenet_ssd_model.h5 - Best model during training")
    print("  • lightweight_mobilenet_ssd_model_final.h5 - Final trained model")
    print("  • class_mapping_balanced.json - Class mapping and weights")
    print("  • confusion_matrix_balanced.png - Classification performance")
    print("  • training_history_balanced.png - Training curves")
    
    print("\n🎯 Model Specifications for ESP32-S3:")
    print(f"  • Input resolution: {INPUT_SIZE}x{INPUT_SIZE}x3")
    print(f"  • Parameters: {model.count_params():,}")
    print(f"  • FP32 size: {model.count_params() * 4 / 1024 / 1024:.1f} MB")
    print(f"  • Expected INT8 size: ~{model.count_params() / 1024 / 1024:.1f} MB")
    print(f"  • Expected inference time: ~2-3 seconds on ESP32-S3")
    
    print("\n💡 Improvements made:")
    print("  • Lighter model (alpha=0.35 vs 0.5)")
    print("  • Balanced sampling with class weights")
    print("  • Reduced training epochs (30 vs 50)")
    print("  • ESP32-compatible operations only")
    
    print("\n💡 Next steps:")
    print("1. Check the generated model files")
    print("2. Convert to TFLite INT8 for ESP32 deployment")
    print("3. Use class_mapping_balanced.json for inference")

if __name__ == "__main__":
    main()
