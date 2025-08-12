#!/usr/bin/env python3
"""
Fixed MDET Training Script - Proper MobileNetV2 + YOLO architecture
"""

import os, glob, math, random, xml.etree.ElementTree as ET
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --------------------
# Config - FIXED VERSION
# --------------------
IMG_SIZE = 224
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
ALPHA = 1.0         # FIXED: Use full MobileNetV2 width (was 0.35)
FREEZE_BACKBONE = False  # FIXED: Allow backbone training (was True)
BATCH_SIZE = 16
EPOCHS = 60

# Single feature map grid size (14x14 for 224 input using stride=16)
S = IMG_SIZE // 16              # 224/16 = 14
A = 3                           # anchors per cell
# Anchors (w,h) relative to image size (on 0..1 scale). Tune to your dataset if needed
ANCHORS = np.array([[0.10,0.08], [0.18,0.15], [0.28,0.24]], dtype=np.float32)

# --- paths: align with previous model ---
DATASET_PATH = "/home/jeavo/Pest_24"   # same root as before
IMAGES_DIR   = os.path.join(DATASET_PATH, "images")
ANN_DIR      = os.path.join(DATASET_PATH, "Annotations")

# --------------------
# Dataset utils (VOC) - Same as before
# --------------------
def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    w = float(size.find("width").text)
    h = float(size.find("height").text)
    objs = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)
        # convert to center/width/height normalized
        xc = ((xmin + xmax) / 2.0) / w
        yc = ((ymin + ymax) / 2.0) / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        objs.append((name, xc, yc, bw, bh))
    return objs

def build_class_map(ann_paths):
    classes = set()
    for ap in ann_paths:
        for name, *_ in parse_voc_xml(ap):
            classes.add(name)
    classes = sorted(list(classes))
    name2id = {n:i for i,n in enumerate(classes)}
    return classes, name2id

def load_items():
    img_paths = sorted(glob.glob(os.path.join(IMAGES_DIR, "*")))
    base_to_img = {os.path.splitext(os.path.basename(p))[0]: p for p in img_paths}
    ann_paths = sorted(glob.glob(os.path.join(ANN_DIR, "*.xml")))
    items = []
    for ap in ann_paths:
        base = os.path.splitext(os.path.basename(ap))[0]
        if base not in base_to_img:
            continue
        objs = parse_voc_xml(ap)
        if len(objs) == 0:
            continue
        items.append((base_to_img[base], ap))
    return items

# --------------------
# Target assignment (single grid + anchors) - Same as before
# --------------------
def iou_wh(box_wh, anchor_wh):
    # box_wh: [N,2], anchor_wh: [A,2] -> IoU on width/height only (aligned centers)
    box_area = box_wh[:,0]*box_wh[:,1]
    anc_area = anchor_wh[:,0]*anchor_wh[:,1]
    inter_w = np.minimum(box_wh[:,0:1], anchor_wh[None,:,0])
    inter_h = np.minimum(box_wh[:,1:2], anchor_wh[None,:,1])
    inter = inter_w*inter_h
    union = box_area[:,None] + anc_area[None,:] - inter
    return inter / np.maximum(union, 1e-9)

def encode_targets(objs, name2id, num_classes):
    # Targets shape: (S, S, A, (1 obj + 4 box + num_classes))
    t = np.zeros((S, S, A, 1 + 4 + num_classes), dtype=np.float32)
    # For each GT, assign to best anchor at its grid cell
    for (name, xc, yc, bw, bh) in objs:
        if bw <= 0 or bh <= 0: 
            continue
        gx = int(np.clip(np.floor(xc * S), 0, S-1))
        gy = int(np.clip(np.floor(yc * S), 0, S-1))
        # Choose best anchor by IoU (w,h)
        ious = iou_wh(np.array([[bw, bh]], dtype=np.float32), ANCHORS)[0]
        a = int(np.argmax(ious))
        # tx,ty are offsets to cell center (0..1), tw,th are log scale ratios
        cell_x = (xc * S) - gx
        cell_y = (yc * S) - gy
        aw, ah = ANCHORS[a]
        tw = math.log(np.clip(bw / (aw + 1e-9), 1e-6, 1e6))
        th = math.log(np.clip(bh / (ah + 1e-9), 1e-6, 1e6))
        cls_id = name2id[name]
        # assign
        t[gy, gx, a, 0] = 1.0                  # objectness
        t[gy, gx, a, 1:5] = [cell_x, cell_y, tw, th]
        t[gy, gx, a, 5 + cls_id] = 1.0         # one-hot class
    return t

# --------------------
# Data generator - Same as before
# --------------------
def letterbox(img):
    # Simple resize-without-padding for speed (keeps aspect via center-crop could be added)
    return tf.image.resize(img, (IMG_SIZE, IMG_SIZE), method="bilinear")

def parse_item(img_path, ann_path, name2id, num_classes):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = letterbox(img)
    # Load GT
    objs = parse_voc_xml(ann_path)
    t = encode_targets(objs, name2id, num_classes)
    return img, t

def tf_gen(items, name2id, num_classes, shuffle=True):
    while True:
        if shuffle:
            random.shuffle(items)
        for i in range(0, len(items), BATCH_SIZE):
            batch = items[i:i+BATCH_SIZE]
            imgs = []
            tgts = []
            for img_path, ann_path in batch:
                img, t = parse_item(img_path, ann_path, name2id, num_classes)
                imgs.append(img.numpy())
                tgts.append(t)
            yield np.stack(imgs,0), np.stack(tgts,0)

# --------------------
# Model: MobileNetV2 backbone + PROPER YOLO head - FIXED VERSION
# --------------------
def proper_yolo_head(feature, num_classes):
    """
    FIXED: Proper YOLO head with adequate capacity
    """
    # Feature extraction layers
    x = layers.Conv2D(256, 1, padding="same", use_bias=False)(feature)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.)(x)
    
    # Deeper network for better feature learning
    x = layers.SeparableConv2D(256, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.)(x)
    
    x = layers.SeparableConv2D(256, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.)(x)
    
    # Final prediction layer
    out_ch = A * (1 + 4 + num_classes)  # 3 * (1 + 4 + 24) = 87
    x = layers.Conv2D(out_ch, 1, padding="same")(x)
    
    # Output shape: (B, S, S, A*(1+4+C)) → reshape
    x = layers.Reshape((S, S, A, 1 + 4 + num_classes))(x)
    return x

def build_model(num_classes):
    inp = keras.Input(shape=INPUT_SHAPE)
    
    # FIXED: Use proper MobileNetV2 with ALPHA=1.0
    base = keras.applications.MobileNetV2(
        input_tensor=inp, 
        include_top=False, 
        alpha=ALPHA,  # Now 1.0 instead of 0.35
        weights="imagenet"
    )
    
    # Take feature at stride 16 - use block_13_expand_relu for 14x14 at 224 input
    feat = base.get_layer("block_13_expand_relu").output  # 14x14 at 224 input
    
    # FIXED: Use proper YOLO head instead of tiny head
    pred = proper_yolo_head(feat, num_classes)
    model = keras.Model(inp, pred)
    
    # FIXED: Don't freeze backbone (was True)
    if FREEZE_BACKBONE:
        for l in base.layers:
            l.trainable = False
    
    return model

# --------------------
# Losses: focal (obj & cls) + Smooth L1 (box) - Same as before
# --------------------
def sigmoid_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    # y_true, y_pred as logits? We'll pass raw logits; use tf.nn.sigmoid_cross_entropy_with_logits
    ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    p = tf.nn.sigmoid(y_pred)
    p_t = y_true * p + (1 - y_true) * (1 - p)
    loss = ce * tf.pow(1 - p_t, gamma)
    if alpha is not None:
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        loss = alpha_t * loss
    return loss

@tf.function
def detection_loss(y_true, y_pred, num_classes, lambda_box=1.0, lambda_cls=1.0, lambda_obj=1.0):
    # y_* shape: (B,S,S,A,1+4+C)
    obj_true = y_true[..., 0:1]                   # 0/1
    box_true = y_true[..., 1:5]                   # tx, ty, tw, th (targets)
    cls_true = y_true[..., 5:]                    # one-hot

    obj_logit = y_pred[..., 0:1]
    box_pred = y_pred[..., 1:5]
    cls_logit = y_pred[..., 5:]

    # Objectness focal
    obj_loss = sigmoid_focal_loss(obj_true, obj_logit)
    # Class focal (only where obj=1)
    cls_loss = sigmoid_focal_loss(cls_true, cls_logit)
    cls_loss = cls_loss * obj_true                # mask negatives

    # Box Smooth L1 (only positives)
    box_loss = tf.abs(box_true - box_pred)
    box_loss = tf.where(box_loss < 1.0, 0.5*box_loss**2, box_loss - 0.5)
    box_loss = tf.reduce_sum(box_loss, axis=-1, keepdims=True) * obj_true

    # Normalization by positives count (avoid bias toward easy negatives)
    pos = tf.reduce_sum(obj_true) + 1e-6
    obj_loss = tf.reduce_sum(obj_loss) / (tf.cast(tf.size(obj_loss), tf.float32))
    cls_loss = tf.reduce_sum(cls_loss) / (pos * tf.cast(num_classes, tf.float32))
    box_loss = tf.reduce_sum(box_loss) / pos

    total = lambda_obj*obj_loss + lambda_cls*cls_loss + lambda_box*box_loss
    return total, obj_loss, cls_loss, box_loss

class DetectorLoss(keras.losses.Loss):
    def __init__(self, num_classes):
        super().__init__(reduction=keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        self.num_classes = num_classes
    def call(self, y_true, y_pred):
        total, _, _, _ = detection_loss(y_true, y_pred, self.num_classes)
        return total

# --------------------
# Inference decode (NMS) - Same as before
# --------------------
def decode_predictions(pred, conf_thresh=0.25, iou_thresh=0.45, max_dets=100):
    # pred: (S,S,A,1+4+C) logits for obj/cls, linear for box
    obj = tf.sigmoid(pred[..., 0])                 # (S,S,A)
    box = pred[..., 1:5]                           # tx,ty,tw,th
    cls = tf.sigmoid(pred[..., 5:])                # (S,S,A,C)
    S_, A_ = pred.shape[0], pred.shape[2]
    # Build grid
    gy, gx = tf.meshgrid(tf.range(S_), tf.range(S_), indexing="ij")
    gx = tf.cast(gx, tf.float32); gy = tf.cast(gy, tf.float32)
    
    # Expand grid dimensions to match anchor dimension [S, S] -> [S, S, 1] -> [S, S, A]
    gx = tf.expand_dims(gx, axis=-1)  # [S, S, 1]
    gy = tf.expand_dims(gy, axis=-1)  # [S, S, 1]

    # Recover boxes (normalized 0..1)
    cell_x = (box[..., 0] + gx) / float(S_)        # x_center
    cell_y = (box[..., 1] + gy) / float(S_)        # y_center
    aw = tf.constant(ANCHORS[:,0], dtype=tf.float32)
    ah = tf.constant(ANCHORS[:,1], dtype=tf.float32)
    bw = tf.exp(box[..., 2]) * aw                  # width
    bh = tf.exp(box[..., 3]) * ah                  # height

    # Flatten
    x1 = tf.clip_by_value(cell_x - bw/2, 0., 1.)
    y1 = tf.clip_by_value(cell_y - bh/2, 0., 1.)
    x2 = tf.clip_by_value(cell_x + bw/2, 0., 1.)
    y2 = tf.clip_by_value(cell_y + bh/2, 0., 1.)
    boxes = tf.stack([y1, x1, y2, x2], axis=-1)    # to y1,x1,y2,x2 for TF NMS

    boxes = tf.reshape(boxes, (-1, 4))
    obj = tf.reshape(obj, (-1,))
    cls = tf.reshape(cls, (-1, cls.shape[-1]))

    # Scores per class = obj * p(class)
    scores = obj[:, None] * cls                    # (N, C)
    # Pick top class per box
    class_ids = tf.argmax(scores, axis=-1)
    class_scores = tf.reduce_max(scores, axis=-1)

    # Filter low confidence
    keep = class_scores > conf_thresh
    boxes, class_ids, class_scores = boxes[keep], tf.cast(class_ids[keep], tf.int32), class_scores[keep]

    # NMS (class-agnostic)
    selected = tf.image.non_max_suppression(boxes, class_scores, max_output_size=max_dets, iou_threshold=iou_thresh)
    return tf.gather(boxes, selected), tf.gather(class_ids, selected), tf.gather(class_scores, selected)

# --------------------
# Train - FIXED VERSION
# --------------------
def main():
    print(" Starting FIXED MDET Training Script")
    print("=" * 50)
    print(f" Configuration:")
    print(f"  • ALPHA: {ALPHA} (was 0.35 - now proper MobileNetV2)")
    print(f"  • FREEZE_BACKBONE: {FREEZE_BACKBONE} (was True - now allows training)")
    print(f"  • Model architecture: Proper YOLO head (was tiny head)")
    print("=" * 50)
    
    # Collect dataset
    items = load_items()
    assert len(items) > 0, "No (image, annotation) pairs found."
    classes, name2id = build_class_map([ap for _, ap in items])
    num_classes = len(classes)
    print(f" Dataset: {len(items)} items, {num_classes} classes")
    print(f" Classes: {classes}")

    # Split
    random.seed(1337)
    random.shuffle(items)
    n_train = int(0.8 * len(items))
    train_items = items[:n_train]
    val_items = items[n_train:]
    print(f" Training: {len(train_items)}, Validation: {len(val_items)}")

    # Model
    print("\n Building model...")
    model = build_model(num_classes)
    print("\n Model Summary:")
    model.summary()
    
    # Calculate expected model size
    total_params = model.count_params()
    model_size_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
    print(f"\n Expected model size: {model_size_mb:.2f} MB")
    print(f" Expected parameters: {total_params:,}")
    
    if model_size_mb < 5.0:
        print(" WARNING: Model size seems too small for MobileNetV2 + YOLO!")
    elif model_size_mb > 20.0:
        print(" WARNING: Model size seems too large!")
    else:
        print(" Model size looks reasonable!")

    # Compile
    print("\n Compiling model...")
    opt = keras.optimizers.Adam(1e-3)
    model.compile(optimizer=opt, loss=DetectorLoss(num_classes))

    # Datasets
    train_gen = tf_gen(train_items, name2id, num_classes, shuffle=True)
    val_gen   = tf_gen(val_items,   name2id, num_classes, shuffle=False)

    steps_tr = max(1, len(train_items)//BATCH_SIZE)
    steps_va = max(1, len(val_items)//BATCH_SIZE)

    # Callbacks
    cbs = [
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint("mdet_best_fixed.h5", monitor="val_loss", save_best_only=True, verbose=1)
    ]

    # Train
    print(f"\n Starting training for {EPOCHS} epochs...")
    model.fit(train_gen, validation_data=val_gen, steps_per_epoch=steps_tr, validation_steps=steps_va,
              epochs=EPOCHS, callbacks=cbs)

    # Save final
    model.save("mdet_final_fixed.h5")
    print(" Training completed! Models saved.")

    # Quick sanity check on one batch + NMS
    print("\n Testing model on validation sample...")
    imgs, tgts = next(val_gen)
    preds = model.predict(imgs[:1], verbose=0)[0]           # (S,S,A,1+4+C)
    boxes, cls_ids, scores = decode_predictions(preds)
    print(f"Detections: {boxes.shape[0]}")
    print("Sample classes:", [classes[int(i)] for i in cls_ids.numpy()[:10]])
    print("Sample scores:", scores.numpy()[:10])

    # TFLite (float) export
    print("\n Exporting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open("mdet_float_fixed.tflite", "wb").write(tflite_model)
    print(" Saved mdet_float_fixed.tflite")

if __name__ == "__main__":
    main()
