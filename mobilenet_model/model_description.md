The trained model is a custom model based on the SSD MobilenetV2 architecture and exported as a TFLite model after a full INT8 quantization. 
This folder features the model as C header (.h) and as a .tflite, the training script and the export and quantization script.

The list of the packages we used in our training environment is avaible in requirements_mobilenet.txt.

--MODEL DESCRIPTION--

** Input **
- Image input: RGB, resized to 224×224, values scaled to [0,1]
- Annotations: Pascal-VOC XML
- Generators: Python generators yield (batch_images, batch_targets) on the fly; no heavy augmentations here (just resize).

** Backbone **
- MobileNetV2 with alpha = 1.0 (full width), ImageNet pretrained
- Feature tap is stride-16 (layer: block_13_expand_relu), which yields a 14×14 map for 224×224 input.

** Head **
- Compact YOLO-style head on top of the 14×14 feature map:
    1×1 Conv 256 → BN → ReLU6
    Two depthwise-separable 3×3 conv blocks
    
** Output **
- 1 objectness logit
- 4 box regressors
- C per-class logits (independent sigmoids)

** Losses & Optimizations **
- Objectness: Sigmoid focal loss (γ=2, α=0.25) on the objectness logit.
- Classes: Sigmoid focal loss on class logits, masked to positives.
- Boxes: Smooth-L1 on (tx,ty,tw,th), positives only. 
- Losses are normalized mainly by number of positives so the myriad easy negatives don’t dominate, and the final loss is a weighted sum (λ all set to 1.0 by default)
- Optimizer: Adam(1e-3). ReduceLROnPlateau, EarlyStopping, ModelCheckpoint enabled. 

** Inference & Decoding **
- Apply sigmoid to objectness & class logits; recover boxes:
    center = ( (tx+gx)/S, (ty+gy)/S )
    size = ( exp(tw)*anchor_w, exp(th)*anchor_h )
- Compute per-box scores = objectness × max_class_prob.
- Filter by confidence (default 0.25), then class-agnostic NMS (IoU 0.45, up to 100 dets).

Note: 
Option to optimize for small insects detection:
--> Second head: Add a small stride-32 (7×7) head or, better, a stride-8 (28×28) head using a light feature-fusion block; expect ~×2 head cost. 
