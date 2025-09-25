# Computer Vision Notes

## Chapter 1: Introduction to Computer Vision
- **Definition**: Computer Vision (CV) is a field of AI that enables machines to interpret and understand visual information from the world.
- **Applications**:
  - Image classification (e.g., identifying objects in photos)
  - Object detection (e.g., detecting pedestrians for autonomous cars)
  - Facial recognition (e.g., biometric authentication)
  - Medical imaging (e.g., tumor detection)
  - Augmented & Virtual Reality
- **Key Components**:
  - Image acquisition
  - Preprocessing
  - Feature extraction
  - High-level vision (recognition, interpretation)

---

## Chapter 2: Basics of Digital Images
- **Image Representation**:
  - An image is a 2D function `f(x, y)` where `x` and `y` are spatial coordinates.
  - The amplitude of `f(x, y)` represents pixel intensity.
- **Types of Images**:
  - Binary (black & white)
  - Grayscale (0–255 intensity)
  - Color (RGB channels)
- **Image Formats**:
  - Lossless: PNG, BMP
  - Lossy: JPEG
- **Resolution & Aspect Ratio**
  - Resolution: number of pixels (width × height)
  - Aspect Ratio: width/height ratio

---

## Chapter 3: Image Preprocessing
- **Noise Removal**:
  - Gaussian blur
  - Median filter
  - Bilateral filter
- **Histogram Equalization**:
  - Enhances contrast by redistributing pixel intensities.
- **Thresholding**:
  - Global threshold (Otsu’s method)
  - Adaptive threshold
- **Geometric Transformations**:
  - Translation, rotation, scaling, affine & perspective transforms
- **Morphological Operations** (on binary images):
  - Erosion, dilation
  - Opening, closing

---

## Chapter 4: Feature Extraction
- **Edges**:
  - Sobel, Prewitt, Canny edge detector
- **Corners & Keypoints**:
  - Harris corner detector
  - FAST, ORB
- **Descriptors**:
  - SIFT (Scale-Invariant Feature Transform)
  - SURF (Speeded-Up Robust Features)
  - BRIEF
- **HOG (Histogram of Oriented Gradients)**:
  - Used for object detection (e.g., pedestrian detection)

---

## Chapter 5: Image Segmentation
- **Definition**: Partitioning an image into regions of interest.
- **Techniques**:
  - Threshold-based segmentation
  - Region-based segmentation (Region growing, Watershed)
  - Edge-based segmentation
  - Clustering (K-Means, Mean-Shift)
  - Graph-based segmentation
  - Deep learning segmentation (U-Net, Mask R-CNN)

---

## Chapter 6: Object Detection
- **Traditional Methods**:
  - Sliding window
  - Viola-Jones algorithm (Haar cascades)
- **Modern Methods**:
  - R-CNN, Fast R-CNN, Faster R-CNN
  - YOLO (You Only Look Once)
  - SSD (Single Shot MultiBox Detector)
  - DETR (DEtection TRansformer)

---

## Chapter 7: Image Classification
- **Traditional Approaches**:
  - Feature extraction + ML classifier (SVM, Random Forest)
- **Deep Learning Approaches**:
  - CNN (Convolutional Neural Networks)
  - Pre-trained networks (VGG, ResNet, Inception)
  - Transfer learning

---

## Chapter 8: Convolutional Neural Networks (CNNs)
- **Architecture**:
  - Convolution layers
  - Pooling layers (Max/Avg pooling)
  - Fully connected layers
  - Activation functions (ReLU, Sigmoid, Tanh)
- **Concepts**:
  - Padding, stride
  - Receptive field
  - Feature maps
- **Regularization**:
  - Dropout
  - Batch Normalization

---

## Chapter 9: Advanced Deep Learning Models in CV
- **Residual Networks (ResNet)**
- **Inception Networks (GoogleNet)**
- **DenseNet**
- **MobileNet / EfficientNet**
- **Vision Transformers (ViT)**
- **Swin Transformer**

---

## Chapter 10: Object Tracking
- **Definition**: Following objects across video frames.
- **Algorithms**:
  - Optical Flow (Lucas-Kanade)
  - Kalman Filter
  - Mean-Shift & CAMShift
  - DeepSORT (Deep Learning-based tracking)

---

## Chapter 11: 3D Computer Vision
- **Stereo Vision**
- **Structure from Motion (SfM)**
- **Depth Estimation**
- **Point Clouds & 3D Reconstruction**
- **SLAM (Simultaneous Localization and Mapping)**

---

## Chapter 12: Generative Models in Computer Vision
- **Autoencoders (AE)**
- **Variational Autoencoders (VAE)**
- **GANs (Generative Adversarial Networks)**
  - DCGAN, CycleGAN, StyleGAN
- **Diffusion Models** (e.g., Stable Diffusion)

---

## Chapter 13: Vision + Language Models
- **Image Captioning**:
  - CNN + RNN/LSTM models
  - Transformer-based captioning (BLIP, OSCAR)
- **Visual Question Answering (VQA)**
- **Multimodal Models (CLIP, ALIGN, Flamingo)**

---

## Chapter 14: Applications of Computer Vision
- Autonomous Vehicles
- Robotics
- Healthcare Imaging
- Security & Surveillance
- Retail & E-commerce (visual search)
- Agriculture (crop monitoring)
- Industry (defect detection)

---

## Chapter 15: Current Trends & Future Directions
- Foundation Models (e.g., CLIP, DINOv2)
- Self-supervised learning in vision
- Edge AI & deployment on low-power devices
- Ethics & Bias in Computer Vision
- Explainability & Trustworthy AI in Vision
