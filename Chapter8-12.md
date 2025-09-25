# Chapters 8–12: CNNs, Advanced Models, Tracking, 3D Vision & Generative Models

## Chapter 8: Convolutional Neural Networks (CNNs)

### 8.1 Introduction
- CNNs are the backbone of modern computer vision.
- Exploit local connectivity, parameter sharing, and translation invariance.
- Ideal for tasks like image classification, segmentation, and object detection.

---

### 8.2 Architecture Components
1. **Convolutional Layer**
   - Applies learnable filters (kernels) across the input.
   - Extracts local patterns like edges, textures, or corners.
   - **Hyperparameters**: stride, padding, kernel size.

2. **Pooling Layer**
   - Downsamples feature maps to reduce computation.
   - **Max pooling**: takes the maximum value.
   - **Average pooling**: takes the mean.
   - Reduces overfitting and improves translation invariance.

3. **Fully Connected Layer**
   - Flattens feature maps and passes through dense layers for classification.

4. **Activation Functions**
   - **ReLU**: `f(x) = max(0, x)` – introduces non-linearity.
   - Leaky ReLU: allows small gradient for negative inputs.
   - Sigmoid / Tanh: less used in CNNs due to vanishing gradient.

---

### 8.3 Concepts
- **Receptive Field**: region of the input affecting a single feature in deeper layers.
- **Stride**: step size of the filter.
- **Padding**: adds zeros to borders; types include *valid* (no padding) and *same* (output size preserved).
- **Feature Maps**: outputs from convolutional filters.

---

### 8.4 Training & Optimization
- **Loss Functions**:
  - Cross-entropy loss for classification.
  - MSE for regression.
- **Optimizers**: SGD, Adam, RMSprop.
- **Regularization**:
  - Dropout, Batch Normalization, Data Augmentation.

---

## Chapter 9: Advanced Deep Learning Models in CV

### 9.1 Residual Networks (ResNet)
- Introduced *skip connections* to bypass layers.
- Solves **vanishing gradient problem** in very deep networks.
- Residual block adds the input directly to the output of a block.

### 9.2 Inception Networks (GoogleNet)
- Uses parallel convolutions with multiple kernel sizes (1×1, 3×3, 5×5).
- **1×1 convolutions** used for dimensionality reduction.
- Encourages multi-scale feature extraction.

### 9.3 DenseNet
- Each layer connects to every subsequent layer.
- Benefits: stronger gradient flow, feature reuse, parameter efficiency.

### 9.4 MobileNet / EfficientNet
- **MobileNet**:
  - Uses depthwise separable convolutions (factorizes standard convolution into depthwise + pointwise).
- **EfficientNet**:
  - Balances depth, width, and resolution using compound scaling.

### 9.5 Vision Transformers (ViT)
- Treats images as patches (e.g., 16×16 pixels).
- Uses **self-attention** to capture global dependencies.
- Replaces convolutions with transformer encoder blocks.

### 9.6 Swin Transformer
- Uses shifted windows for hierarchical representation.
- Combines advantages of CNNs (locality, hierarchy) with transformers.

---

## Chapter 10: Object Tracking

### 10.1 Introduction
- Tracking = following detected objects across frames.
- Key difference from detection: ensures *temporal consistency*.

### 10.2 Classical Methods
- **Optical Flow (Lucas-Kanade)**:
  - Tracks pixel motion by brightness constancy.
  - Assumes small displacements and smooth motion.
- **Kalman Filter**:
  - Predicts object state (position, velocity) using linear motion models.
  - Updates with new observations.
- **Particle Filter**:
  - Extends Kalman with non-linear and non-Gaussian assumptions.
- **Mean-Shift & CAMShift**:
  - Iteratively move search window to maximize similarity (color histogram).

### 10.3 Deep Learning Approaches
- **DeepSORT**:
  - Combines detection (bounding boxes) with appearance embedding.
  - Improves multi-object tracking.
- **Siamese Networks**:
  - Learn similarity between patches for robust tracking.
- **Correlation Filters**:
  - Learn filters that maximize response on the target.

---

## Chapter 11: 3D Computer Vision

### 11.1 Introduction
- Extends 2D image processing into depth & 3D understanding.
- Core in robotics, AR/VR, autonomous driving.

### 11.2 Stereo Vision
- Two cameras mimic human eyes.
- Disparity = difference in pixel locations between left & right images.
- Depth is calculated using focal length, baseline distance, and disparity.

### 11.3 Structure from Motion (SfM)
- Recovers 3D structure from 2D images with camera motion.
- Steps:
  1. Feature detection & matching.
  2. Camera pose estimation.
  3. 3D reconstruction.

### 11.4 Depth Estimation
- **Stereo-based**: triangulation.
- **Monocular**: deep learning-based single-image depth estimation.

### 11.5 3D Representations
- **Point Clouds**: set of 3D points (x, y, z).
- **Meshes**: vertices + edges + faces.
- **Voxel Grids**: 3D equivalent of pixels.

### 11.6 SLAM (Simultaneous Localization and Mapping)
- Builds map while estimating camera position.
- Approaches:
  - Feature-based (ORB-SLAM).
  - Direct (uses intensity values).

---

## Chapter 12: Generative Models in Computer Vision

### 12.1 Introduction
- Aim: generate new images that mimic real data.
- Applications: image synthesis, super-resolution, style transfer.

### 12.2 Autoencoders (AE)
- Encoder: compresses image into latent representation.
- Decoder: reconstructs image.
- Trained with reconstruction loss (MSE, L1).

### 12.3 Variational Autoencoders (VAE)
- Introduce stochastic latent variables.
- Loss combines reconstruction quality with distribution regularization.
- Generate new samples by sampling latent space.

### 12.4 Generative Adversarial Networks (GANs)
- Two-player minimax game:
  - Generator (G) creates fake images.
  - Discriminator (D) distinguishes real from fake.
- Variants:
  - **DCGAN**: deep CNN-based GAN.
  - **CycleGAN**: unpaired image-to-image translation.
  - **StyleGAN**: generates highly realistic human faces.

### 12.5 Diffusion Models
- Iteratively denoise noise to form images.
- Forward process: add Gaussian noise step by step.
- Reverse process: denoise using neural networks.
- Example: **Stable Diffusion**.

---
