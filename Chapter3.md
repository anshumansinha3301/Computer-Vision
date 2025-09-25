# Chapter 3: Image Preprocessing

## 3.1 Introduction
- **Definition**: Image preprocessing refers to the set of operations applied to images to enhance their quality or prepare them for further analysis.
- **Objective**: Improve image clarity, highlight important features, and reduce noise or irrelevant details.
- **Importance**: Preprocessing ensures better accuracy for feature extraction, segmentation, and recognition tasks.

---

## 3.2 Noise in Images
- **Noise**: Unwanted random variations in pixel intensities.
- **Types of Noise**:
  - Gaussian noise (normal distribution of intensity variations)
  - Salt-and-pepper noise (random black and white pixels)
  - Speckle noise (common in radar images)

---

## 3.3 Noise Removal Techniques
- **Spatial Domain Filters**:
  - **Mean Filter**: Smoothens image by averaging neighborhood pixels.
  - **Median Filter**: Replaces each pixel with the median of neighbors (effective for salt-and-pepper noise).
  - **Gaussian Filter**: Weighted average using Gaussian distribution.
- **Frequency Domain Filters**:
  - Low-pass filters: Preserve low frequencies, remove noise.
  - High-pass filters: Enhance edges and fine details.

---

## 3.4 Histogram Processing
- **Histogram**: A graphical representation of pixel intensity distribution.
- **Histogram Equalization**:
  - Improves contrast by redistributing pixel intensities.
  - Useful in low-contrast images.
- **Adaptive Histogram Equalization (CLAHE)**:
  - Works on small regions to enhance local contrast.

---

## 3.5 Thresholding
- **Definition**: Converting a grayscale image into a binary image based on a threshold value.
- **Types**:
  - **Global Thresholding**: Single threshold for entire image.
  - **Otsu’s Method**: Automatic threshold selection based on histogram.
  - **Adaptive Thresholding**: Threshold varies over regions of the image.

---

## 3.6 Geometric Transformations
- **Translation**: Shifting an image along x or y axis.
- **Scaling**: Enlarging or reducing image size.
- **Rotation**: Rotating image around a point.
- **Shearing**: Slanting the shape of an image.
- **Affine Transformation**: Preserves lines and parallelism.
- **Perspective Transformation**: Mimics depth by projecting onto a plane.

---

## 3.7 Morphological Operations (for Binary Images)
- **Erosion**: Shrinks object boundaries (removes small white noise).
- **Dilation**: Expands object boundaries (fills small holes).
- **Opening**: Erosion followed by dilation (removes small objects).
- **Closing**: Dilation followed by erosion (fills small gaps).

---

## 3.8 Edge Enhancement & Smoothing
- **Smoothing (Blurring)**:
  - Removes noise and reduces detail.
- **Sharpening**:
  - Enhances edges and fine details using high-pass filters.

---

## 3.9 Summary
- Image preprocessing improves image quality for better analysis.
- Techniques include noise removal, histogram processing, thresholding, transformations, and morphology.
- Choice of method depends on the type of noise, application, and desired features.

---
