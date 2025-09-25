# Chapter 2: Basics of Digital Images

## 2.1 What is a Digital Image?
- A **digital image** is a two-dimensional array of numbers, where each number represents the intensity (brightness or color) of a pixel.
- Mathematically: `f(x, y)` where `x` and `y` are spatial coordinates and `f(x, y)` gives pixel intensity.

---

## 2.2 Pixels
- **Pixel (Picture Element)**: The smallest unit of a digital image.
- Each pixel has:
  - **Location** (coordinates: x, y)
  - **Intensity Value** (brightness for grayscale, RGB values for color)

---

## 2.3 Image Types
1. **Binary Image**
   - Pixels have only two values: 0 (black), 1 (white).
   - Example: Document scans, masks.
2. **Grayscale Image**
   - Pixel intensity ranges from 0 (black) to 255 (white).
   - Requires only one channel.
3. **Color Image**
   - Usually represented in **RGB model**:
     - R (Red), G (Green), B (Blue)
   - Each channel ranges from 0–255.
   - Example: `(255, 0, 0)` = pure red.

---

## 2.4 Image Resolution
- **Resolution**: Number of pixels in an image, usually written as width × height.
  - Example: 1920×1080 = Full HD.
- **Aspect Ratio**: Ratio of width to height.
  - Example: 16:9 widescreen format.

---

## 2.5 Color Models
- **RGB (Red, Green, Blue)**: Additive model, common in screens.
- **CMYK (Cyan, Magenta, Yellow, Key/Black)**: Subtractive model, used in printing.
- **HSV (Hue, Saturation, Value)**: Better for color analysis.
- **YCbCr / YUV**: Separates luminance (Y) and chrominance (Cb, Cr) for video compression.

---

## 2.6 Image File Formats
- **Lossless Compression**:
  - PNG, BMP, TIFF
  - Preserve exact pixel values.
- **Lossy Compression**:
  - JPEG, WebP
  - Remove some information to reduce file size.

---

## 2.7 Bit Depth
- **Bit depth**: Number of bits used to represent each pixel.
  - 1-bit = Binary image
  - 8-bit = 256 grayscale levels
  - 24-bit = RGB color (8 bits per channel → 16.7 million colors)

---

## 2.8 Image Representation in Memory
- **Matrix Representation**:
  - Grayscale: 2D matrix of pixel intensities.
  - Color: 3D matrix (Width × Height × Channels).
- Example:
  ```
  Grayscale Image = [[0, 125, 255],
                     [34, 200, 100]]

  Color Image = [[[255, 0, 0], [0, 255, 0]],
                 [[0, 0, 255], [255, 255, 0]]]
  ```

---

## 2.9 Summary
- Images are digital representations of visual information.
- Each pixel holds intensity or color information.
- Different color models and formats exist for various applications.
- Resolution, bit depth, and compression affect image quality and storage.

---
