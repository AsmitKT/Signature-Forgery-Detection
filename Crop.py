# Crop.py
import cv2
import numpy as np
import math

ASPECT_RATIO = 1.0  # width / height for the fixed-ratio crop

def crop_signature_rotated(binary, aspect_ratio=ASPECT_RATIO):
    """
    Crop a binary signature image to a fixed aspect ratio so that the signature touches
    all sides when possible; if not, it is centered on the axis it can't touch.
    No rotation or stretching is ever performed—only cropping.

    Args:
        binary (numpy.ndarray or str): Grayscale binary image array or path.
        aspect_ratio (float): Desired output width/height ratio.

    Returns:
        numpy.ndarray: Cropped binary image.
    """
    # Load if path provided
    if isinstance(binary, str):
        img = cv2.imread(binary, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image from {binary}")
        binary = img

    # Ensure strict binary (ink=0, bg=255)
    _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)

    h, w = binary.shape
    # Invert so signature pixels are non-zero
    inv = cv2.bitwise_not(binary)
    pts = cv2.findNonZero(inv)
    if pts is None:
        # No ink found: return as-is
        return binary

    # Get bounding box of signature
    x, y, w_sig, h_sig = cv2.boundingRect(pts)

    # Compute crop dimensions to match aspect ratio
    if (w_sig / h_sig) >= aspect_ratio:
        crop_w = w_sig
        crop_h = int(math.ceil(crop_w / aspect_ratio))
    else:
        crop_h = h_sig
        crop_w = int(math.ceil(crop_h * aspect_ratio))

    # Clamp crop size to image size
    crop_w = min(crop_w, w)
    crop_h = min(crop_h, h)

    # Center the signature bbox within the crop area
    cx = x + w_sig / 2.0
    cy = y + h_sig / 2.0
    x0 = int(round(cx - crop_w / 2.0))
    y0 = int(round(cy - crop_h / 2.0))

    # Clamp origin to lie within the image
    x0 = max(0, min(x0, w - crop_w))
    y0 = max(0, min(y0, h - crop_h))

    return binary[y0:y0+crop_h, x0:x0+crop_w]


if __name__ == "__main__":
    input_path  = r"C:\Coding\Python\Signature Forgery Detection\Database\original_1_1.png"   # Change to your image path :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}
    output_path = r"C:\Coding\Python\Signature Forgery Detection\Database\cropped_signature.png"
    cropped = crop_signature_rotated(input_path)
    cv2.imshow("Cropped Signature", cropped)                                                     # :contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}
    cv2.waitKey(0)
    cv2.destroyAllWindows()
