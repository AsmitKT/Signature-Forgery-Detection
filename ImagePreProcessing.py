import cv2
import numpy as np
from Binary import binarize_signature
from Crop import crop_signature_rotated
from Resize import resize_signature

def load_image(image_path):
    """Load the image from a given path."""
    return cv2.imread(image_path, cv2.IMREAD_COLOR)

def process_signature(image_path, save_path=None):
    """Pipeline to process the signature."""
    image = load_image(image_path)
    binary = binarize_signature(image)
    cropped = crop_signature_rotated(binary)
    resized = resize_signature(cropped)

    if save_path:
        cv2.imwrite(save_path, cropped)
    
    return resized

# Example usage
if __name__ == "__main__":
    input_path = "C:\Coding\Python\Signature Forgery Detection\Database\original_1_1.png"  # Change to your image path
    output_path = "C:\Coding\Python\Signature Forgery Detection\Database\processed_signature.jpg"
    processed = process_signature(input_path, output_path)
    cv2.imshow("Processed Signature", processed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
