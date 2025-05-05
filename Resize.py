# Resize.py
import cv2
import numpy as np

# Hard-coded output canvas size (width, height), divisible by 32 for easy tiling in ML pipelines
TARGET_SIZE = (256, 256)

def resize_signature(image, target_size=TARGET_SIZE):
    """
    1. Load a binary/grayscale signature.
    2. Rotate it so its longest side is horizontal (no stretching).
    3. Uniformly scale it to fit within `target_size`, preserving aspect ratio.
    4. Place it centered on a white canvas of size `target_size`.

    Args:
        image (str or numpy.ndarray): Path to image or loaded image array.
        target_size (tuple[int,int]): (width, height) of the output canvas.

    Returns:
        numpy.ndarray: The rotated, resized & padded image of shape target_size.
    """
    # 1. Load
    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image from {image}")
    else:
        img = image.copy()

    # 2. Rotate to make the longer side horizontal
    h, w = img.shape
    if h > w:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        h, w = img.shape

    # 3. Compute uniform scale factor
    targ_w, targ_h = target_size
    scale = min(targ_w / w, targ_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    # Choose interpolation: AREA for downscaling, CUBIC for upscaling
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC

    # 4a. Resize signature
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    # 4b. Paste onto white canvas
    canvas = np.full((targ_h, targ_w), 255, dtype=resized.dtype)
    x_off = (targ_w - new_w) // 2
    y_off = (targ_h - new_h) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized

    return canvas

if __name__ == "__main__":
    input_path  = r"C:\path\to\your\cropped_signature.png"   # Change to your file path
    output_path = r"C:\path\to\your\resized_signature.png"   # Defined for consistency

    final = resize_signature(input_path)

    # To save the result, uncomment:
    # cv2.imwrite(output_path, final)

    cv2.imshow("Resized + Padded Signature", final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
