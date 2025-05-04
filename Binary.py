import cv2
import numpy as np

def binarize_image(image):
    """Divide image into 12 parts and apply local binarization to each, keeping ink black."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    rows, cols = 3, 4  # Divide into 12 parts
    box_h, box_w = h // rows, w // cols

    binary = np.zeros_like(gray)
    for i in range(rows):
        for j in range(cols):
            y1 = i * box_h
            y2 = (i + 1) * box_h if i < rows - 1 else h
            x1 = j * box_w
            x2 = (j + 1) * box_w if j < cols - 1 else w

            roi = gray[y1:y2, x1:x2]

            # Apply median filtering to further reduce salt-and-pepper noise.
            roi = cv2.medianBlur(roi, 3)

            # Apply Gaussian smoothing before thresholding to reduce high-frequency noise.
            roi = cv2.GaussianBlur(roi, (3, 3), 0)


            # Adaptive thresholding to create a binary image with black ink.
            local_thresh = cv2.adaptiveThreshold(roi, 255,
                                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY,
                                                 11, 2)
            binary[y1:y2, x1:x2] = local_thresh

    return binary

def refine_signature(binary, original):
    """
    Refine the signature using the binary image as a mask:
    - Extract signature pixel intensities from the original grayscale image.
    - Compute a refined threshold based on the dark signature strokes.
    - Reapply thresholding so that ink strokes are black (0) and the background is white (255).
    """
    # Convert original image to grayscale.
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    
    # Create a mask where the binary image identifies the signature (assumes ink pixels are black).
    mask = (binary == 0)
    
    # Extract the intensities for the signature pixels from the grayscale image.
    signature_pixels = gray[mask]
    
    # Compute a refined threshold using the median of the signature pixels.
    if signature_pixels.size > 0:
        refined_thresh_value = np.median(signature_pixels)
    else:
        refined_thresh_value = 127  # Fallback value if no signature pixels are found.

    # Reapply thresholding so that pixels with intensity lower than or equal to the threshold become black.
    _, refined_binary = cv2.threshold(gray, refined_thresh_value, 255, cv2.THRESH_BINARY)
    
    return refined_binary

def smooth_final(refined):
    """
    Apply an overall light Gaussian smoothing to the refined image to remove residual small dots.
    Then reapply binary thresholding to ensure the output has clear black ink on a white background.
    """
    # Apply Gaussian smoothing with a small kernel. 
    # Increase the kernel size if you still see small residual dots but be cautious of blurring the signature.
    smoothed = cv2.GaussianBlur(refined, (5, 5), 0)
    
    # Reapply a simple binary threshold to clean up the smoothed image.
    # Using a threshold value of 127 here, but you might adjust it based on the image properties.
    _, final_image = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)
    
    return final_image

def binarize_signature(image):
    if isinstance(image, str):  # If a path was passed
        image = cv2.imread(image)
    if image is None:
        raise ValueError("Image not found or could not be read.")
    
    # Step 1: Perform local adaptive binarization.
    binary = binarize_image(image)
    
    # Step 2: Refine the signature based on the original image intensities.
    refined = refine_signature(binary, image)
    
    # Step 3: Apply overall Gaussian smoothing and re-threshold to remove small dots.
    final_result = smooth_final(refined)
    
    return final_result

if __name__ == "__main__":
    input_path = r"C:\Coding\Python\Signature Forgery Detection\Database\original_1_1.png"  # Change to your image path
    output_path = r"C:\Coding\Python\Signature Forgery Detection\Database\processed_signature.jpg"
    processed = binarize_signature(input_path)
    cv2.imshow("Processed Signature", processed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
