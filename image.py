import cv2
import pytesseract
import numpy as np

# Load the image
image = cv2.imread('image_10_0.jpeg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Optional: Apply Gaussian Blur to smooth the image (optional)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply adaptive thresholding to highlight the text
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY_INV, 11, 2)

# Use Tesseract to extract text and its bounding box data
custom_config = r'--oem 3 --psm 6'
data = pytesseract.image_to_data(thresh, config=custom_config, output_type=pytesseract.Output.DICT)

# Show the image (for debugging purposes)
cv2.imshow('Processed Image', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
