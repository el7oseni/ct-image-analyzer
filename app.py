import cv2
import numpy as np
import pandas as pd
import pydicom
import os

# Function to load the DICOM image and apply rescale slope and intercept
def load_image(file_path):
    dicom_data = pydicom.dcmread(file_path)
    image = dicom_data.pixel_array.astype(np.float32)
    
    # Retrieve rescale slope and intercept
    rescale_slope = getattr(dicom_data, 'RescaleSlope', 1)
    rescale_intercept = getattr(dicom_data, 'RescaleIntercept', 0)
    
    # Apply transformation
    image = (image * rescale_slope) + rescale_intercept
    return image, dicom_data

# Function to draw a precise, anti-aliased circle for visualization
def draw_circle(image, center, diameter, color=(0, 255, 255), thickness=1):
    """
    Draw a precise, thin circle for visualization using OpenCV's built-in anti-aliasing.
    Args:
        image: The image on which to draw.
        center: (x, y) coordinates of the circle center.
        diameter: Diameter of the circle in pixels.
        color: Color of the circle in BGR (default: yellow).
        thickness: Thickness of the circle edge.
    """
    radius = int(diameter / 2)
    center_rounded = (int(center[0]), int(center[1]))
    # Use OpenCV's anti-aliased drawing method to ensure smooth edges
    cv2.circle(image, center_rounded, radius, color, thickness, lineType=cv2.LINE_AA)

# Initialize variables
circle_diameter = 9  # Start with a circle diameter of 9 pixels
zoom_factor = 1.0
results = []
selected_points = []

# Function to handle mouse clicks and perform analysis
def click_event(event, x, y, flags, param):
    global results, selected_points, circle_diameter, pixel_spacing, zoom_factor
    if event == cv2.EVENT_LBUTTONDOWN:
        # Map clicked coordinates back to original image
        x_original = int(x / zoom_factor)
        y_original = int(y / zoom_factor)
        selected_points.append((x_original, y_original))
        print(f"Selected Point: ({x_original}, {y_original}), Circle Diameter: {circle_diameter:.1f} pixels")

        # Create a circular mask
        mask = np.zeros_like(image, dtype=np.uint8)
        y_indices, x_indices = np.ogrid[:image.shape[0], :image.shape[1]]
        distance_from_center = np.sqrt((x_indices - x_original)**2 + (y_indices - y_original)**2)
        mask[distance_from_center <= circle_diameter / 2] = 1

        # Extract pixel values within the circle
        pixels = image[mask == 1]

        # Calculate metrics
        area_pixels = np.sum(mask)
        area_mm2 = area_pixels * (pixel_spacing**2)
        mean = np.mean(pixels)
        stddev = np.std(pixels)
        min_val = np.min(pixels)
        max_val = np.max(pixels)

        # Save the result with formatted precision
        results.append({
            'Point': f"({x_original}, {y_original})",
            'Area (mm²)': f"{area_mm2:.3f}",
            'Mean': f"{mean:.3f}",
            'StdDev': f"{stddev:.3f}",
            'Min': f"{min_val:.3f}",
            'Max': f"{max_val:.3f}"
        })

        # Draw a yellow circle using OpenCV's anti-aliased method
        draw_circle(image_display, (x_original, y_original), circle_diameter, color=(0, 255, 255))

# File selection
file_path = input("Enter the DICOM file path: ")
image, dicom_data = load_image(file_path)

# Pixel spacing
pixel_spacing = float(dicom_data.PixelSpacing[0])  # Assuming square pixels

# Normalize image for display
image_display = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Convert grayscale to BGR for color visualization
if len(image_display.shape) == 2:  # Check if the image is grayscale
    image_display = cv2.cvtColor(image_display, cv2.COLOR_GRAY2BGR)

# Display the image
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", click_event)

print("Instructions:")
print(" - Click on the image to select points.")
print(" - Use '+' to increase the circle diameter.")
print(" - Use '-' to decrease the circle diameter.")
print(" - Use 'z' to zoom in, 'x' to zoom out.")
print(" - Press 'q' to finish and save results.")

while True:
    # Resize the image for zoom
    zoomed_image = cv2.resize(image_display, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Image", zoomed_image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('+'):
        circle_diameter += 1.0
        print(f"Circle diameter increased: {circle_diameter:.1f} pixels")
    elif key == ord('-') and circle_diameter > 1:
        circle_diameter -= 1.0
        print(f"Circle diameter decreased: {circle_diameter:.1f} pixels")
    elif key == ord('z'):  # Zoom in
        zoom_factor *= 1.1
        print(f"Zoomed in. Current zoom factor: {zoom_factor:.2f}")
    elif key == ord('x'):  # Zoom out
        zoom_factor /= 1.1
        print(f"Zoomed out. Current zoom factor: {zoom_factor:.2f}")
    elif key == ord('q'):  # Quit the program
        break

cv2.destroyAllWindows()

# Save the results to an Excel file
output_file = os.path.join(os.getcwd(), "analysis_results.xlsx")
df = pd.DataFrame(results)
df.to_excel(output_file, index=False)
print(f"Results saved to: {output_file}")
