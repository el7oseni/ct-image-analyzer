import streamlit as st
import cv2
import numpy as np
import pandas as pd
import pydicom
from io import BytesIO

# Function to load the DICOM image and apply rescale slope and intercept
def load_image(file):
    dicom_data = pydicom.dcmread(file)
    image = dicom_data.pixel_array.astype(np.float32)
    rescale_slope = getattr(dicom_data, 'RescaleSlope', 1)
    rescale_intercept = getattr(dicom_data, 'RescaleIntercept', 0)
    image = (image * rescale_slope) + rescale_intercept
    return image, dicom_data

# Function to calculate metrics for the selected region
def calculate_metrics(image, center, diameter, pixel_spacing):
    x, y = center
    radius = diameter / 2
    mask = np.zeros_like(image, dtype=np.uint8)
    y_indices, x_indices = np.ogrid[:image.shape[0], :image.shape[1]]
    distance_from_center = np.sqrt((x_indices - x)**2 + (y_indices - y)**2)
    mask[distance_from_center <= radius] = 1

    # Extract pixel values within the mask
    pixels = image[mask == 1]
    area_pixels = np.sum(mask)
    area_mm2 = area_pixels * (pixel_spacing**2)
    mean_intensity = np.mean(pixels)
    stddev_intensity = np.std(pixels)
    min_intensity = np.min(pixels)
    max_intensity = np.max(pixels)

    return {
        "Area (mm²)": round(area_mm2, 3),
        "Mean": round(mean_intensity, 3),
        "StdDev": round(stddev_intensity, 3),
        "Min": round(min_intensity, 3),
        "Max": round(max_intensity, 3),
    }

# Streamlit interface
st.title("CT Image Analyzer")
st.markdown("""
A simple web-based application to analyze CT images.
- **Upload a DICOM file**
- **Select a region for analysis**
- **Download the results**
""")

# File upload section
uploaded_file = st.file_uploader("Upload a DICOM file", type=["dcm", "IMA"])

# Circle diameter input
circle_diameter = st.number_input("Circle Diameter (pixels):", min_value=1.0, value=10.0)

if uploaded_file:
    # Load the DICOM image
    st.write("Loading the DICOM file...")
    image, dicom_data = load_image(uploaded_file)
    pixel_spacing = float(dicom_data.PixelSpacing[0])  # Assuming square pixels

    # Normalize image for display
    image_display = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image_display = cv2.cvtColor(image_display, cv2.COLOR_GRAY2BGR)

    # Display the image
    st.image(image_display, caption="CT Image", channels="BGR", use_column_width=True)

    # Interaction: Select points for analysis
    st.markdown("### Enter the coordinates of the center point for analysis (x, y):")
    coords = st.text_input("Coordinates (comma-separated):", "100, 100")

    if st.button("Analyze"):
        try:
            x, y = map(int, coords.split(","))
            metrics = calculate_metrics(image, (x, y), circle_diameter, pixel_spacing)
            st.markdown("### Analysis Results")
            st.write(metrics)

            # Save results to an Excel file
            results = pd.DataFrame([metrics])
            buffer = BytesIO()
            results.to_excel(buffer, index=False)
            buffer.seek(0)
            st.download_button(
                label="Download Results as Excel",
                data=buffer,
                file_name="analysis_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.error(f"Error: {e}")
