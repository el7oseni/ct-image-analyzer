import streamlit as st
import cv2
import numpy as np
import pandas as pd
import pydicom
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# Function to load DICOM image
def load_dicom(file):
    dicom_data = pydicom.dcmread(file)
    image = dicom_data.pixel_array.astype(np.float32)
    rescale_slope = getattr(dicom_data, 'RescaleSlope', 1)
    rescale_intercept = getattr(dicom_data, 'RescaleIntercept', 0)
    image = (image * rescale_slope) + rescale_intercept
    return image, dicom_data

# Streamlit app
st.title("CT Image Analyzer")

# File uploader
uploaded_file = st.file_uploader("Upload a DICOM file", type=["dcm", "IMA"])

if uploaded_file:
    image, dicom_data = load_dicom(uploaded_file)
    pixel_spacing = float(dicom_data.PixelSpacing[0])
    image_display = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image_display = cv2.cvtColor(image_display, cv2.COLOR_GRAY2BGR)
    image_pil = Image.fromarray(image_display)

    # Display image with drawable canvas
    st.write("Draw a circle on the image to select ROI:")
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=2,
        stroke_color="#ff0000",
        background_image=image_pil,
        update_streamlit=True,
        height=image_display.shape[0],
        width=image_display.shape[1],
        drawing_mode="circle",
        key="canvas",
    )

    # Process the drawn circle
    if canvas_result.json_data and canvas_result.json_data["objects"]:
        obj = canvas_result.json_data["objects"][-1]
        x, y, radius = obj["left"], obj["top"], obj["radius"]

        # Create a mask for the circular region
        mask = np.zeros_like(image, dtype=np.uint8)
        y_indices, x_indices = np.ogrid[:image.shape[0], :image.shape[1]]
        distance_from_center = np.sqrt((x_indices - x) ** 2 + (y_indices - y) ** 2)
        mask[distance_from_center <= radius] = 1

        # Extract pixel values within the circle
        pixels = image[mask == 1]

        # Calculate metrics
        area_pixels = np.sum(mask)
        area_mm2 = area_pixels * (pixel_spacing ** 2)
        mean = np.mean(pixels)
        stddev = np.std(pixels)
        min_val = np.min(pixels)
        max_val = np.max(pixels)

        # Display results
        st.write(f"**Selected ROI Metrics:**")
        st.write(f"- Area: {area_mm2:.2f} mm²")
        st.write(f"- Mean Intensity: {mean:.2f}")
        st.write(f"- Standard Deviation: {stddev:.2f}")
        st.write(f"- Min Intensity: {min_val:.2f}")
        st.write(f"- Max Intensity: {max_val:.2f}")

        # Save results to DataFrame
        results = pd.DataFrame([{
            "Area (mm²)": area_mm2,
            "Mean Intensity": mean,
            "Standard Deviation": stddev,
            "Min Intensity": min_val,
            "Max Intensity": max_val,
        }])

        # Download results as Excel
        st.download_button(
            label="Download Results as Excel",
            data=results.to_excel(index=False, engine='openpyxl'),
            file_name="analysis_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
