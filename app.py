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
st.title("CT Image Analyzer (Two Images)")

# File uploaders for two images
st.subheader("Upload the First DICOM File:")
first_file = st.file_uploader("First DICOM File", type=["dcm", "IMA"], key="first")

st.subheader("Upload the Second DICOM File:")
second_file = st.file_uploader("Second DICOM File", type=["dcm", "IMA"], key="second")

if first_file and second_file:
    # Load both images
    first_image, first_dicom = load_dicom(first_file)
    second_image, second_dicom = load_dicom(second_file)

    # Normalize and prepare images for display
    first_display = cv2.normalize(first_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    second_display = cv2.normalize(second_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    first_display = cv2.cvtColor(first_display, cv2.COLOR_GRAY2BGR)
    second_display = cv2.cvtColor(second_display, cv2.COLOR_GRAY2BGR)

    first_pil = Image.fromarray(first_display)
    second_pil = Image.fromarray(second_display)

    # Circle diameter slider
    circle_diameter = st.slider("Adjust Circle Diameter (pixels):", min_value=1, max_value=50, value=9, step=1)

    st.subheader("Draw a circle on the First Image to select ROI:")
    first_canvas = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=2,
        stroke_color="#ff0000",
        background_image=first_pil,
        update_streamlit=True,
        height=first_display.shape[0],
        width=first_display.shape[1],
        drawing_mode="circle",
        key="first_canvas",
    )

    st.subheader("Draw a circle on the Second Image to select ROI:")
    second_canvas = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=2,
        stroke_color="#ff0000",
        background_image=second_pil,
        update_streamlit=True,
        height=second_display.shape[0],
        width=second_display.shape[1],
        drawing_mode="circle",
        key="second_canvas",
    )

    def process_canvas(canvas_result, image, dicom_data, image_label):
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
            pixel_spacing = float(dicom_data.PixelSpacing[0])
            area_mm2 = area_pixels * (pixel_spacing ** 2)
            mean = np.mean(pixels)
            stddev = np.std(pixels)
            min_val = np.min(pixels)
            max_val = np.max(pixels)

            # Display results
            st.write(f"**{image_label} ROI Metrics:**")
            st.write(f"- Area: {area_mm2:.2f} mm²")
            st.write(f"- Mean Intensity: {mean:.2f}")
            st.write(f"- Standard Deviation: {stddev:.2f}")
            st.write(f"- Min Intensity: {min_val:.2f}")
            st.write(f"- Max Intensity: {max_val:.2f}")

            return {
                "Image": image_label,
                "Area (mm²)": area_mm2,
                "Mean Intensity": mean,
                "Standard Deviation": stddev,
                "Min Intensity": min_val,
                "Max Intensity": max_val,
            }

    # Process both canvases
    results = []
    if first_canvas:
        result = process_canvas(first_canvas, first_image, first_dicom, "First Image")
        if result:
            results.append(result)

    if second_canvas:
        result = process_canvas(second_canvas, second_image, second_dicom, "Second Image")
        if result:
            results.append(result)

    # Save results as Excel
    if results:
        st.subheader("Download Results")
        results_df = pd.DataFrame(results)
        st.download_button(
            label="Download Results as Excel",
            data=results_df.to_excel(index=False, engine='openpyxl'),
            file_name="analysis_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
