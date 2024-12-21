import streamlit as st
import cv2
import numpy as np
import pandas as pd
import pydicom
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from io import BytesIO

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

# File uploaders
st.write("**Upload the First DICOM File:**")
uploaded_file1 = st.file_uploader("First DICOM File", type=["dcm", "IMA"], key="file1")

st.write("**Upload the Second DICOM File:**")
uploaded_file2 = st.file_uploader("Second DICOM File", type=["dcm", "IMA"], key="file2")

if uploaded_file1 and uploaded_file2:
    # Load the first DICOM image
    image1, dicom_data1 = load_dicom(uploaded_file1)
    pixel_spacing1 = float(dicom_data1.PixelSpacing[0])
    image_display1 = cv2.normalize(image1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image_display1 = cv2.cvtColor(image_display1, cv2.COLOR_GRAY2BGR)
    image_pil1 = Image.fromarray(image_display1)

    # Load the second DICOM image
    image2, dicom_data2 = load_dicom(uploaded_file2)
    pixel_spacing2 = float(dicom_data2.PixelSpacing[0])
    image_display2 = cv2.normalize(image2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image_display2 = cv2.cvtColor(image_display2, cv2.COLOR_GRAY2BGR)
    image_pil2 = Image.fromarray(image_display2)

    # Display both images with drawable canvases
    st.write("**Draw a circle on the First Image to select ROI:**")
    canvas_result1 = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=2,
        stroke_color="#ff0000",
        background_image=image_pil1,
        update_streamlit=True,
        height=image_display1.shape[0],
        width=image_display1.shape[1],
        drawing_mode="circle",
        key="canvas1",
    )

    st.write("**Draw a circle on the Second Image to select ROI:**")
    canvas_result2 = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=2,
        stroke_color="#ff0000",
        background_image=image_pil2,
        update_streamlit=True,
        height=image_display2.shape[0],
        width=image_display2.shape[1],
        drawing_mode="circle",
        key="canvas2",
    )

    results = []

    # Process ROI for the first image
    if canvas_result1.json_data and canvas_result1.json_data["objects"]:
        obj = canvas_result1.json_data["objects"][-1]
        x, y, radius = obj["left"], obj["top"], obj["radius"]

        mask = np.zeros_like(image1, dtype=np.uint8)
        y_indices, x_indices = np.ogrid[:image1.shape[0], :image1.shape[1]]
        distance_from_center = np.sqrt((x_indices - x) ** 2 + (y_indices - y) ** 2)
        mask[distance_from_center <= radius] = 1

        pixels = image1[mask == 1]
        area_pixels = np.sum(mask)
        area_mm2 = area_pixels * (pixel_spacing1 ** 2)
        mean = np.mean(pixels)
        stddev = np.std(pixels)
        min_val = np.min(pixels)
        max_val = np.max(pixels)

        st.write(f"**First Image ROI Metrics:**")
        st.write(f"- Area: {area_mm2:.2f} mm²")
        st.write(f"- Mean Intensity: {mean:.2f}")
        st.write(f"- Standard Deviation: {stddev:.2f}")
        st.write(f"- Min Intensity: {min_val:.2f}")
        st.write(f"- Max Intensity: {max_val:.2f}")

        results.append({
            "Image": "First Image",
            "Area (mm²)": area_mm2,
            "Mean Intensity": mean,
            "Standard Deviation": stddev,
            "Min Intensity": min_val,
            "Max Intensity": max_val,
        })

    # Process ROI for the second image
    if canvas_result2.json_data and canvas_result2.json_data["objects"]:
        obj = canvas_result2.json_data["objects"][-1]
        x, y, radius = obj["left"], obj["top"], obj["radius"]

        mask = np.zeros_like(image2, dtype=np.uint8)
        y_indices, x_indices = np.ogrid[:image2.shape[0], :image2.shape[1]]
        distance_from_center = np.sqrt((x_indices - x) ** 2 + (y_indices - y) ** 2)
        mask[distance_from_center <= radius] = 1

        pixels = image2[mask == 1]
        area_pixels = np.sum(mask)
        area_mm2 = area_pixels * (pixel_spacing2 ** 2)
        mean = np.mean(pixels)
        stddev = np.std(pixels)
        min_val = np.min(pixels)
        max_val = np.max(pixels)

        st.write(f"**Second Image ROI Metrics:**")
        st.write(f"- Area: {area_mm2:.2f} mm²")
        st.write(f"- Mean Intensity: {mean:.2f}")
        st.write(f"- Standard Deviation: {stddev:.2f}")
        st.write(f"- Min Intensity: {min_val:.2f}")
        st.write(f"- Max Intensity: {max_val:.2f}")

        results.append({
            "Image": "Second Image",
            "Area (mm²)": area_mm2,
            "Mean Intensity": mean,
            "Standard Deviation": stddev,
            "Min Intensity": min_val,
            "Max Intensity": max_val,
        })

    # Save results to Excel
    if results:
        df = pd.DataFrame(results)
        buffer = BytesIO()
        df.to_excel(buffer, index=False, engine="openpyxl")
        buffer.seek(0)

        st.download_button(
            label="Download Results as Excel",
            data=buffer,
            file_name="analysis_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
