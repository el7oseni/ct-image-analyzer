import streamlit as st
import cv2
import numpy as np
import pandas as pd
import pydicom
import io
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Set page config
st.set_page_config(page_title="DICOM Image Analyzer", layout="wide")

# Initialize session state variables
if 'results' not in st.session_state:
    st.session_state.results = []
if 'circle_diameter' not in st.session_state:
    st.session_state.circle_diameter = 9
if 'zoom_factor' not in st.session_state:
    st.session_state.zoom_factor = 1.0
if 'last_click1' not in st.session_state:
    st.session_state.last_click1 = None
if 'last_click2' not in st.session_state:
    st.session_state.last_click2 = None

def load_dicom(uploaded_file):
    try:
        if uploaded_file is not None:
            bytes_data = uploaded_file.getvalue()
            dicom_data = pydicom.dcmread(io.BytesIO(bytes_data))
            image = dicom_data.pixel_array.astype(np.float32)
            
            rescale_slope = getattr(dicom_data, 'RescaleSlope', 1)
            rescale_intercept = getattr(dicom_data, 'RescaleIntercept', 0)
            image = (image * rescale_slope) + rescale_intercept
            
            return image, dicom_data
    except Exception as e:
        st.error(f"Error loading DICOM file: {str(e)}")
        return None, None
    return None, None

def analyze_point(image, dicom_data, x, y):
    try:
        mask = np.zeros_like(image, dtype=np.uint8)
        y_indices, x_indices = np.ogrid[:image.shape[0], :image.shape[1]]
        distance_from_center = np.sqrt((x_indices - x)**2 + (y_indices - y)**2)
        mask[distance_from_center <= st.session_state.circle_diameter / 2] = 1

        pixels = image[mask == 1]
        area_pixels = np.sum(mask)
        pixel_spacing = float(dicom_data.PixelSpacing[0])
        area_mm2 = area_pixels * (pixel_spacing**2)
        mean = np.mean(pixels)
        stddev = np.std(pixels)
        min_val = np.min(pixels)
        max_val = np.max(pixels)

        return {
            'Area (mm²)': f"{area_mm2:.3f}",
            'Mean': f"{mean:.3f}",
            'StdDev': f"{stddev:.3f}",
            'Min': f"{min_val:.3f}",
            'Max': f"{max_val:.3f}"
        }
    except Exception as e:
        st.error(f"Error analyzing point: {str(e)}")
        return None

def draw_circle_on_image(image, x, y, diameter, color=(0, 255, 255)):
    try:
        image_copy = image.copy()
        cv2.circle(image_copy, 
                  (int(x), int(y)), 
                  int(diameter/2), 
                  color, 
                  1, 
                  lineType=cv2.LINE_AA)
        return image_copy
    except Exception as e:
        st.error(f"Error drawing circle: {str(e)}")
        return image

# Streamlit UI
st.title("DICOM Image Analyzer")

# File uploaders
col1, col2 = st.columns(2)
with col1:
    file1 = st.file_uploader("Upload first DICOM file", type=['dcm', 'IMA'])
with col2:
    file2 = st.file_uploader("Upload second DICOM file", type=['dcm', 'IMA'])

# Controls
st.sidebar.header("Controls")
st.session_state.circle_diameter = st.sidebar.slider("Circle Diameter", 1, 20, st.session_state.circle_diameter)
st.session_state.zoom_factor = st.sidebar.slider("Zoom Factor", 0.5, 3.0, st.session_state.zoom_factor)

if st.sidebar.button("Add Blank Row"):
    st.session_state.results.append({
        'Image': '',
        'Point': '',
        'Area (mm²)': '',
        'Mean': '',
        'StdDev': '',
        'Min': '',
        'Max': ''
    })

if file1 is not None and file2 is not None:
    # Load images
    image1, dicom_data1 = load_dicom(file1)
    image2, dicom_data2 = load_dicom(file2)

    if image1 is not None and image2 is not None:
        # Normalize and prepare images for display
        image_display1 = cv2.normalize(image1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image_display2 = cv2.normalize(image2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Display images side by side with canvas
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Image 1")
            canvas_result1 = st_canvas(
                fill_color="rgba(255, 255, 0, 0.3)",
                stroke_width=1,
                stroke_color="#yellow",
                background_image=Image.fromarray(image_display1),
                drawing_mode="point",
                key="canvas1",
                width=image_display1.shape[1],
                height=image_display1.shape[0]
            )
            
            if canvas_result1.json_data is not None and len(canvas_result1.json_data["objects"]) > 0:
                last_point = canvas_result1.json_data["objects"][-1]
                current_click = (int(last_point["left"]), int(last_point["top"]))
                
                if st.session_state.last_click1 != current_click:
                    st.session_state.last_click1 = current_click
                    x1, y1 = current_click
                    
                    # Draw circle and analyze point
                    marked_image1 = draw_circle_on_image(image_display1, x1, y1, st.session_state.circle_diameter)
                    results = analyze_point(image1, dicom_data1, x1, y1)
                    if results:
                        results['Image'] = "Image 1"
                        results['Point'] = f"({x1}, {y1})"
                        st.session_state.results.append(results)

        with col2:
            st.header("Image 2")
            canvas_result2 = st_canvas(
                fill_color="rgba(255, 255, 0, 0.3)",
                stroke_width=1,
                stroke_color="#yellow",
                background_image=Image.fromarray(image_display2),
                drawing_mode="point",
                key="canvas2",
                width=image_display2.shape[1],
                height=image_display2.shape[0]
            )
            
            if canvas_result2.json_data is not None and len(canvas_result2.json_data["objects"]) > 0:
                last_point = canvas_result2.json_data["objects"][-1]
                current_click = (int(last_point["left"]), int(last_point["top"]))
                
                if st.session_state.last_click2 != current_click:
                    st.session_state.last_click2 = current_click
                    x2, y2 = current_click
                    
                    # Draw circle and analyze point
                    marked_image2 = draw_circle_on_image(image_display2, x2, y2, st.session_state.circle_diameter)
                    results = analyze_point(image2, dicom_data2, x2, y2)
                    if results:
                        results['Image'] = "Image 2"
                        results['Point'] = f"({x2}, {y2})"
                        st.session_state.results.append(results)

        # Display results
        if st.session_state.results:
            st.header("Results")
            df = pd.DataFrame(st.session_state.results)
            st.dataframe(df)
            
            # Download buttons
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download results as CSV",
                data=csv,
                file_name="analysis_results.csv",
                mime="text/csv"
            )
            
            # Excel download
            excel_buffer = io.BytesIO()
            df.to_excel(excel_buffer, index=False)
            excel_data = excel_buffer.getvalue()
            st.download_button(
                label="Download results as Excel",
                data=excel_data,
                file_name="analysis_results.xlsx",
                mime="application/vnd.ms-excel"
            )

        if st.button("Clear Results"):
            st.session_state.results = []
            st.experimental_rerun()

else:
    st.info("Please upload both DICOM files to begin analysis.")
