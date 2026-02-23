import streamlit as st
import cv2
import numpy as np
from PIL import Image

def calculate_metrics(pred_edges, gt_edges):
    """Calculates Precision, Recall, and F1-score."""
    pred_bool = pred_edges > 0
    gt_bool = gt_edges > 0

    TP = np.sum(pred_bool & gt_bool)
    FP = np.sum(pred_bool & ~gt_bool)
    FN = np.sum(~pred_bool & gt_bool)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1_score

def generate_pseudo_ground_truth(image_array):
    """Generates a baseline edge map to act as a fake answer key."""
    # We use a standard blur and Canny with median thresholds to create a "decent" baseline
    blurred = cv2.GaussianBlur(image_array, (5, 5), 0)
    # Using Otsu's method to find a decent dynamic threshold
    high_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5 * high_thresh
    
    pseudo_gt = cv2.Canny(blurred, int(low_thresh), int(high_thresh))
    return pseudo_gt

# --- App UI Layout ---
st.set_page_config(layout="wide")
st.title("🎯 Edge Detection Evaluator & Parameter Tuner")
st.write("Upload an image to start. You can upload a manual ground truth, or let the app generate a synthetic one for testing!")

# --- File Uploaders ---
col1, col2 = st.columns(2)
with col1:
    img_file = st.file_uploader("1. Upload Original Image (Required)", type=['png', 'jpg', 'jpeg'])
with col2:
    gt_file = st.file_uploader("2. Upload Ground Truth (Optional)", type=['png', 'jpg', 'jpeg'])

if img_file is not None:
    # Convert uploaded image to OpenCV grayscale format
    img_pil = Image.open(img_file).convert('L') 
    img_array = np.array(img_pil)

    # Determine Ground Truth
    if gt_file is not None:
        gt_pil = Image.open(gt_file).convert('L')
        gt_array = np.array(gt_pil)
        _, gt_binary = cv2.threshold(gt_array, 127, 255, cv2.THRESH_BINARY)
        st.success("✅ Using your uploaded Ground Truth.")
    else:
        gt_binary = generate_pseudo_ground_truth(img_array)
        st.warning("⚠️ No Ground Truth uploaded. Using an auto-generated 'Pseudo-Ground Truth' for demonstration purposes.")

    st.markdown("---")
    st.subheader("⚙️ Tune Parameters")
    
    # --- Interactive Sliders ---
    slider_col1, slider_col2 = st.columns(2)
    with slider_col1:
        low_thresh = st.slider("Canny Low Threshold", min_value=0, max_value=255, value=50, step=5)
    with slider_col2:
        high_thresh = st.slider("Canny High Threshold", min_value=0, max_value=255, value=150, step=5)

    # --- Process Edge Detection ---
    blurred_img = cv2.GaussianBlur(img_array, (5, 5), 0)
    predicted_edges = cv2.Canny(blurred_img, low_thresh, high_thresh)

    # Calculate metrics
    precision, recall, f1 = calculate_metrics(predicted_edges, gt_binary)

    # --- Display Metrics ---
    st.markdown("### 📊 Live Evaluation Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric(label="Precision", value=f"{precision:.4f}")
    m2.metric(label="Recall", value=f"{recall:.4f}")
    m3.metric(label="🏆 F1-Score", value=f"{f1:.4f}")

    # --- Display Images ---
    st.markdown("### 👁️ Visual Comparison")
    img_col1, img_col2, img_col3 = st.columns(3)
    
    with img_col1:
        st.image(img_array, caption="Original Image", use_container_width=True)
    with img_col2:
        st.image(gt_binary, caption="Ground Truth (Answer Key)", use_container_width=True)
    with img_col3:
        st.image(predicted_edges, caption=f"Predicted Edges (Low:{low_thresh}, High:{high_thresh})", use_container_width=True)
else:
    st.info("👆 Please upload an original image to start.")