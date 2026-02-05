import streamlit as st
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import datetime
import base64
import os

# Define constants
SMOOTH = 100

# Add background image using CSS
background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://i.imgur.com/61K5Yv8.jpg");
    background-size: cover;
    background-position: center;  
    background-repeat: no-repeat;
}
</style>
"""
st.markdown(background_image, unsafe_allow_html=True)

# Define custom metrics and loss functions
def dice_coef(y_true, y_pred):
    y_truef = K.flatten(y_true)
    y_predf = K.flatten(y_pred)
    intersection = K.sum(y_truef * y_predf)
    return (2. * intersection + SMOOTH) / (K.sum(y_truef) + K.sum(y_predf) + SMOOTH)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + SMOOTH) / (sum_ - intersection + SMOOTH)
    return jac

def jac_distance(y_true, y_pred):
    return -iou(y_true, y_pred)

# Load the trained model
model = load_model('unet_brain_mri_seg.hdf5', custom_objects={'dice_coef_loss': dice_coef_loss, 'iou': iou, 'dice_coef': dice_coef})

# Function to perform segmentation on the uploaded image
def predict_segmentation(image):
    img_resized = cv2.resize(image, (256, 256))
    img_normalized = img_resized / 255.0
    img = img_normalized[np.newaxis, :, :, :]
    pred = model.predict(img)
    return pred

# Function to generate PDF report
def generate_pdf(image, segmentation, tumor_detected, total_pixels, tumor_pixels):
    filename = f"segmentation_report_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pdf"
    pdf_dir = "pdf_reports"
    os.makedirs(pdf_dir, exist_ok=True)  # Create the directory if it doesn't exist
    pdf_path = os.path.join(pdf_dir, filename)
    c = canvas.Canvas(pdf_path, pagesize=letter)
    
    # Project name in bold
    project_name = "BrainTune: Smart Brain Tumor Detection and Localization"
    c.setFont("Helvetica-Bold", 14)
    # Calculate the width of the text
    text_width = c.stringWidth(project_name, "Helvetica-Bold", 14)
    # Calculate x position to center the text
    x_position = (letter[0] - text_width) / 2
    c.drawString(x_position, 750, project_name)
    
    c.setFont("Helvetica", 10)
    c.drawString(100, 730, "Generated on: " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    c.drawString(100, 710, f"Tumor Detected: {'Yes' if tumor_detected else 'No'}")
    c.drawString(100, 690, f"Total Pixels: {total_pixels}")
    c.drawString(100, 670, f"Tumor Pixels: {tumor_pixels}")
    
    # Draw original image
    original_img = Image.fromarray(image)
    original_img_path = "original_img.png"
    original_img.save(original_img_path)
    
    # Section title in bold
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, 640, "Original Image:")
    
    c.setFont("Helvetica", 10)
    c.drawImage(original_img_path, 100, 360, width=300, height=270)
    
    # Draw segmentation prediction
    segmentation_img = Image.fromarray((np.squeeze(segmentation) > 0.5).astype(np.uint8) * 255)
    segmentation_img_path = "segmentation_img.png"
    segmentation_img.save(segmentation_img_path)
    
    # Section title in bold
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, 320, "Segmentation Prediction:")
    
    c.setFont("Helvetica", 10)
    c.drawImage(segmentation_img_path, 100, 40, width=300, height=270)
    
    c.drawString(100, 20, "Developed by: Engr. Ameer Hamza (70111624), Engr. Omer Faizan (70112244)")
    
    c.save()
    return pdf_path

# Function to convert PDF to data URL
def get_pdf_data_uri(pdf_path):
    with open(pdf_path, "rb") as f:
        data = f.read()
        data_uri = base64.b64encode(data).decode("utf-8")
        return data_uri

# Streamlit app
def main():
    st.title("BrainTune: Smart Brain Tumor Detection and Localization")
    st.write("Upload a brain MRI image (TIFF format) to get the segmentation prediction.")

    uploaded_image = st.file_uploader("Choose an image...", type=["tif", "tiff"])

    if uploaded_image is not None:
        # Read TIFF image using PIL and convert to numpy array
        image = np.array(Image.open(uploaded_image))

        st.image(image, caption='Uploaded Image', use_column_width=True, channels="BGR")

        if st.button('Predict'):
            segmentation = predict_segmentation(image)
            tumor_detected = np.any(np.squeeze(segmentation) > 0.5)
            total_pixels = image.shape[0] * image.shape[1]
            tumor_pixels = np.sum(np.squeeze(segmentation) > 0.5)
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(image, cmap='gray')
            axes[0].set_title('Original Image')
            axes[1].imshow(np.squeeze(segmentation) > 0.5, cmap='gray')
            axes[1].set_title(f'Segmentation Prediction\nTumor Detected: {"Yes" if tumor_detected else "No"}\nTotal Pixels: {total_pixels}\nTumor Pixels: {tumor_pixels}')
            st.pyplot(fig)

            # Generate PDF report
            pdf_filename = generate_pdf(image, segmentation, tumor_detected, total_pixels, tumor_pixels)
            pdf_data_uri = get_pdf_data_uri(pdf_filename)
            st.markdown(f"Download the PDF report [here](data:application/pdf;base64,{pdf_data_uri})", unsafe_allow_html=True)
            st.markdown("---")
            
    # Add footer
    st.markdown(
        """
        <div class="footer">
        <p style="font-weight:bold;">Developed by: Engr. Ameer Hamza (70111624), Engr. Omer Faizan (70112244)</p>
        </div>
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            color: black;
            text-align: center;
            padding: 5px;
            font-size: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
