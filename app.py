import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io
from fpdf import FPDF
import os

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
def load_model():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)  # Tumor vs No Tumor
    model.load_state_dict(torch.load("model.pth", map_location=device))  # Load model weights
    model.to(device)
    model.eval()
    return model

# Grad-CAM implementation
def get_gradcam_heatmap(model, image_tensor):
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    final_conv = model.layer4[1].conv2
    forward_handle = final_conv.register_forward_hook(forward_hook)
    backward_handle = final_conv.register_backward_hook(backward_hook)

    output = model(image_tensor)
    pred_class = output.argmax(dim=1)
    loss = output[:, pred_class]
    model.zero_grad()
    loss.backward()

    grad = gradients[0].detach().cpu().numpy()[0]
    act = activations[0].detach().cpu().numpy()[0]

    weights = np.mean(grad, axis=(1, 2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)

    # Convert to heatmap image
    plt.imshow(cam, cmap='jet')
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    return buf

# Generate PDF report
def generate_report(name, age, gender, confidence, heatmap_buf):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Brain Tumor Detection Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Patient Name: {name}", ln=True)
    pdf.cell(200, 10, txt=f"Age: {age}", ln=True)
    pdf.cell(200, 10, txt=f"Gender: {gender}", ln=True)
    pdf.cell(200, 10, txt=f"Prediction Confidence: {confidence:.2f}%", ln=True)
    pdf.ln(10)

    # Save heatmap image temporarily
    temp_path = "temp_heatmap.png"
    with open(temp_path, "wb") as f:
        f.write(heatmap_buf.read())

    pdf.image(temp_path, x=10, y=80, w=180)

    
    pdf_bytes = pdf.output(dest='S').encode('latin1')

    os.remove(temp_path)

    return pdf_bytes

# Main Streamlit app
def main():
    st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
    st.title("🧠 Brain Tumor Detection App")

    model = load_model()

    name = st.text_input("Enter Patient Name")
    age = st.number_input("Enter Age", min_value=0, max_value=120)
    gender = st.radio("Select Gender", ["Male", "Female", "Other"])

    uploaded_file = st.file_uploader("Upload Brain MRI Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        image_tensor = preprocess(image).unsqueeze(0).to(device)

        if st.button("Analyze"):
            with st.spinner("Analyzing..."):
                outputs = model(image_tensor)
                _, predicted = torch.max(outputs, 1)
                confidence = torch.softmax(outputs, dim=1)[0][predicted].item() * 100

                label = "Tumor" if predicted.item() == 1 else "No Tumor"
                st.success(f"Prediction: **{label}** ({confidence:.2f}% confidence)")

                heatmap_buf = get_gradcam_heatmap(model, image_tensor)

                st.image(heatmap_buf, caption="Grad-CAM Heatmap", use_column_width=True)

                pdf_bytes = generate_report(name, age, gender, confidence, heatmap_buf)

                st.download_button(
                    label="📄 Download Report as PDF",
                    data=pdf_bytes,
                    file_name="tumor_detection_report.pdf",
                    mime="application/pdf"
                )

if __name__ == "__main__":
    main()
