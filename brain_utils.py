import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io
from fpdf import FPDF

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

    # Patient info
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Brain Tumor Detection Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Patient Name: {name}", ln=True)
    pdf.cell(200, 10, txt=f"Age: {age}", ln=True)
    pdf.cell(200, 10, txt=f"Gender: {gender}", ln=True)
    pdf.cell(200, 10, txt=f"Prediction Confidence: {confidence:.2f}%", ln=True)
    pdf.ln(10)

   
    temp_img_path = "temp_heatmap.png"
    with open(temp_img_path, "wb") as f:
        f.write(heatmap_buf.read())

    # Insert heatmap into PDF
    pdf.image(temp_img_path, x=10, y=80, w=180)

    # Save final PDF to memory
    pdf_output = io.BytesIO()
    pdf.output(pdf_output, 'F')
    pdf_output.seek(0)

    # Remove temp file
    import os
    os.remove(temp_img_path)

    return pdf_output.getvalue()
