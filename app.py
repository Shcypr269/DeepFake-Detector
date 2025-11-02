import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


# Define your CNN architecture

class FraudCNN(nn.Module):
    def __init__(self):
        super(FraudCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# Load Model
model = FraudCNN()
model.load_state_dict(torch.load(r"C:\Users\KIIT\Downloads\Model\cnn_fraud_model.pth", map_location=torch.device("cpu")))
model.eval()

# Image Transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Streamlit UI

st.title("🧠 Image Fraud Detection (CNN)")
st.write("Upload an image to check if it's **FAKE** or **REAL**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "tif"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = transform(image).unsqueeze(0)

    # Prediction
    with torch.no_grad():
        probs = F.softmax(model(img), dim=1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs[0][pred_class].item()

    labels = ["REAL", "FAKE"]
    st.markdown(f"### Prediction: **{labels[pred_class]}** ({confidence:.2f} score)")

