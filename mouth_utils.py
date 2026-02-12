
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from model import MouthCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MouthCNN().to(device)

try:
    model.load_state_dict(torch.load("mouth_cnn.pth", map_location=device))
    model.eval()
except Exception as e:
    print("Warning: mouth_cnn.pth not found. Train the model first.", e)

def preprocess_mouth(mouth_img):
    mouth_img = cv2.cvtColor(mouth_img, cv2.COLOR_BGR2GRAY)
    mouth_img = cv2.resize(mouth_img, (64, 64))
    mouth_img = mouth_img / 255.0
    mouth_img = np.expand_dims(mouth_img, axis=0)
    mouth_img = np.expand_dims(mouth_img, axis=0)
    return torch.tensor(mouth_img, dtype=torch.float32)

def is_yawning(mouth_img):
    tensor = preprocess_mouth(mouth_img).to(device)
    with torch.no_grad():
        output = model(tensor)
        prob = F.softmax(output, dim=1)
        pred = torch.argmax(prob, dim=1).item()
        confidence = prob[0][1].item()

# Only treat as yawning if confidence is high
    return pred == 1 and confidence > 0.85

