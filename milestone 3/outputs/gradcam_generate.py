import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn as nn

# -----------------------------
# DEVICE
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# MODEL (MUST MATCH TRAINING)
class TraceFinderCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.gap  = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# -----------------------------
# LOAD MODEL
# -----------------------------
num_classes = 5   # change ONLY if needed
model = TraceFinderCNN(num_classes).to(DEVICE)
model.load_state_dict(torch.load("best_forensic_cnn.pth", map_location=DEVICE))
model.eval()

# -----------------------------
# LOAD IMAGE
# -----------------------------
img_path = "sample_test_image2.png"   # <-- CHANGE THIS IMAGE NAME ONLY

img = cv2.imread(img_path, 0)
img = cv2.resize(img, (224,224))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

img_tensor = transform(img).unsqueeze(0).to(DEVICE)

# -----------------------------
# GRAD-CAM
# -----------------------------

# -----------------------------
# GRAD-CAM (USING HOOKS)
# -----------------------------
feature_maps = []
gradients = []

def forward_hook(module, input, output):
    feature_maps.append(output)

def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

# Register hooks on last convolution layer
model.conv4.register_forward_hook(forward_hook)
model.conv4.register_backward_hook(backward_hook)

# Forward pass
output = model(img_tensor)
pred_class = output.argmax(dim=1).item()

# Backward pass
score = output[0, pred_class]
model.zero_grad()
score.backward()

# Compute Grad-CAM
fmap = feature_maps[0].squeeze().detach().cpu().numpy()
grad = gradients[0].mean(dim=(1, 2)).detach().cpu().numpy()

cam = np.zeros(fmap.shape[1:], dtype=np.float32)
for i, w in enumerate(grad):
    cam += w * fmap[i]

cam = np.maximum(cam, 0)
cam = cam / (cam.max() + 1e-8)
cam = cv2.resize(cam, (224, 224))


heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(
    cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
    0.6,
    heatmap,
    0.4,
    0
)

# -----------------------------
# SAVE RESULT
# -----------------------------
plt.figure(figsize=(5,5))
plt.imshow(overlay)
plt.axis("off")
plt.title("Grad-CAM Visualization")
plt.savefig("gradcam_output2.png", bbox_inches="tight")
plt.show()

print("✅ Grad-CAM saved as gradcam_output.png")
