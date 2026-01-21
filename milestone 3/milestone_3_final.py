# =========================
# IMPORTS
# =========================
import os, cv2, torch, numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from collections import defaultdict, Counter
import matplotlib.pyplot as plt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# DATASET
# =========================
class ForensicDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx], 0)
        img = self.transform(img)
        return img, self.labels[idx], os.path.basename(self.paths[idx])

# =========================
# CNN MODEL
# =========================
class TraceFinderCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128,num_classes)
        )

    def forward(self,x):
        fmap = self.features(x)
        x = self.gap(fmap).view(x.size(0),-1)
        return self.fc(x), fmap

# =========================
# LOAD DATA (IMAGE-WISE SPLIT)
# =========================
def load_data(root):
    paths, labels = [], []
    class_map = {}
    for idx, cls in enumerate(sorted(os.listdir(root))):
        class_map[cls] = idx
        for f in os.listdir(os.path.join(root, cls)):
            paths.append(os.path.join(root, cls, f))
            labels.append(idx)

    # IMAGE-WISE split
    image_ids = [os.path.basename(p).split("_patch_")[0] for p in paths]
    unique_ids = list(set(image_ids))
    train_ids, temp_ids = train_test_split(unique_ids, test_size=0.3, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

    def filter(ids):
        return [(p,l) for p,l,i in zip(paths,labels,image_ids) if i in ids]

    train = filter(train_ids)
    val   = filter(val_ids)
    test  = filter(test_ids)

    return train, val, test, list(class_map.keys())

# =========================
# TRAINING
# =========================
def train_epoch(model, loader, opt, loss_fn):
    model.train()
    correct, total = 0, 0
    for x,y,_ in loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        out,_ = model(x)
        loss = loss_fn(out,y)
        loss.backward()
        opt.step()
        correct += (out.argmax(1)==y).sum().item()
        total += y.size(0)
    return 100*correct/total

# =========================
# IMAGE-WISE VOTING
# =========================
def evaluate(model, loader):
    model.eval()
    votes = defaultdict(list)
    gt = {}
    with torch.no_grad():
        for x,y,names in loader:
            out,_ = model(x.to(DEVICE))
            preds = out.argmax(1).cpu().numpy()
            for p,l,n in zip(preds,y,names):
                img_id = n.split("_patch_")[0]
                votes[img_id].append(p)
                gt[img_id] = l.item()

    final_preds, final_gt = [], []
    for img in votes:
        final_preds.append(Counter(votes[img]).most_common(1)[0][0])
        final_gt.append(gt[img])

    acc = accuracy_score(final_gt, final_preds)*100
    f1 = f1_score(final_gt, final_preds, average="weighted")
    return acc, f1, final_gt, final_preds

# =========================
# GRAD-CAM
# =========================
def gradcam(model, img_tensor, class_id):
    model.eval()
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    out, fmap = model(img_tensor)
    score = out[0,class_id]
    model.zero_grad()
    score.backward()
    cam = fmap.grad.mean(dim=1).squeeze().cpu().numpy()
    cam = np.maximum(cam,0)
    cam = cam / cam.max()
    return cam

# =========================
# MAIN
# =========================
def main():
    ROOT = "C:/Forensic_Project/data"
    train, val, test, classes = load_data(ROOT)

    tf_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ColorJitter(brightness=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])

    tf_eval = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])

    train_ds = ForensicDataset(*zip(*train), tf_train)
    val_ds   = ForensicDataset(*zip(*val), tf_eval)
    test_ds  = ForensicDataset(*zip(*test), tf_eval)

    train_ld = DataLoader(train_ds,32,shuffle=True)
    val_ld   = DataLoader(val_ds,32)
    test_ld  = DataLoader(test_ds,32)

    model = TraceFinderCNN(len(classes)).to(DEVICE)
    counts = Counter([l for _,l,_ in train_ds])
    weights = torch.tensor([1/counts[i] for i in range(len(classes))]).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = optim.Adam(model.parameters(), lr=2e-4)
    train_acc_history = []

    for e in range(20):
        acc = train_epoch(model, train_ld, opt, loss_fn)
        train_acc_history.append(acc)
        print(f"Epoch {e+1}: Train Acc {acc:.2f}%")



    plt.figure()
    plt.plot(train_acc_history, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Train Accuracy (%)")
    plt.title("Training Accuracy Curve")
    plt.grid(True)
    plt.savefig("training_curves.png")
    plt.close()

    acc,f1,gt,pred = evaluate(model,test_ld)
    print(f"\nFINAL IMAGE-WISE ACCURACY: {acc:.2f}%")
    print(f"F1-Score: {f1:.4f}")
    print("Confusion Matrix:\n",confusion_matrix(gt,pred))

if __name__ == "__main__":
    main()
