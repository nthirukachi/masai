import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Definition (Copied to avoid import execution)
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

# Data
print("Loading validation data...")
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
val_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

# Validation Function
def validate(model, loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / total, 100 * correct / total

# Check Run A
print("Checking Run A...")
model_a = SimpleMLP().to(device)
try:
    if os.path.exists("outputs/Run_A_HighLR_best.pth"):
        state_dict = torch.load("outputs/Run_A_HighLR_best.pth", map_location=device, weights_only=True)
        model_a.load_state_dict(state_dict)
        loss_a, acc_a = validate(model_a, val_loader)
        print(f"Run_A_HighLR_Best: Loss={loss_a:.4f}, Acc={acc_a:.2f}%")
    else:
        print("Run A Log: Best model file not found in outputs/")
except Exception as e:
    print(f"Error loading Run A: {e}")

# Check Run B
print("Checking Run B...")
model_b = SimpleMLP().to(device)
try:
    if os.path.exists("outputs/Run_B_LowLR_best.pth"):
        state_dict = torch.load("outputs/Run_B_LowLR_best.pth", map_location=device, weights_only=True)
        model_b.load_state_dict(state_dict)
        loss_b, acc_b = validate(model_b, val_loader)
        print(f"Run_B_LowLR_Best: Loss={loss_b:.4f}, Acc={acc_b:.2f}%")
    else:
        print("Run B Log: Best model file not found in outputs/")
except Exception as e:
    print(f"Error loading Run B: {e}")
