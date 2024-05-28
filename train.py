import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from sklearn.model_selection import train_test_split


# Save the trained model
def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

# Load the trained model
def load_model(model, load_path):
    model.load_state_dict(torch.load(load_path))
    model.eval()
    return model

# Perform inference
def inference(model, image, device):
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        yaw_pred, pitch_pred, roll_pred = model(image)
        yaw = torch.sum(yaw_pred.softmax(dim=1) * torch.arange(62).to(device)).item()
        pitch = torch.sum(pitch_pred.softmax(dim=1) * torch.arange(62).to(device)).item()
        roll = torch.sum(roll_pred.softmax(dim=1) * torch.arange(62).to(device)).item()
    return yaw, pitch, roll


# Function to convert TensorFlow tensors to PyTorch tensors
def tf_to_torch(image, label):
    image = image.numpy()  # Convert TensorFlow tensor to a NumPy array
    label = label.numpy()  # Assuming label is already in the required format as an array
    image = torch.tensor(image, dtype=torch.float32)  # Convert to PyTorch tensor
    label = torch.tensor(label, dtype=torch.float32)
    return image.permute(2, 0, 1), label  # Rearrange the dimensions to CxHxW for PyTorch

# Custom dataset class for PyTorch
class TensorFlow300WLPDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, landmarks = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, landmarks

def load_300w_train():
    """
    Loads the 300W-LP dataset, splits it into training and validation sets.

    Returns
    -------
    train_x, train_y, valid_x, valid_y : tuple
        Tuple containing training and validation sets for features and labels.
    """
    # Load the dataset
    data, info = tfds.load('the300w_lp', split='train', shuffle_files=True, with_info=True, data_dir='../../data/300w')

    # Extract images and landmarks
    images = []
    landmarks = []
    for example in tfds.as_numpy(data):
        images.append(example['image'])
        landmarks.append(example['landmarks_2d'])

    images = np.array(images)
    landmarks = np.array(landmarks)

    # Splitting into training and validation sets
    print("splitting...")
    train_x, valid_x, train_y, valid_y = train_test_split(images, landmarks, test_size=0.1)
    print("splitting complete")

    return train_x, train_y, valid_x, valid_y


# Define the head pose estimation model
class HPENet(nn.Module):
    def __init__(self, num_bins=62):
        super(HPENet, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()
        self.fc_yaw = nn.Linear(512, num_bins)
        self.fc_pitch = nn.Linear(512, num_bins)
        self.fc_roll = nn.Linear(512, num_bins)

    def forward(self, x):
        features = self.backbone(x)
        yaw = self.fc_yaw(features)
        pitch = self.fc_pitch(features)
        roll = self.fc_roll(features)
        return yaw, pitch, roll

# Define the loss function
def head_pose_loss(yaw_pred, pitch_pred, roll_pred, yaw_true, pitch_true, roll_true):
    yaw_loss = nn.CrossEntropyLoss()(yaw_pred, yaw_true) + nn.MSELoss()(yaw_pred.softmax(dim=1) * torch.arange(62).to(yaw_pred.device), yaw_true.float())
    pitch_loss = nn.CrossEntropyLoss()(pitch_pred, pitch_true) + nn.MSELoss()(pitch_pred.softmax(dim=1) * torch.arange(62).to(pitch_pred.device), pitch_true.float())
    roll_loss = nn.CrossEntropyLoss()(roll_pred, roll_true) + nn.MSELoss()(roll_pred.softmax(dim=1) * torch.arange(62).to(roll_pred.device), roll_true.float())
    return yaw_loss + pitch_loss + roll_loss

# Training loop
def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for images, labels in dataloader:
        images = images.to(device)
        yaw_true, pitch_true, roll_true = labels[:, 0].long(), labels[:, 1].long(), labels[:, 2].long()
        yaw_true, pitch_true, roll_true = yaw_true.to(device), pitch_true.to(device), roll_true.to(device)

        optimizer.zero_grad()
        yaw_pred, pitch_pred, roll_pred = model(images)
        loss = head_pose_loss(yaw_pred, pitch_pred, roll_pred, yaw_true, pitch_true, roll_true)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# Create the data loaders
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.GaussianBlur(kernel_size=5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



# Define the knowledge distillation training loop
def distill_train(student_model, teacher_models, dataloader, optimizer, device):
    student_model.train()
    total_loss = 0
    for images, _ in dataloader:
        images = images.to(device)

        optimizer.zero_grad()
        yaw_pred, pitch_pred, roll_pred = student_model(images)

        ensemble_yaw = torch.zeros_like(yaw_pred)
        ensemble_pitch = torch.zeros_like(pitch_pred)
        ensemble_roll = torch.zeros_like(roll_pred)

        for teacher_model in teacher_models:
            with torch.no_grad():
                teacher_yaw, teacher_pitch, teacher_roll = teacher_model(images)
                ensemble_yaw += teacher_yaw.softmax(dim=1)
                ensemble_pitch += teacher_pitch.softmax(dim=1)
                ensemble_roll += teacher_roll.softmax(dim=1)

        ensemble_yaw /= len(teacher_models)
        ensemble_pitch /= len(teacher_models)
        ensemble_roll /= len(teacher_models)

        loss = nn.KLDivLoss(reduction='batchmean')(nn.LogSoftmax(dim=1)(yaw_pred), ensemble_yaw) + \
               nn.KLDivLoss(reduction='batchmean')(nn.LogSoftmax(dim=1)(pitch_pred), ensemble_pitch) + \
               nn.KLDivLoss(reduction='batchmean')(nn.LogSoftmax(dim=1)(roll_pred), ensemble_roll)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# Evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            yaw_true, pitch_true, roll_true = labels[:, 0].long(), labels[:, 1].long(), labels[:, 2].long()
            yaw_true, pitch_true, roll_true = yaw_true.to(device), pitch_true.to(device), roll_true.to(device)

            yaw_pred, pitch_pred, roll_pred = model(images)
            loss = head_pose_loss(yaw_pred, pitch_pred, roll_pred, yaw_true, pitch_true, roll_true)
            total_loss += loss.item()

    return total_loss / len(dataloader)

# Load AFLW-2000 dataset
def load_aflw2000():
    # Implement code to load AFLW-2000 dataset
    # Return test_x and test_y
    pass

# Load BIWI dataset
def load_biwi():
    # Implement code to load BIWI dataset
    # Return test_x and test_y
    pass


train_x, train_y, valid_x, valid_y = load_300w_train()
train_dataset = TensorFlow300WLPDataset(list(zip(train_x, train_y)), transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

valid_dataset = TensorFlow300WLPDataset(list(zip(valid_x, valid_y)), transform=transforms.ToTensor())
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

# Load test datasets
aflw2000_test_x, aflw2000_test_y = load_aflw2000()
aflw2000_test_dataset = TensorFlow300WLPDataset(list(zip(aflw2000_test_x, aflw2000_test_y)), transform=transforms.ToTensor())
aflw2000_test_loader = DataLoader(aflw2000_test_dataset, batch_size=64, shuffle=False)

biwi_test_x, biwi_test_y = load_biwi()
biwi_test_dataset = TensorFlow300WLPDataset(list(zip(biwi_test_x, biwi_test_y)), transform=transforms.ToTensor())
biwi_test_loader = DataLoader(biwi_test_dataset, batch_size=64, shuffle=False)

# Create the teacher models
teacher_model1 = HPENet().to('cuda')
teacher_model2 = HPENet().to('cuda')
teacher_model3 = HPENet().to('cuda')

# Train the teacher models
optimizer1 = optim.Adam(teacher_model1.parameters(), lr=1e-4)
optimizer2 = optim.Adam(teacher_model2.parameters(), lr=1e-4)
optimizer3 = optim.Adam(teacher_model3.parameters(), lr=1e-4)

for epoch in range(100):
    train_loss1 = train(teacher_model1, train_loader, optimizer1, 'cuda')
    train_loss2 = train(teacher_model2, train_loader, optimizer2, 'cuda')
    train_loss3 = train(teacher_model3, train_loader, optimizer3, 'cuda')
    
    # Evaluate on the validation set
    valid_loss1 = evaluate(teacher_model1, valid_loader, 'cuda')
    valid_loss2 = evaluate(teacher_model2, valid_loader, 'cuda')
    valid_loss3 = evaluate(teacher_model3, valid_loader, 'cuda')
    
    print(f"Epoch [{epoch+1}/100], Teacher 1 Train Loss: {train_loss1:.4f}, Valid Loss: {valid_loss1:.4f}")
    print(f"Epoch [{epoch+1}/100], Teacher 2 Train Loss: {train_loss2:.4f}, Valid Loss: {valid_loss2:.4f}")
    print(f"Epoch [{epoch+1}/100], Teacher 3 Train Loss: {train_loss3:.4f}, Valid Loss: {valid_loss3:.4f}")

# Save the trained teacher models
save_model(teacher_model1, 'teacher_model1.pth')
save_model(teacher_model2, 'teacher_model2.pth')
save_model(teacher_model3, 'teacher_model3.pth')

# Create the student model
student_model = HPENet().to('cuda')
optimizer = optim.Adam(student_model.parameters(), lr=1e-4)

# Perform knowledge distillation training
teacher_models = [teacher_model1, teacher_model2, teacher_model3]
for epoch in range(200):
    distill_loss = distill_train(student_model, teacher_models, train_loader, optimizer, 'cuda')
    
    # Evaluate on the validation set
    valid_loss = evaluate(student_model, valid_loader, 'cuda')
    
    print(f"Epoch [{epoch+1}/200], Distill Train Loss: {distill_loss:.4f}, Valid Loss: {valid_loss:.4f}")

# Save the trained student model
save_model(student_model, 'HPENet.pth')

# Evaluate on test datasets
aflw2000_test_loss = evaluate(student_model, aflw2000_test_loader, 'cuda')
biwi_test_loss = evaluate(student_model, biwi_test_loader, 'cuda')

print(f"AFLW-2000 Test Loss: {aflw2000_test_loss:.4f}")
print(f"BIWI Test Loss: {biwi_test_loss:.4f}")