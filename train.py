import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms

# Define the head pose estimation model
class EHPNet(nn.Module):
    def __init__(self, num_bins=62):
        super(EHPNet, self).__init__()
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

# Define the training loop
def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for images, labels in dataloader:
        images = images.to(device)
        yaw_true = labels[:, 0].long().to(device)
        pitch_true = labels[:, 1].long().to(device)
        roll_true = labels[:, 2].long().to(device)

        optimizer.zero_grad()
        yaw_pred, pitch_pred, roll_pred = model(images)
        loss = head_pose_loss(yaw_pred, pitch_pred, roll_pred, yaw_true, pitch_true, roll_true)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

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

# Create the data loaders
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.GaussianBlur(kernel_size=5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = YourDataset(data_dir='path/to/300W-LPA', transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Create the teacher models
teacher_model1 = EHPNet().to('cuda')
teacher_model2 = EHPNet().to('cuda')
teacher_model3 = EHPNet().to('cuda')

# Train the teacher models
optimizer1 = optim.Adam(teacher_model1.parameters(), lr=1e-4)
optimizer2 = optim.Adam(teacher_model2.parameters(), lr=1e-4)
optimizer3 = optim.Adam(teacher_model3.parameters(), lr=1e-4)

for epoch in range(100):
    train_loss1 = train(teacher_model1, train_loader, optimizer1, 'cuda')
    train_loss2 = train(teacher_model2, train_loader, optimizer2, 'cuda')
    train_loss3 = train(teacher_model3, train_loader, optimizer3, 'cuda')
    print(f"Epoch [{epoch+1}/100], Teacher 1 Loss: {train_loss1:.4f}, Teacher 2 Loss: {train_loss2:.4f}, Teacher 3 Loss: {train_loss3:.4f}")

# Create the student model
student_model = EHPNet().to('cuda')
optimizer = optim.Adam(student_model.parameters(), lr=1e-4)

# Perform knowledge distillation training
teacher_models = [teacher_model1, teacher_model2, teacher_model3]
for epoch in range(200):
    distill_loss = distill_train(student_model, teacher_models, train_loader, optimizer, 'cuda')
    print(f"Epoch [{epoch+1}/200], Distill Loss: {distill_loss:.4f}")
