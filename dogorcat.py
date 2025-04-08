import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import vgg16
from sklearn.model_selection import train_test_split
import h5py
import torch.nn.functional as F

# データセットクラスの定義
class DogCatDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        valid_paths = []
        valid_labels = []
        for path, label in zip(img_paths, labels):
            # 画像の有効性をチェック
            image = cv2.imread(path)
            if image is not None:
                valid_paths.append(path)
                valid_labels.append(label)
            else:
                print(f"Warning: 画像の読み込みに失敗しました: {path}")
        self.img_paths = valid_paths
        self.labels = valid_labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label
# 画像の前処理
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((100, 100)),
    transforms.ToTensor()
])

cats_path = [os.path.join("D:/dog_or_cat_data/dog_or_cat_data/train/cat/", fname).encode('utf-8').decode('utf-8') for fname in os.listdir("D:/dog_or_cat_data/dog_or_cat_data/train/cat/")]
dogs_path = [os.path.join("D:/dog_or_cat_data/dog_or_cat_data/train/dog/", fname).encode('utf-8').decode('utf-8') for fname in os.listdir("D:/dog_or_cat_data/dog_or_cat_data/train/dog/")]

img_paths = cats_path + dogs_path
labels = [0] * len(cats_path) + [1] * len(dogs_path)

# データセットの分割
x_train, x_test, y_train, y_test = train_test_split(img_paths, labels, test_size=0.2, random_state=42)

# データローダーの作成
train_dataset = DogCatDataset(x_train, y_train, transform=transform)
test_dataset = DogCatDataset(x_test, y_test, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# モデルの定義
class VGG16Model(nn.Module):
    def __init__(self):
        super(VGG16Model, self).__init__()
        self.vgg16 = vgg16(pretrained=True)
        self.vgg16.classifier[6] = nn.Linear(4096, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.vgg16(x)
        return x

model = VGG16Model()

# 損失関数と最適化関数の設定
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

# 訓練ループ
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

    # モデルの保存
    torch.save(model.state_dict(), 'C:/Users/neore/Documents/Python_code/ML-models/my_model_pytorch.pth')

    # .h5形式でモデルを保存
    with h5py.File('C:/Users/neore/Documents/Python_code/ML-models/my_model_pytorch.h5', 'w') as f:
        for key, value in model.state_dict().items():
            f.create_dataset(key, data=value.cpu().numpy())
    
    # 最後の層のパラメータを取得する
    final_layer_params = model.vgg16.classifier[6].state_dict()
    torch.save(final_layer_params, 'C:/Users/neore/Documents/Python_code/ML-models/my_final_layer_params.pth')

# テストループ
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total}%')