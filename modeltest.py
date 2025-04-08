import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import h5py
import numpy as np
from torchvision.models import vgg16

# モデルの定義（dogorcat.py と同様の構造）
class VGG16Model(nn.Module):
    def __init__(self):
        super(VGG16Model, self).__init__()
        # pretrainedはFalseにして、後で読み込むパラメータで上書き
        self.vgg16 = vgg16(pretrained=False)
        self.vgg16.classifier[6] = nn.Linear(4096, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.vgg16(x)
        return x

# 画像の前処理
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((100, 100)),
    transforms.ToTensor()
])

def load_model_from_h5(h5_path):
    model = VGG16Model()
    state_dict = {}
    with h5py.File(h5_path, 'r') as f:
        for key in f.keys():
            # 各層のパラメータをnumpy配列からTensorに変換
            state_dict[key] = torch.tensor(np.array(f[key]))
    # 読み込んだstate_dictでモデルのパラメータを更新
    model.load_state_dict(state_dict)
    return model

def load_model_with_final_layer(pth_path):
    model = VGG16Model()
    # 最後の全結合層のパラメータだけを読み込み
    final_layer_params = torch.load(pth_path)
    model.vgg16.classifier[6].load_state_dict(final_layer_params)
    return model

def predict_image(model, image_path):
    # 画像の読み込みと前処理
    image = cv2.imread(image_path)
    if image is None:
        print("画像がロードできません:", image_path)
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image)
    image = image.unsqueeze(0)  # バッチサイズの次元を追加

    model.eval()
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)[0]
        # ラベル0:猫、ラベル1:犬と仮定
        cat_prob = probabilities[0].item() * 100
        dog_prob = probabilities[1].item() * 100
    print(f"猫の確率: {cat_prob:.2f}%")
    print(f"犬の確率: {dog_prob:.2f}%")

if __name__ == "__main__":
    model_pth_path = 'C:/Users/neore/Documents/Python_code/ML-models/my_final_layer_params.pth'  # 最後の層のパラメータファイル
    image_path = 'D:/dog_or_cat_data/dog_or_cat_data/train/cat/cat.883.jpg'
    model = load_model_with_final_layer(model_pth_path)
    predict_image(model, image_path)