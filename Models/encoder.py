from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
import torch
import numpy as np                                   # ② 추가
from PIL import Image
from torchvision import transforms

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH = 32                                           # ① 추가


def get_resnet_model():
    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    for p in model.parameters():
        p.requires_grad = False

    model.fc = nn.Identity()
    model.eval()

    return model.to(DEVICE)


tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225)),
])


def encode(model, paths, batch=BATCH):
    out_list = []

    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]

        imgs = [tf(Image.open(p).convert('RGB')) for p in chunk]   # ④ chunk
        x = torch.stack(imgs).to(DEVICE)

        with torch.no_grad():
            vecs = model(x)

        vecs = vecs / vecs.norm(dim=1, keepdim=True)
        out_list.append(vecs.cpu().numpy())
        # ③ 여기까지 전부 반복문 안

    return np.concatenate(out_list, axis=0)          # ② 철자