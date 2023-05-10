import torch
from YoloNet import YOLO


if __name__ == "__main__":
    img = torch.randn(1, 3, 640, 640)
    model = YOLO(3, 64, 20)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    img = img.to(device)
    out = model(img)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
