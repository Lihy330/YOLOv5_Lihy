import PIL.Image

from YoloNet import YOLOBody
from _utils._utils_box import DecodeBox
from _utils._utils import get_anchors
import torch
from torchvision.transforms import transforms


class YOLO:
    def __init__(self, mode):
        super(YOLO, self).__init__()
        self.mode = mode
        self.base_depth = 3
        self.base_channels = 64
        self.num_classes = 20
        self.anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        self.input_shape = [640, 640]
        self.feature_shape = [int(self.input_shape[0] / num) for num in [8, 16, 32]]
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.anchors_path = r"D:\YOLO\yolov5-pytorch-main\model_data\yolo_anchor.txt"
        self.anchors = get_anchors(self.anchors_path)
        self.model = self.generate_model(
            YOLOBody(self.base_depth, self.base_channels, self.num_classes, self.anchors_mask))
        self.resize = transforms.Resize(self.input_shape)
        self.decode = DecodeBox(self.anchors, self.device)

    def generate_model(self, model):
        if self.mode == 'train':
            return model.to(self.device)
        else:
            return model.eval().to(self.device)

    def detect_image(self, image):
        image = self.resize(image)
        image = transforms.ToTensor()(image).unsqueeze(0).to(self.device)
        output = self.decode.decode_box(self.model, image)
        print(output[0].shape)



if __name__ == "__main__":
    yolo = YOLO('train')
    img = PIL.Image.open(r"./img/000009.jpg")
    yolo.detect_image(img)

