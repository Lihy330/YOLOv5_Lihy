import PIL.Image
from PIL.ImageDraw import Draw
from YoloNet import YOLOBody
from _utils._utils_box import DecodeBox
from _utils._utils import get_anchors, get_classes_dict
import torch
from torchvision.transforms import transforms


class YOLO:
    def __init__(self):
        super(YOLO, self).__init__()
        self.confidence = 0.5
        self.base_depth = 3
        self.base_channels = 64
        self.num_classes = 80
        self.anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        self.input_shape = [640, 640]
        self.feature_shape = [int(self.input_shape[0] / num) for num in [8, 16, 32]]
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.anchors_path = r"D:\YOLO\yolov5-pytorch-main\model_data\yolo_anchor.txt"
        self.anchors = get_anchors(self.anchors_path)
        self.model = self.generate_model(
            YOLOBody(self.base_depth, self.base_channels, self.num_classes, self.anchors_mask))
        self.resize = transforms.Resize(self.input_shape)
        self.toTensor = transforms.ToTensor()
        self.decode = DecodeBox(self.anchors, self.device)

    def generate_model(self, model):
        # model.load_state_dict(torch.load(r'./model_data/yolov5_l.pth'))
        return model.to(self.device)

    def detect_image(self, image):
        i_w, i_h = image.size
        w_scale = i_w / self.input_shape[0]
        h_scale = i_h / self.input_shape[0]
        input_image = self.toTensor(self.resize(image)).unsqueeze(0).to(self.device)
        output = self.decode.decode_box(self.model, input_image)
        # shape (bs, all_boxes, 5 + num_classes)
        output = torch.concat(output, dim=1)
        output = output[output[..., 4] > self.confidence]
        score = output[:, 4]
        output = output[torch.argsort(score, descending=True)]
        output_x_min = (output[..., 0] - output[..., 2] / 2) * self.input_shape[0] * w_scale
        output_x_max = (output[..., 0] + output[..., 2] / 2) * self.input_shape[0] * w_scale
        output_y_min = (output[..., 1] - output[..., 3] / 2) * self.input_shape[0] * h_scale
        output_y_max = (output[..., 1] + output[..., 3] / 2) * self.input_shape[0] * h_scale
        output_class = output[..., 5:]
        draw = Draw(image)
        for index in range(output.shape[0]):
            draw.rectangle([output_x_min[index], output_y_min[index], output_x_max[index], output_y_max[index]],
                           outline=(0, 255, 0), width=1)
            print(torch.argmax(output_class[index]))
        image.show()


if __name__ == "__main__":
    yolo = YOLO()
    img = PIL.Image.open(r"./img/000009.jpg")
    yolo.detect_image(img)
