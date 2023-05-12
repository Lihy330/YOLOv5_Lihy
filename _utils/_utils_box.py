import torch
from YoloNet import YOLO


class DecodeBox:
    def __init__(self, mode='train'):
        super(DecodeBox, self).__init__()
        self.mode = mode
        self.base_depth = 3
        self.base_channels = 64
        self.num_classes = 20
        self.anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        self.input_shape = [640, 640]
        self.feature_shape = [int(self.input_shape[0] / num) for num in [8, 16, 32]]
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = self.generate_model(YOLO(self.base_depth, self.base_channels, self.num_classes, self.anchors_mask)).to(self.device)

    def generate_model(self, model):
        if self.mode == 'train':
            return model
        else:
            return model.eval()

    def decode(self, image):
        batch_size = image.shape[0]
        image = image.to(self.device)
        output = self.model(image)
        # 枚举每一个特征层
        for layer in range(len(self.anchors_mask)):
            output_decode = output[layer].permute(0, 2, 3, 1).reshape(-1, self.feature_shape[layer],
                                                                      self.feature_shape[layer], len(self.anchors_mask),
                                                                      self.num_classes + 5)
            print(output_decode.shape)


de = DecodeBox('train')
im = torch.randn(1, 3, 640, 640)
de.decode(im)


