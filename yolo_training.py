import torch
import torch.nn as nn
from YoloNet import YOLOBody
from _utils._utils import get_anchors


class YOLO_Loss(nn.Module):
    def __init__(self, input_shape, anchors, num_classes, anchors_mask):
        super(YOLO_Loss, self).__init__()
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask
        self.num_classes = num_classes
        self.anchors = anchors
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # output 即神经网络输出的某个特征层
    # y_true 即先验框，网格与标签的对应情况
    def forward(self, l, output, y_true):
        batch_size = output.shape[0]
        # 当前特征层的尺寸
        feature_shape = output.shape[2:4]
        FloatTensor = torch.cuda.FloatTensor if self.device == 'cuda:0' else torch.FloatTensor

        # 先解码，先验框计算误差
        prediction = output.view(batch_size, len(self.anchors_mask), self.num_classes + 5,
                                 feature_shape[0], feature_shape[1]).permute(0, 1, 3, 4, 2).contiguous()
        grid_x = torch.linspace(0, feature_shape[0] - 1, feature_shape[0]).repeat(feature_shape[0], 1) \
            .repeat(batch_size * len(self.anchors_mask), 1, 1). \
            view(batch_size, -1, feature_shape[0], feature_shape[1]).type(FloatTensor)
        grid_y = torch.linspace(0, feature_shape[0] - 1, feature_shape[0]).view(feature_shape[0], 1). \
            repeat(1, feature_shape[0]).repeat(batch_size * len(self.anchors_mask), 1, 1). \
            view(batch_size, -1, feature_shape[0], feature_shape[1]).type(FloatTensor)

        # 获取当前特征层宽高尺寸
        anchors_layer_w = torch.from_numpy(self.anchors[self.anchors_mask[l]][:, 0]) / self.input_shape[0] * \
                          feature_shape[1]
        anchors_layer_h = torch.from_numpy(self.anchors[self.anchors_mask[l]][:, 1]) / self.input_shape[0] * \
                          feature_shape[0]
        # 宽高调整坐标网格
        grid_w = torch.stack([torch.tensor(item).repeat(feature_shape[0]).repeat(feature_shape[0], 1)
                              for item in anchors_layer_w], dim=0)

        grid_h = torch.stack([torch.tensor(item).repeat(feature_shape[0]).repeat(feature_shape[0], 1)
                              for item in anchors_layer_h], dim=0)

        pred_boxes = torch.zeros_like(prediction).type(FloatTensor)

        tx = torch.sigmoid(prediction[..., 0])
        ty = torch.sigmoid(prediction[..., 1])
        tw = torch.sigmoid(prediction[..., 2])
        th = torch.sigmoid(prediction[..., 3])
        conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])

        # pred_boxes 就是最终的预测框的尺寸信息
        pred_boxes[..., 0] = 2 * tx - 0.5 + grid_x
        pred_boxes[..., 1] = 2 * ty - 0.5 + grid_y
        pred_boxes[..., 2] = grid_w * (2 * tw ** 2)
        pred_boxes[..., 3] = grid_h * (2 * th ** 2)
        pred_boxes[..., 4] = conf
        pred_boxes[..., 5:] = pred_cls


if __name__ == '__main__':
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    num_classes = 20
    input_shape = [640, 640]
    anchors_path = r"D:\YOLO\yolov5-pytorch-main\model_data\yolo_anchor.txt"
    anchors = get_anchors(anchors_path)
    criterion = YOLO_Loss(input_shape, anchors, num_classes, anchors_mask)
    model = YOLOBody(3, 64, num_classes, anchors_mask).to(criterion.device)
    image = torch.randn(1, 3, 640, 640).to(criterion.device)
    y_true = torch.randn(3, 20, 20, num_classes + 5).to(criterion.device)
    out = model(image)
    for layer in range(len(out)):
        criterion(layer, out[layer], y_true)
        break
