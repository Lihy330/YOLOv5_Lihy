import torch
import torch.nn as nn
from YoloNet import YOLOBody
from _utils._utils import get_anchors, collate_fn
from _utils.dataloader import Yolo_Dataset
from torch.utils.data import DataLoader


# pred.shape = (bs, 3, fs, fs, 2)
# target.shape = (bs, 3, fs, fs, 2)
def get_CIOU_v(pred, target):
    arctan_target = torch.arctan(target[..., 0] / target[..., 1])
    arctan_pred = torch.arctan(pred[..., 0] / pred[..., 1])
    return (4 / (torch.pi ** 2)) * (arctan_target - arctan_pred)


def get_CIOU_alpha(IOU, v):
    return v / (1 - IOU + v)


def get_IOU(pred_xy, pred_wh, target_xy, target_wh):
    # 获取到角点坐标
    min_pred_xy = pred_xy - pred_wh / 2.
    max_pred_xy = min_pred_xy + pred_wh
    min_target_xy = target_xy - target_wh / 2.
    max_target_xy = min_target_xy + target_wh
    # 计算相交的面积
    # 获取到相交的框的角点坐标
    intersect_min_box_xy = torch.max(min_pred_xy, min_target_xy)
    intersect_max_box_xy = torch.min(max_pred_xy, max_target_xy)
    intersect_box_area = (intersect_max_box_xy[..., 0] - intersect_min_box_xy[..., 0]) * \
                         (intersect_max_box_xy[..., 1] - intersect_min_box_xy[..., 1])
    # 如果出现面积是负值，直接置0
    intersect_box_area[intersect_box_area < 0] = 0.
    # 求最小外接矩形坐标
    union_min_box_xy = torch.min(min_pred_xy, min_target_xy)
    union_max_box_xy = torch.max(max_pred_xy, max_target_xy)
    # 最小外接矩形的对角线距离
    c = (union_max_box_xy[..., 0] - union_min_box_xy[..., 0]) ** 2 + \
        (union_max_box_xy[..., 1] - union_min_box_xy[..., 1]) ** 2
    # 获取预测框与标签框的中心点坐标的欧氏距离
    rou = (pred_xy[..., 0] - target_xy[..., 0]) ** 2 + \
          (pred_xy[..., 1] - target_xy[..., 1]) ** 2
    # 计算预测框与标签框相并的面积
    union_box_area = (pred_wh[..., 0] * pred_wh[..., 1]) + (target_wh[..., 0] * target_wh[..., 1]) - intersect_box_area
    union_box_area[union_box_area < 0] = 0.
    # 计算交并比
    IOU = intersect_box_area / union_box_area
    return IOU, c, rou


# pred.shape = (bs, 3, fs, fs, 4)
# target.shape = (bs, 3, fs, fs, 4)
def get_CIOU(pred, target):
    pred_box_xy = pred[..., :2]
    target_box_xy = target[..., :2]
    pred_box_wh = pred[..., 2:4]
    target_box_wh = target[..., 2:4]
    CIOU_v = get_CIOU_v(pred_box_wh, target_box_wh)
    IOU, c, rou = get_IOU(pred_box_xy, pred_box_wh, target_box_xy, target_box_wh)
    CIOU_alpha = get_CIOU_alpha(IOU, CIOU_v)
    CIOU = IOU - ((rou / c) + CIOU_alpha * CIOU_v)
    return CIOU


class YOLO_Loss(nn.Module):
    def __init__(self, input_shape, anchors, num_classes, anchors_mask):
        super(YOLO_Loss, self).__init__()
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask
        self.num_classes = num_classes
        self.anchors = anchors
        self.lambda_pos = 5
        self.lambda_neg = 0.5
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def get_pred_boxes(self, l, output):
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
        grid_w = torch.stack([torch.tensor([item]).repeat(feature_shape[0], feature_shape[1])
                              for item in anchors_layer_w], dim=0).type(FloatTensor)

        grid_h = torch.stack([torch.tensor([item]).repeat(feature_shape[0], feature_shape[1])
                              for item in anchors_layer_h], dim=0).type(FloatTensor)

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

        return pred_boxes

    # output 即神经网络输出的某个特征层
    # y_true 即先验框，网格与标签的对应情况
    def forward(self, l, output, y_true):
        batch_size = output.shape[0]
        # 当前特征层的尺寸
        feature_shape = output.shape[2:4]
        # pred_boxes就是当前特征层上的预测框  shape = (bs, 3, feature_shape[0], feature_shape[1], num_classes + 5)
        pred_boxes = self.get_pred_boxes(l, output)

        # 损失计算
        # y_true[..., 4] == 1 表示负责预测物体的那部分先验框（也就是正样本）
        pos_mask = y_true[..., 4] == 1
        # 负样本
        neg_mask = y_true[..., 4] == 0
        # 1.定位误差计算
        #   先计算所有的CIOU损失，后续根据正负样本取值即可
        #   pred_boxes[:, :, :, :, :4].shape = (bs, 3, fs, fs, 4)
        CIOU = get_CIOU(pred_boxes[..., :4], y_true[..., :4])
        #   只有正样本才会计算定位损失
        loc_loss = self.lambda_pos * torch.sum(
            (1 - CIOU[pos_mask]) * (2 - (y_true[pos_mask][..., 2] / feature_shape[0]) *
                                    (y_true[pos_mask][..., 3] / feature_shape[1])))

        # 2.置信度误差计算
        #   正样本置信度误差
        pos_conf_loss = (-1) * torch.sum(torch.nn.BCELoss()(pred_boxes[pos_mask][..., 4], y_true[pos_mask][..., 4]))
        #   负样本置信度误差
        neg_conf_loss = (-1) * self.lambda_neg * torch.sum(torch.nn.BCELoss()(pred_boxes[neg_mask][..., 4],
                                                                              y_true[neg_mask][..., 4]))

        # 3.分类损失
        cls_loss = (-1) * torch.sum(torch.nn.BCELoss()(pred_boxes[pos_mask][..., 5:], y_true[pos_mask][..., 5:]))

        total_loss = loc_loss + pos_conf_loss + neg_conf_loss + cls_loss

        return total_loss


if __name__ == '__main__':
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    num_classes = 20
    input_shape = [640, 640]
    anchors_path = r"D:\YOLO\yolov5-pytorch-main\model_data\yolo_anchor.txt"
    train_data_file_path = r"./voc_train_data.txt"
    val_data_file_path = r"./voc_val_data.txt"
    anchors = get_anchors(anchors_path)
    criterion = YOLO_Loss(input_shape, anchors, num_classes, anchors_mask)
    model = YOLOBody(3, 64, num_classes, anchors_mask).to(criterion.device)
    yolo_dt = Yolo_Dataset(anchors, train_data_file_path, val_data_file_path, num_classes, anchors_mask, mode='train')
    yolo_dl = DataLoader(yolo_dt, batch_size=1, shuffle=False, collate_fn=collate_fn)
    for idx, (inputs, boxs, y_true) in enumerate(yolo_dl):
        inputs = inputs.to(criterion.device)
        out = model(inputs)
        for layer in range(len(out)):
            if layer == 0:
                y_true[layer] = y_true[layer].to(criterion.device)
                criterion(layer, out[layer], y_true[layer])
