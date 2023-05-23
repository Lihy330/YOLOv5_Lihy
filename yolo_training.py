import torch
import torch.nn as nn
from YoloNet import YOLOBody
from _utils._utils import get_anchors, collate_fn
from _utils.dataloader import Yolo_Dataset
from torch.utils.data import DataLoader


# pred.shape = (bs, 3, fs, fs, 2)
# target.shape = (bs, 3, fs, fs, 2)
def get_CIOU_v(pred, target):
    epsilon = 1e-7
    arctan_target = torch.arctan(target[..., 0] / (target[..., 1] + epsilon))
    arctan_pred = torch.arctan(pred[..., 0] / (pred[..., 1] + epsilon))
    return (4 / (torch.pi ** 2)) * (arctan_target - arctan_pred)


def get_CIOU_alpha(IOU, v):
    epsilon = 1e-7
    return v / (1 - IOU + v + epsilon)


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
    # 计算交并比
    IOU = intersect_box_area / union_box_area
    return IOU, c, rou


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)


# pred.shape = (bs, 3, fs, fs, 4)
# target.shape = (bs, 3, fs, fs, 4)
def get_CIOU(pred, target):
    epsilon = 1e-7
    pred_box_xy = pred[..., :2]
    target_box_xy = target[..., :2]
    pred_box_wh = pred[..., 2:4]
    target_box_wh = target[..., 2:4]
    CIOU_v = get_CIOU_v(pred_box_wh, target_box_wh)
    IOU, c, rou = get_IOU(pred_box_xy, pred_box_wh, target_box_xy, target_box_wh)
    CIOU_alpha = get_CIOU_alpha(IOU, CIOU_v)
    CIOU = IOU - ((rou / (c + epsilon)) + CIOU_alpha * CIOU_v)
    return CIOU


def calculate_CIOU(box1, box2):
    # 获取框的坐标信息
    box1_x, box1_y, box1_w, box1_h = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
    box2_x, box2_y, box2_w, box2_h = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    # 计算框的中心点坐标
    box1_center_x = box1_x + box1_w / 2
    box1_center_y = box1_y + box1_h / 2
    box2_center_x = box2_x + box2_w / 2
    box2_center_y = box2_y + box2_h / 2

    # 计算相交框的左上角和右下角坐标
    intersect_x1 = torch.max(box1_x - box1_w / 2, box2_x - box2_w / 2)
    intersect_y1 = torch.max(box1_y - box1_h / 2, box2_y - box2_h / 2)
    intersect_x2 = torch.min(box1_x + box1_w / 2, box2_x + box2_w / 2)
    intersect_y2 = torch.min(box1_y + box1_h / 2, box2_y + box2_h / 2)

    # 计算相交框的面积
    intersect_area = torch.clamp(intersect_x2 - intersect_x1, min=0) * torch.clamp(intersect_y2 - intersect_y1, min=0)

    # 计算框的面积
    box1_area = box1_w * box1_h
    box2_area = box2_w * box2_h

    # 计算相并框的面积
    union_area = box1_area + box2_area - intersect_area

    # 计算框的对角线距离的平方
    c = (box2_center_x - box1_center_x) ** 2 + (box2_center_y - box1_center_y) ** 2

    # 计算预测框与目标框的IoU
    iou = intersect_area / union_area

    epsilon = 1e-7  # 很小的常数

    v = (4 / (torch.pi ** 2)) * torch.pow(
        torch.atan(box2_w / (box2_h + epsilon)) - torch.atan(box1_w / (box1_h + epsilon)), 2)

    # 计算CIoU损失
    ciou_loss = 1 - iou + v / (1 - iou + v) * c / ((box1_w + box1_h) ** 2 + (box2_w + box2_h) ** 2 - c)

    return ciou_loss


class YOLO_Loss(nn.Module):
    def __init__(self, input_shape, anchors, num_classes, anchors_mask, device):
        super(YOLO_Loss, self).__init__()
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask
        self.num_classes = num_classes
        self.anchors = anchors
        self.device = device
        # 置信度损失权重(三个特征层不同)，list[0]就是第0个特征层的权重，以此类推
        self.balance = [0.4, 1.0, 4]
        # 定位误差的权重
        self.box_ratio = 0.05
        # 置信度损失权重
        self.obj_ratio = 1 * (input_shape[0] * input_shape[1]) / (640 ** 2)
        # 分类误差的权重
        self.cls_ratio = 0.5 * (num_classes / 80)

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

        tx = torch.sigmoid(prediction[..., 0])
        ty = torch.sigmoid(prediction[..., 1])
        tw = torch.sigmoid(prediction[..., 2])
        th = torch.sigmoid(prediction[..., 3])
        conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])

        # pred_boxes 就是最终的预测框的尺寸信息
        coordination_x = (2 * tx - 0.5 + grid_x).unsqueeze(-1)
        coordination_y = (2 * ty - 0.5 + grid_y).unsqueeze(-1)
        pred_box_w = (grid_w * ((2 * tw) ** 2)).unsqueeze(-1)
        pred_box_h = (grid_h * ((2 * th) ** 2)).unsqueeze(-1)

        pred_boxes = torch.concat([coordination_x, coordination_y, pred_box_w, pred_box_h], dim=-1)

        return pred_boxes, conf, pred_cls

    def clip_by_tensor(self, t, t_min, t_max):
        t = t.float()
        # t >= t_min ==> t, t < t_min ==> t_min，意义就是将t中小于t_min的数值变成t_min，用t_min替换掉数据中的较小的值
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        # result <= t_max ==> result, result > t_max ==> t_max，意义就是将result中大于t_max的数值变成t_max
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        # 最终的result就是将t的数据截取到[t_min, t_max]
        return result

    def BCELoss(self, pred, target):
        epsilon = 1e-7
        # 先将pred的数据截取到[epsilon, 1.0 - epsilon]
        pred = self.clip_by_tensor(pred, epsilon, 1.0 - epsilon)
        # output = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        output = torch.nn.BCELoss()(pred, target)
        return output

    # output 即神经网络输出的某个特征层
    # y_true 即先验框，网格与标签的对应情况
    def forward(self, l, output, y_true):
        batch_size = output.shape[0]
        # 当前特征层的尺寸
        feature_shape = output.shape[2:4]
        # pred_boxes就是当前特征层上的预测框  shape = (bs, 3, feature_shape[0], feature_shape[1], 4)
        pred_boxes, conf, pred_cls = self.get_pred_boxes(l, output)

        # 损失计算
        # y_true[..., 4] == 1 表示负责预测物体的那部分先验框（也就是正样本）
        pos_mask = y_true[..., 4] == 1
        # 负样本
        neg_mask = y_true[..., 4] == 0
        # 1.定位误差计算
        # #   先计算所有的CIOU损失，后续根据正负样本取值即可
        # CIOU = get_CIOU(pred_boxes, y_true[..., :4])
        # #   只有正样本才会计算定位损失
        # loc_loss = self.lambda_pos * torch.sum(
        #     (1 - CIOU[pos_mask]) * (2 - (y_true[pos_mask][..., 2] / feature_shape[0]) *
        #                             (y_true[pos_mask][..., 3] / feature_shape[1])))

        # GPT实现的CIOU_Loss
        CIOU_Loss = calculate_CIOU(pred_boxes, y_true[..., :4])
        CIOU = 1 - CIOU_Loss
        loc_loss = torch.mean(
            CIOU_Loss[pos_mask] * (2 - (y_true[pos_mask][..., 2] / feature_shape[0]) *
                                   (y_true[pos_mask][..., 3] / feature_shape[1])))

        # 2.置信度误差计算
        # 置信度标签
        #   满足条件说明当前先验框是正样本，那么置信度标签值就是CIOU值
        #   负样本的标签值就是0
        conf_pos_neg_label = torch.where(y_true[..., 4] == 1, CIOU.detach().clamp(0), torch.zeros_like(y_true[..., 4]))

        conf_loss = torch.mean(self.BCELoss(conf, conf_pos_neg_label))

        # 3.分类损失
        cls_loss = torch.mean(self.BCELoss(pred_cls[pos_mask], y_true[pos_mask][..., 5:]))

        total_loss = loc_loss * self.box_ratio + conf_loss * self.balance[l] * self.obj_ratio + cls_loss * self.cls_ratio

        return total_loss


if __name__ == '__main__':
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    num_classes = 20
    input_shape = [640, 640]
    anchors_path = r"D:\YOLO\yolov5-pytorch-main\model_data\yolo_anchor.txt"
    train_data_file_path = r"./voc_train_data.txt"
    val_data_file_path = r"./voc_val_data.txt"
    anchors = get_anchors(anchors_path)
    criterion = YOLO_Loss(input_shape, anchors, num_classes, anchors_mask, device)
    model = YOLOBody(3, 64, num_classes, anchors_mask).to(criterion.device)
    yolo_dt = Yolo_Dataset(anchors, train_data_file_path, val_data_file_path, num_classes, anchors_mask, mode='train')
    yolo_dl = DataLoader(yolo_dt, batch_size=1, shuffle=False, collate_fn=collate_fn)
    for idx, (inputs, boxs, y_true) in enumerate(yolo_dl):
        inputs = inputs.to(criterion.device)
        out = model(inputs)
        for layer in range(len(out)):
            if layer == 0:
                y_true[layer] = y_true[layer].to(criterion.device)
                loss = criterion(layer, out[layer], y_true[layer])
                print(loss)
                break
