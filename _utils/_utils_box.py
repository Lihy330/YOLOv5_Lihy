import torch


class DecodeBox:
    def __init__(self, input_shape, anchors, device, num_classes, anchors_mask):
        super(DecodeBox, self).__init__()
        self.num_classes = num_classes
        self.anchors_mask = anchors_mask
        self.input_shape = input_shape
        self.feature_shape = [int(self.input_shape[0] / num) for num in [32, 16, 8]]
        self.anchors_path = r"D:\YOLO\yolov5-pytorch-main\model_data\yolo_anchor.txt"
        self.anchors = anchors
        self.device = device

    def decode_box(self, model, inputs):
        output_boxes = []
        batch_size = inputs.shape[0]
        output = model(inputs)
        # 宽高坐标网格列表
        # 列表长度为3，表示三个特征层
        # 每个元素的shape: (3, 20or40or80, 20or40or80) 存储的值是映射到特征层尺寸的宽度和高度
        anchors_w = []
        anchors_h = []
        for layer in range(len(self.anchors_mask)):
            anchors_mask_layer = self.anchors_mask[layer]
            anchors_w.append(torch.stack([torch.tensor(self.anchors[:, 0][idx] / self.input_shape[0] *
                                                       self.feature_shape[layer], dtype=torch.float32).
                                         repeat(self.feature_shape[layer]).repeat(self.feature_shape[layer], 1)
                                          for idx in anchors_mask_layer], dim=0))
            anchors_h.append(torch.stack([torch.tensor(self.anchors[:, 1][idx] / self.input_shape[0] *
                                                       self.feature_shape[layer], dtype=torch.float32).
                                         repeat(self.feature_shape[layer]).repeat(self.feature_shape[layer], 1)
                                          for idx in anchors_mask_layer], dim=0))
        # 枚举每一个特征层
        for layer in range(len(self.anchors_mask)):
            # output_decode.shape = (1, 3, 20or40or80, 20or40or80, 25)
            output_decode = output[layer].view(batch_size, len(self.anchors_mask), 5 + self.num_classes,
                                               self.feature_shape[layer],
                                               self.feature_shape[layer]).permute(0, 1, 3, 4, 2).contiguous()
            # 当前特征层每个先验框的调整参数
            tx = torch.sigmoid(output_decode[..., 0])
            ty = torch.sigmoid(output_decode[..., 1])
            tw = torch.sigmoid(output_decode[..., 2])
            th = torch.sigmoid(output_decode[..., 3])
            conf = torch.sigmoid(output_decode[..., 4])
            pred_cls = torch.sigmoid(output_decode[..., 5:])
            # 保证设备统一
            FloatTensor = torch.cuda.FloatTensor if self.device == 'cuda:0' else torch.FloatTensor
            LongTensor = torch.cuda.LongTensor if self.device == 'cuda:0' else torch.LongTensor

            # 搞一个先验框的网格，方便调整坐标
            grid_x = torch.linspace(0, self.feature_shape[layer] - 1,
                                    self.feature_shape[layer]).repeat(self.feature_shape[layer], 1). \
                repeat(batch_size * len(self.anchors_mask), 1, 1).view(tx.shape).type(FloatTensor)
            grid_y = torch.linspace(0, self.feature_shape[layer] - 1,
                                    self.feature_shape[layer]).view(self.feature_shape[layer], 1). \
                repeat(1, self.feature_shape[layer]).repeat(batch_size * len(self.anchors_mask), 1, 1). \
                view(ty.shape).type(FloatTensor)
            grid_w = anchors_w[layer].unsqueeze(0).type(FloatTensor)
            grid_h = anchors_h[layer].unsqueeze(0).type(FloatTensor)

            # 调整坐标
            pred_boxes = FloatTensor(output_decode[..., :4].shape)
            pred_boxes[..., 0] = 2 * tx.data - 0.5 + grid_x
            pred_boxes[..., 1] = 2 * ty.data - 0.5 + grid_y
            pred_boxes[..., 2] = grid_w * (2 * tw.data) ** 2
            pred_boxes[..., 3] = grid_h * (2 * th.data) ** 2
            # 归一化到0~1
            _scale = torch.tensor(self.feature_shape[layer]). \
                repeat(self.feature_shape[layer], self.feature_shape[layer]).unsqueeze(2). \
                repeat(1, 1, 4).type(FloatTensor)
            # 将位置坐标与置信度和类别概率拼接起来
            out_boxes = torch.concat([pred_boxes / _scale, conf.data.unsqueeze(-1), pred_cls.data], dim=-1). \
                view(batch_size, -1, self.num_classes + 5)
            output_boxes.append(out_boxes)
        # output_boxes 列表，内有三个元素，分别是三个特征层经过解码后获得的预测框信息
        # shape: (bs, num_boxes_layer1, 25)
        # shape: (bs, num_boxes_layer2, 25)
        # shape: (bs, num_boxes_layer3, 25)
        return output_boxes
