import torch
from YoloNet import YOLOBody
from _utils._utils import get_anchors, collate_fn
from yolo_training import YOLO_Loss
from _utils.dataloader import Yolo_Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os


if __name__ == "__main__":
    # 先验框索引
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    # 类别数量
    num_classes = 20
    # 网络输入图像的宽高尺寸
    input_shape = [640, 640]
    # 先验框尺寸存放的路径
    # anchors_path = r"D:\YOLO\yolov5-pytorch-main\model_data\yolo_anchor.txt"
    anchors_path = r"./model_data/yolo_anchor.txt"
    # 训练以及测试数据集图片路径以及标签存放文件路径
    train_data_file_path = r"./voc_train_data.txt"
    val_data_file_path = r"./voc_val_data.txt"

    # 获取先验框尺寸
    anchors = get_anchors(anchors_path)
    # 设备
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # 评价器，损失函数
    criterion = YOLO_Loss(input_shape, anchors, num_classes, anchors_mask, device)
    # 模型
    model = YOLOBody(3, 64, num_classes, anchors_mask).to(device)
    # 数据集
    yolo_dt = Yolo_Dataset(anchors, train_data_file_path, val_data_file_path, num_classes, anchors_mask, mode='train')
    yolo_dl = DataLoader(yolo_dt, batch_size=1, shuffle=False, collate_fn=collate_fn)
    yolo_dl = tqdm(yolo_dl, total=len(yolo_dl))
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    # 训练总迭代数量
    n_epochs = 200000

    # 训练 loop
    min_loss = 1e9
    for epoch in range(n_epochs):
        loss_file = open('./training_loss.txt', 'a')
        loss_batch = 0
        batch_cnt = 0
        for inputs, boxs, y_true in yolo_dl:
            # 将训练数据扔到设备中
            with torch.no_grad():
                inputs = inputs.cuda(device)
                y_true = [ann.cuda(device) for ann in y_true]
            batch_cnt += inputs.shape[0]
            optimizer.zero_grad()
            out = model(inputs)
            # 计算每个特征层的损失
            for layer in range(len(anchors_mask)):
                loss = criterion(layer, out[layer], y_true[layer])
                loss_batch += loss
            loss_batch.backward()
            yolo_dl.set_postfix(Loss=loss_batch.item())
            optimizer.step()
        loss_file.write(loss_batch.item() + '\n')
        loss_file.close()
        if loss_batch < min_loss:
            min_loss = loss_batch.item()
            torch.save(model.state_dict(), 'yolov5_lihy_l.pth')
