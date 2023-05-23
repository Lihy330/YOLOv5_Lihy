import torch
from YoloNet import YOLOBody
from _utils._utils import get_anchors, collate_fn
from yolo_training import YOLO_Loss, weights_init
from _utils.dataloader import Yolo_Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


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
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    # 评价器，损失函数
    criterion = YOLO_Loss(input_shape, anchors, num_classes, anchors_mask, device)
    # 模型
    model = YOLOBody(3, 64, num_classes, anchors_mask).to(device)
    # 数据集
    yolo_dt = Yolo_Dataset(anchors, train_data_file_path, val_data_file_path, num_classes, anchors_mask, mode='train')
    yolo_dl = DataLoader(yolo_dt, batch_size=4, shuffle=False, collate_fn=collate_fn)
    # 优化器
    momentum = 0.937
    weight_decay = 5e-4
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-5, momentum=momentum, weight_decay=weight_decay)
    # 训练总迭代数量
    n_epochs = 200000

    # 初始化训练参数
    weights_init(model)

    # 训练 loop
    min_loss = 1e12
    for epoch in range(n_epochs):
        total_loss = 0
        yolo_dl = tqdm(yolo_dl, total=len(yolo_dl))
        loss_file = open('./training_loss.txt', 'a')
        batch_cnt = 0
        for index, (inputs, boxs, y_true) in enumerate(yolo_dl):
            # 将训练数据扔到设备中
            with torch.no_grad():
                inputs = inputs.to(device)
                y_true = [ann.to(device) for ann in y_true]
            batch_cnt += inputs.shape[0]
            loss_batch = 0
            optimizer.zero_grad()
            out = model(inputs)
            # 计算每个特征层的损失
            for layer in range(len(anchors_mask)):
                loss = criterion(layer, out[layer], y_true[layer])
                loss_batch += loss
            loss_batch.backward()
            if loss_batch.item() == torch.nan:
                with open('./error_log.txt', 'a') as f:
                    f.write(str(batch_cnt / 4))
                    f.write('\n')
            if loss_batch.item() != torch.nan:
                total_loss += loss_batch.item()
            yolo_dl.set_postfix(Loss=loss_batch.item() / batch_cnt)
            yolo_dl.update(1)
            optimizer.step()
        # 计算每一代的平均损失
        average_loss = total_loss / len(yolo_dl)
        loss_file.write(str(average_loss))
        loss_file.write('\n')
        loss_file.close()
        if average_loss < min_loss:
            min_loss = average_loss
            torch.save(model.state_dict(), 'yolov5_lihy_l_loss_' + str(average_loss) + '.pth')