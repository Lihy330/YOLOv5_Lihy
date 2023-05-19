import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
from torchvision import transforms
import numpy as np
from _utils import get_anchors


# map辅助函数
def split_images_path_func(data):
    data_list = data.split(' ')
    return data_list[0]


# map辅助函数
def split_single_annotation_func(data):
    single_annotation_list = data.split(',')
    # 将字符串批量处理成int
    new_single_annotation_list = list(map(lambda x: int(x), single_annotation_list[:-1]))
    return new_single_annotation_list


# map辅助函数
def split_annotations_path_func(data):
    data_list = data.split(' ')
    annotations = list(map(split_single_annotation_func, data_list[1:-1]))
    return annotations


# map辅助函数
def handle_annotations_numpy(data):
    annotations_tensor = np.array(data)
    return annotations_tensor


# map辅助函数
def handle_annotations_tensor(data):
    annotations_tensor = torch.LongTensor(data)
    return annotations_tensor


def get_images_path(path):
    fr = open(path, 'r', encoding='utf-8')
    content = fr.readlines()
    fr.close()
    # 提取图片路径
    images_path = list(map(split_images_path_func, content))
    return images_path


def get_annotations(path):
    fr = open(path, 'r', encoding='utf-8')
    content = fr.readlines()
    fr.close()
    # 提取标签信息
    annotations = list(map(split_annotations_path_func, content))
    return annotations


def draw_origin_image_label(image, labels, input_shape):
    image = transforms.Resize(input_shape)(image)
    draw = ImageDraw.Draw(image)
    for label in labels:
        draw.rectangle([label[0], label[1], label[2], label[3]], outline=(0, 255, 0), width=3)
    image.show()


# 该方法用来将标签框的坐标参数同样进行缩放，保持与图片缩放一致
    # annotations 未处理的标签框
    # resized_shape 表示图片缩放后的尺寸大小
    # image_shape 表示原图的尺寸大小
def resize_annotations(annotations, resized_shape, image_shape):
    # 注意Image读入的图片尺寸信息是宽高，而tensor是行列，因此是高宽
    iw, ih = image_shape[0], image_shape[1]
    resized_w, resized_h = resized_shape[1], resized_shape[0]
    # 分别计算宽和高的缩放比例
    scale_w = resized_w / iw
    scale_h = resized_h / ih
    resized_annotations = np.zeros_like(annotations)
    resized_annotations[:, 0] = annotations[:, 0] * scale_w
    resized_annotations[:, 2] = annotations[:, 2] * scale_w
    resized_annotations[:, 1] = annotations[:, 1] * scale_h
    resized_annotations[:, 3] = annotations[:, 3] * scale_h
    resized_annotations[:, 4] = annotations[:, 4]
    return resized_annotations


class Yolo_Dataset(Dataset):
    def __init__(self, train_data_file_path, val_data_file_path, anchors_path, num_classes,
                 anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]], mode='train'):
        super(Yolo_Dataset, self).__init__()
        # 数据集路径
        self.train_data_file_path = train_data_file_path
        self.val_data_file_path = val_data_file_path
        # 先验框路径
        self.anchors_path = anchors_path
        # 先验框尺寸
        self.anchors = get_anchors(self.anchors_path)
        # 先验框索引
        self.anchors_mask = anchors_mask
        # 网络输入的图像尺寸
        self.input_shape = [640, 640]
        # 类别数量
        self.num_classes = num_classes
        # 正负样本阈值
        self.threshold = 4

        if mode == 'train':
            self.images_path = get_images_path(self.train_data_file_path)
            self.annotations = get_annotations(self.train_data_file_path)
        else:
            self.images_path = get_images_path(self.val_data_file_path)
            self.annotations = get_annotations(self.val_data_file_path)
        # 将标签annotations处理成numpy，形状是(num_boxes, 5)
        self.annotations_numpy = list(map(handle_annotations_numpy, self.annotations))

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        image = Image.open(self.images_path[index])
        transform = transforms.Compose([
            transforms.Resize((640, 640)),
            # 转成tensor时已经自动归一化
            transforms.ToTensor()
        ])
        image_tensor = transform(image)
        # 对标签同样需要进行Resize处理
        resized_annotations = resize_annotations(self.annotations_numpy[index], self.input_shape, image.size)
        # # 将标签绘制在原图上查看缩放效果
        # draw_origin_image_label(image.copy(), resized_annotations, self.input_shape)
        # 先将标签坐标信息归一化，并转换成角点坐标
        box = np.zeros_like(resized_annotations, dtype=np.float32)
        # 归一化
        box[:, 4] = resized_annotations[:, 4]
        box[:, :-1] = resized_annotations[:, :-1] / self.input_shape[0]
        # 转换成中心点坐标
        temp_width = box[:, 2] - box[:, 0]
        temp_height = box[:, 3] - box[:, 1]
        box[:, 0] = 0.5 * (box[:, 0] + box[:, 2])
        box[:, 1] = 0.5 * (box[:, 1] + box[:, 3])
        box[:, 2] = temp_width
        box[:, 3] = temp_height
        # 获取到每个标签框与特征层、先验框、网格点的对应情况，方便后续计算损失
        y_true = self.get_targets(box)
        print(y_true[0].shape)
        return image_tensor, box, y_true

    # targets表示当前图片的标签，形状是(num_boxes, 5)  '5' => xmin, ymin, xmax, ymax, class_index
    def get_targets(self, targets):
        feature_layers = len(self.anchors_mask)
        # 特征层尺寸：[20, 40, 80]
        feature_shape = [int(self.input_shape[0] / num) for num in [32, 16, 8]]
        # y_true是一个列表，每个元素代表一个特征层上与标签框的对应信息
        # shape: ((3, 20, 20, 25), (3, 40, 40, 25), (3, 80, 80, 25))
        y_true = [
            torch.zeros(
                (len(self.anchors_mask[layer]), feature_shape[layer], feature_shape[layer], self.num_classes + 5),
                dtype=torch.float32) for layer in range(feature_layers)]
        # 防止多个标签框占据同一个网格，需要记录每个网格上的标签框与先验框宽高比的最小值
        best_ratios = [
            torch.zeros(
                (len(self.anchors_mask[layer]), feature_shape[layer], feature_shape[layer]),
                dtype=torch.float32) for layer in range(feature_layers)]
        # 枚举每一个特征层
        for layer in range(feature_layers):
            # 将标签框的数据映射到当前特征层的尺寸
            box = np.zeros_like(targets, dtype=np.float32)
            box[:, :-1] = targets[:, :-1] * feature_shape[layer]
            box[:, -1] = targets[:, -1]
            box = torch.from_numpy(box)
            # 将先验框尺寸映射到当前特征层
            feature_anchors = ((self.anchors[self.anchors_mask[layer]]) / self.input_shape[0]) * feature_shape[layer]
            # 计算标签框与先验框的宽高比
            # np.expand_dims(box[:, 2:4], 1)  shape: (num_boxes, 2) => (num_boxes, 1, 2)  这样才能通过广播计算
            ratios_box_anchors = (torch.unsqueeze(box[:, 2:4], 1) / feature_anchors)
            # 计算先验框与标签框的宽高比
            ratios_anchors_box = (feature_anchors / torch.unsqueeze(box[:, 2:4], 1))
            # 拼接起来两组比值，同时求出最大值
            ratios = torch.max(torch.concat([ratios_box_anchors, ratios_anchors_box], 2), 2)[0]
            # 枚举每一个标签框
            for label_box_index in range(box.shape[0]):
                label_box_mask = ratios[label_box_index] < self.threshold
                # 如果全部都大于阈值，就取最小的作为正样本
                if (label_box_mask == False).all():
                    label_box_mask = ratios[label_box_index] <= torch.min(ratios[label_box_index])
                # 枚举每一个先验框
                for anchor_index in range(len(label_box_mask)):
                    # 如果当前先验框是负样本，不予处理
                    if not label_box_mask[anchor_index]:
                        continue
                    # 如果当前先验框是正样本
                    # 那么就判断当前标签框落在了哪一个网格内
                    local_x = int(box[label_box_index][0])
                    local_y = int(box[label_box_index][1])
                    # 如果当前网格已经被其他标签框占据，需要判断当前标签框与当前先验框的宽之比高之比最小值是否更小
                    #   如果更小，那么就将当前网格上的标签框替换为当前枚举到的标签框，否则不处理
                    # 已经占据
                    if y_true[layer][anchor_index, local_y, local_x, 0]:
                        # 发现比值更小
                        if ratios[label_box_index][anchor_index] < best_ratios[layer][anchor_index, local_y, local_x]:
                            y_true[layer][anchor_index, local_y, local_x] = 0
                        else:
                            continue
                    # 当前网格没有被占据
                    y_true[layer][anchor_index, local_y, local_x, 0] = box[label_box_index][0]
                    y_true[layer][anchor_index, local_y, local_x, 1] = box[label_box_index][1]
                    y_true[layer][anchor_index, local_y, local_x, 2] = box[label_box_index][2]
                    y_true[layer][anchor_index, local_y, local_x, 3] = box[label_box_index][3]
                    # 置信度
                    y_true[layer][anchor_index, local_y, local_x, 4] = 1
                    # 类别
                    y_true[layer][anchor_index, local_y, local_x, box[label_box_index][4].long() + 5] = 1
                    # 保存当前标签框与先验框的宽之比高之比的值，用于后续判重处理
                    best_ratios[layer][anchor_index, local_y, local_x] = ratios[label_box_index][anchor_index]
        return y_true


# 保证整体取出数据集使用的数据整理打包函数
def collate_fn(data):
    images_list = [single_data[0] for single_data in data]
    annotations = [single_data[1] for single_data in data]
    return torch.stack(images_list, dim=0), tuple(annotations)


if __name__ == "__main__":
    train_data_file_path = r"../voc_train_data.txt"
    val_data_file_path = r"../voc_val_data.txt"
    anchors_path = r"../model_data/yolo_anchor.txt"
    num_classes = 20
    dt = Yolo_Dataset(train_data_file_path, val_data_file_path, anchors_path, num_classes)
    dl = DataLoader(dt, batch_size=2, shuffle=False, collate_fn=collate_fn)
    dt[1291]
    # for idx, (images, labels) in enumerate(dl):
    #     print(labels)
    #     break
