import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
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


class Yolo_Dataset(Dataset):
    def __init__(self, train_data_file_path, val_data_file_path, anchors_path, num_classes, anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]], mode='train'):
        super(Yolo_Dataset, self).__init__()
        # 数据集路径
        self.train_data_file_path = train_data_file_path
        self.val_data_file_path = val_data_file_path
        # 先验框路径
        self.anchors_path = anchors_path
        self.anchors = get_anchors(self.anchors_path)
        self.anchors_mask = anchors_mask
        self.input_shape = [640, 640]
        self.num_classes = num_classes
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
        image = Image.open(self.images_path[0])
        transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor()
        ])
        image_tensor = transform(image)
        # 先将标签坐标信息归一化，并转换成角点坐标
        box = np.zeros_like(self.annotations_numpy[index], dtype=np.float32)
        # 归一化
        box[:, 4] = self.annotations_numpy[index][:, 4]
        box[:, :-1] = self.annotations_numpy[index][:, :-1] / self.input_shape[0]
        # 转换成中心点坐标
        temp_width = box[:, 2] - box[:, 0]
        temp_height = box[:, 3] - box[:, 1]
        box[:, 0] = 0.5 * (box[:, 0] + box[:, 2])
        box[:, 1] = 0.5 * (box[:, 1] + box[:, 3])
        box[:, 2] = temp_width
        box[:, 3] = temp_height
        # 获取到每个标签框与特征层、先验框、网格点的对应情况，方便后续计算损失
        self.get_targets(box)

        return image_tensor, box

    # targets表示当前图片的标签，形状是(num_boxes, 5)  '5' => xmin, ymin, xmax, ymax, class_index
    def get_targets(self, targets):
        feature_layers = len(self.anchors_mask)
        # 特征层尺寸：[80, 40, 20]
        feature_shape = [int(self.input_shape[0] / num) for num in [32, 16, 8]]
        # y_true是一个列表，每个元素代表一个特征层上与标签框的对应信息
        # shape: ((3, 80, 80, 25), (3, 40, 40, 25), (3, 20, 20, 25))
        y_true = [np.zeros((len(self.anchors_mask[layer]), feature_shape[layer], feature_shape[layer], self.num_classes+5),
                           dtype=np.float32) for layer in range(feature_layers)]
        # 枚举每一个特征层
        for layer in range(feature_layers):
            # 将标签框的数据映射到当前特征层的尺寸
            box = np.zeros_like(targets, dtype=np.float32)
            box[:, :-1] = targets[:, :-1] * feature_shape[layer]
            box[:, -1] = targets[:, -1]
            # 将先验框尺寸映射到当前特征层
            feature_anchors = ((self.anchors[self.anchors_mask[layer]]) / self.input_shape[0]) * feature_shape[layer]
            print(feature_anchors)
            print(box)
            break

        return y_true[0].shape, y_true[1].shape, y_true[2].shape


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
    dt[0]
    # for idx, (images, labels) in enumerate(dl):
    #     print(labels)
    #     break