import numpy as np
import torch


# 将类别映射成字典
def get_classes_dict(class_path):
    classes_dict = {}
    with open(class_path, "r") as fr:
        classes_name = fr.readlines()
        for idx in range(len(classes_name)):
            classes_name[idx] = classes_name[idx].strip()
        for idx, item in enumerate(classes_name):
            classes_dict[item] = idx
    return classes_dict


def get_anchors(anchors_path):
    with open(anchors_path, 'r') as fr:
        content = fr.readlines()
        # 将读取到的先验框尺寸信息矩阵的形状转换成(9,, 2)
        content = np.array(list(map(lambda x: int(x), content[0].split(',')))).reshape(9, 2)
    return content


# 保证整体取出数据集使用的数据整理打包函数
def collate_fn(data):
    images_list = [single_data[0] for single_data in data]
    annotations = [single_data[1] for single_data in data]
    # 对y_true进行打包处理
    # 列表的长度为3，表示三个特征层，每个索引位置的形状是(bs, 3, fs, fs, num_classes + 5)
    y_true = [[], [], []]
    for single_data in data:
        for idx in range(3):
            y_true[idx].append(single_data[2][idx])
    for index in range(len(y_true)):
        y_true[index] = torch.stack(y_true[index], dim=0)
    return torch.stack(images_list, dim=0), annotations, y_true
