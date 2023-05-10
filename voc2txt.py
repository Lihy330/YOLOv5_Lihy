import os
import random
from xml.dom.minidom import parse
from _utils._utils import get_classes_dict


# 生成训练集与测试集的索引
def get_split_datasets_random_index(total_num, split_percent=0.9):
    train_index_list = []
    val_index_list = []
    while len(train_index_list) != int(split_percent * total_num):
        train_index_list.append(random.randint(0, total_num - 1))
        train_index_list = list(set(train_index_list))
    for index in range(0, total_num):
        if index not in train_index_list:
            val_index_list.append(index)

    return train_index_list, val_index_list


def get_label_info(content, written_name):
    root = content.documentElement
    image_name = root.getElementsByTagName("filename")
    image_path = os.path.join(voc_img_abs_path, image_name[0].firstChild.data)
    label_infos = root.getElementsByTagName("object")
    with open(written_name, 'a', encoding='utf-8') as fw:
        fw.write(image_path + " ")

    # 枚举所有的object
    for label in label_infos:
        # 标签信息
        target = []
        # 获取到名字叫name的标签
        label_classes = label.getElementsByTagName("name")
        label_class_name = label_classes[0].firstChild.data
        # 获取到名字叫bndbox的标签
        box_infos = label.getElementsByTagName("bndbox")
        box_xmin = box_infos[0].getElementsByTagName("xmin")
        box_ymin = box_infos[0].getElementsByTagName("ymin")
        box_xmax = box_infos[0].getElementsByTagName("xmax")
        box_ymax = box_infos[0].getElementsByTagName("ymax")
        target.append(int(box_xmin[0].firstChild.data))
        target.append(int(box_ymin[0].firstChild.data))
        target.append(int(box_xmax[0].firstChild.data))
        target.append(int(box_ymax[0].firstChild.data))
        # 最后添加类别信息
        target.append(classes_dict[label_class_name])
        # 写入标签信息
        with open(written_name, 'a', encoding='utf-8') as fw:
            [fw.write(str(item) + ',') for item in target]
            fw.write(' ')

    with open(written_name, 'a', encoding='utf-8') as fw:
        fw.write("\n")


if __name__ == "__main__":

    classes_path = r"./model_data/voc_classes.txt"
    # 将类别映射成字典
    classes_dict = get_classes_dict(classes_path)
    # 数据集图片以及标签文件的存储路径（绝对路径以及相对路径）
    voc_img_abs_path = r"D:\YOLO\yolov5-pytorch-main\VOCdevkit\VOC2007\JPEGImages"
    voc_img_rel_path = r"./VOCdevkit/VOC2007/JPEGImages"
    voc_label_abs_path = r"D:\YOLO\yolov5-pytorch-main\VOCdevkit\VOC2007\Annotations"
    voc_label_rel_path = r"./VOCdevkit/VOC2007/Annotations"

    labels_list = os.listdir(voc_label_rel_path)
    num_labels = len(labels_list)

    # 生成标签文件的训练集测试集索引
    # 数据集划分的比例
    split_percent = 0.9
    train_index, val_index = get_split_datasets_random_index(num_labels, split_percent)
    # 处理训练集文件标签，写入voc_train_data.txt
    for tr in train_index:
        label_content = parse(os.path.join(voc_label_rel_path, labels_list[tr]))
        get_label_info(label_content, "voc_train_data.txt")

    # 处理测试集文件标签，写入voc_val_data.txt
    for val in val_index:
        label_content = parse(os.path.join(voc_label_rel_path, labels_list[val]))
        get_label_info(label_content, "voc_val_data.txt")
