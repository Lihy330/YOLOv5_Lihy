import numpy as np


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
