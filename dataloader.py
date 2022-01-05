#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/21 上午11:36
# @Author  : zhangyunfei
# @File    : dataloader.py
# @Software: PyCharm
import os
import random
import xml.etree.ElementTree as ET

"""
    金盛兰吸盘数据处理模块
"""
# 类别名称
classes = ['green_fire', 'mf', 'yb', 'red_fire', 'cdz', 'yw']


# abs_path = os.getcwd()
# print(abs_path)
# current_work_dir = os.path.dirname(__file__)
# print(current_work_dir)

# 获取所有的文件路径
def travel_path(dir, files=None, extension_list=('jpg',), ignore_files=None):
    """
    :param dir: 文件路径
    :param files: 文件列表
    :param extension_list: 支持的文件格式
    :param ignore_files: 需要过滤的文件
    :return:
    """
    if ignore_files is None:
        ignore_files = []
    if files is None:
        files = []
    # 读取文件列表
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path):
            if '.' in path:
                _, extension = path.rsplit('.', 1)
                if extension.lower() in extension_list and file not in ignore_files:
                    files.append(path)
        elif os.path.isdir(path):
            travel_path(path, files, extension_list, ignore_files)


# 将数据集划分训练集和验证集
def split_data(files):
    """
    :param files:
    :return:
    """
    random.shuffle(files)
    # 计算比例系数，分割数据训练集和验证集
    ratio = 0.9
    offset = int(len(files) * ratio)
    train_data = files[:offset]
    val_data = files[offset:]
    return train_data, val_data


# 将xml标注格式转化成yolo数据格式
def convert(size, box):
    """
    将位置坐标转化成中心点坐标，宽高等，并做归一化处理
    :param size:图像尺寸
    :param box:标注框位置坐标[min_x,max_x,min_y,max_y]
    :return:
    """
    # 归一化系数
    dw = 1. / size[0]
    dh = 1. / size[1]
    # 获取中心点坐标
    center_x = (box[0] + box[1]) / 2.0
    center_y = (box[2] + box[3]) / 2.0
    box_w = box[1] - box[0]
    box_h = box[3] - box[2]
    # 归一化
    c_x = center_x * dw
    c_y = center_y * dh
    w = box_w * dw
    h = box_h * dh
    return [c_x, c_y, w, h]


total_cla = []
# 读取xml文件数据
def convert_annotation(xml_path, label_path):
    """
    读取xml标注信息，并将其转化
    :param xml_path: xml路径
    :param label_path: 保存txt的路径
    :return:
    """
    read_xml = open(xml_path, 'r', encoding='utf8')
    out_file = open(label_path, 'w')
    # xml结构
    tree = ET.parse(read_xml)
    root = tree.getroot()
    size = root.find('size')
    # 图像宽高
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    # 标注框
    for obj in root.iter('object'):
        bndbox = obj.find('bndbox')
        box = (float(bndbox.find('xmin').text), float(bndbox.find('xmax').text), float(bndbox.find('ymin').text),
               float(bndbox.find('ymax').text))
        # 标注越界修正
        if box[1] > width:
            box[1] = width
        if box[3] > height:
            box[3] = height
        label_data = convert((width, height), box)
        # 获取标注框的类别
        cls = obj.find('name').text
        total_cla.append(cls)

        # difficult = obj.find('difficult').text
        if cls not in classes:
            print('---------error:', xml_path)
            continue
        cls_id = classes.index(cls)
        # 写入文件
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in label_data]) + '\n')
    out_file.close()


# 主函数
def main():
    # 标注数据目录
    train_base_image_dir = '/Users/clustar/Desktop/项目/马钢密封件/mf_fire/train_images'
    train_base_xml_dir = '/Users/clustar/Desktop/项目/马钢密封件/mf_fire/train_annots'
    # base_dir = '../datasets/label_data'
    train_files = []
    travel_path(train_base_image_dir, train_files)


    test_base_image_dir = '/Users/clustar/Desktop/项目/马钢密封件/mf_fire/valid_images'
    test_base_xml_dir = '/Users/clustar/Desktop/项目/马钢密封件/mf_fire/valid_annots'
    # base_dir = '../datasets/label_data'
    test_files = []
    travel_path(test_base_image_dir, test_files)



    # 拆分数据集
    #train_data, val_data = split_data(files)
    # 创建存放label及训练文件
    train_label_dir = '/Users/clustar/Desktop/项目/马钢密封件/mf_fire/train_labels'
    os.makedirs(train_label_dir, exist_ok=True)
    # print(len(files), len(train_data), len(val_data))
    print(len(train_files))
    # 记录训练集
    with open('/Users/clustar/Desktop/项目/马钢密封件/mf_fire/train_labels/train.txt', 'w') as fp:
        # for i, item in enumerate(train_data):
        for i, item in enumerate(train_files):
            print("train  --->  begin process id:{}, path:{}".format(i, item))
            train_xml_path = item.replace(train_base_image_dir,train_base_xml_dir).replace('.jpg', '.xml')
            train_label_path = item.replace(train_base_image_dir,train_label_dir).replace('.jpg', '.txt')
            convert_annotation(train_xml_path, train_label_path)
            fp.write(item + '\n')


    test_label_dir = '/Users/clustar/Desktop/项目/马钢密封件/mf_fire/test_labels'
    os.makedirs(test_label_dir, exist_ok=True)
    # print(len(files), len(train_data), len(val_data))
    print(len(test_files))
    # 记录验证集
    with open('/Users/clustar/Desktop/项目/马钢密封件/mf_fire/train_labels/test.txt', 'w') as fp:
        for i, item in enumerate(test_files):
            val_xml_path = item.replace(test_base_image_dir,test_base_xml_dir).replace('.jpg', '.xml')
            print("val --->   begin process id:{}, path:{}".format(i, item))
            val_label_path = item.replace(test_base_image_dir,test_label_dir).replace('.jpg', '.txt')
            convert_annotation(val_xml_path, val_label_path)
            fp.write(item + '\n')



# def get_xml_name():
#     read_xml = open(xml_path, 'r', encoding='utf8')
#     root = tree.getroot()




if __name__ == '__main__':
    main()

    total_cla = list(set(total_cla))
    print(total_cla)
