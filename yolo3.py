# -*- coding: utf-8 -*-
# !/usr/bin/python3
# utils for yolo3 detection

import os
import cv2
import hilens
import numpy as np


class Yolo3:
    def __init__(self, model_path):
        '''
        根据模型地址初始化Yolo3类的实例
        除检测类别外，本模板中的yolo3模型相关参数与ModelArts AI Gallery中的YOLOv3_ResNet18算法相同，
        即使用该算法训练出的模型可直接用于本模板
        '''
        # 初始化检测模型
        self.det_model = hilens.Model(model_path)

        # 模型输入尺寸
        self.net_h = 352
        self.net_w = 640

        # 检测模型的类别
        self.class_names = ["face", "mask", "person"]
        self.class_num = len(self.class_names)

        # 检测模型的anchors，用于解码出检测框
        self.stride_list = [8, 16, 32]
        anchors_1 = np.array(
            [[10, 13],  [16, 30],   [33, 23]]) / self.stride_list[0]
        anchors_2 = np.array(
            [[30, 61],  [62, 45],   [59, 119]]) / self.stride_list[1]
        anchors_3 = np.array(
            [[116, 90], [156, 198], [163, 326]]) / self.stride_list[2]
        self.anchor_list = [anchors_1, anchors_2, anchors_3]

        # 检测框的输出阈值与NMS筛选阈值
        self.conf_threshold = 0.3
        self.iou_threshold = 0.4
        self.face_cover_threshold = 0.9
        self.mask_cover_threshold = 0.6

    def infer(self, img_rgb):
        '''模型推理，并进行后处理得到检测框'''
        img_preprocess, img_w, img_h = self.preprocess(img_rgb)
        output = self.det_model.infer([img_preprocess.flatten()])
        bboxes = self.get_result(output, img_w, img_h)
        return bboxes

    def preprocess(self, img_rgb):
        '''图片预处理：缩放到模型输入尺寸'''
        h, w, c = img_rgb.shape
        new_image = cv2.resize(img_rgb, (self.net_w, self.net_h))
        return new_image, w, h

    def overlap(self, x1, x2, x3, x4):
        left = max(x1, x3)
        right = min(x2, x4)
        return right - left

    def cal_iou(self, box1, box2):
        '''计算两个矩形框的IOU'''
        w = self.overlap(box1[0], box1[2], box2[0], box2[2])
        h = self.overlap(box1[1], box1[3], box2[1], box2[3])
        if w <= 0 or h <= 0:
            return 0
        inter_area = w * h
        union_area = (box1[2] - box1[0]) * (box1[3] - box1[1]) + \
            (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter_area
        return inter_area * 1.0 / union_area

    def cover_ratio(self, box1, box2):
        '''计算两个矩形框的IOU与box2区域的比值'''
        w = self.overlap(box1[0], box1[2], box2[0], box2[2])
        h = self.overlap(box1[1], box1[3], box2[1], box2[3])
        if w <= 0 or h <= 0:
            return 0
        inter_area = w * h
        small_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter_area * 1.0 / small_area

    def apply_nms(self, all_boxes, thres):
        '''使用NMS筛选检测框'''
        res = []

        for cls in range(self.class_num):
            cls_bboxes = all_boxes[cls]
            sorted_boxes = sorted(cls_bboxes, key=lambda d: d[5])[::-1]

            p = dict()
            for i in range(len(sorted_boxes)):
                if i in p:
                    continue

                truth = sorted_boxes[i]
                for j in range(i+1, len(sorted_boxes)):
                    if j in p:
                        continue
                    box = sorted_boxes[j]
                    iou = self.cal_iou(box, truth)
                    if iou >= thres:
                        p[j] = 1

            for i in range(len(sorted_boxes)):
                if i not in p:
                    res.append(sorted_boxes[i])
        return res

    def decode_bbox(self, conv_output, anchors, img_w, img_h):
        '''从模型输出的特征矩阵中解码出检测框的位置、类别、置信度等信息'''
        def _sigmoid(x):
            s = 1 / (1 + np.exp(-x))
            return s

        h, w, _ = conv_output.shape
        pred = conv_output.reshape((h * w, 3, 5+self.class_num))

        pred[..., 4:] = _sigmoid(pred[..., 4:])
        pred[..., 0] = (_sigmoid(pred[..., 0]) +
                        np.tile(range(w), (3, h)).transpose((1, 0))) / w
        pred[..., 1] = (_sigmoid(pred[..., 1]) + np.tile(np.repeat(
                        range(h), w), (3, 1)).transpose((1, 0))) / h
        pred[..., 2] = np.exp(pred[..., 2]) * anchors[:,
                                                      0:1].transpose((1, 0)) / w
        pred[..., 3] = np.exp(pred[..., 3]) * anchors[:,
                                                      1:2].transpose((1, 0)) / h

        bbox = np.zeros((h * w, 3, 4))
        bbox[..., 0] = np.maximum(
            (pred[..., 0] - pred[..., 2] / 2.0) * img_w, 0)     # x_min
        bbox[..., 1] = np.maximum(
            (pred[..., 1] - pred[..., 3] / 2.0) * img_h, 0)     # y_min
        bbox[..., 2] = np.minimum(
            (pred[..., 0] + pred[..., 2] / 2.0) * img_w, img_w)  # x_max
        bbox[..., 3] = np.minimum(
            (pred[..., 1] + pred[..., 3] / 2.0) * img_h, img_h)  # y_max

        pred[..., :4] = bbox
        pred = pred.reshape((-1, 5+self.class_num))
        pred[:, 4] = pred[:, 4] * pred[:, 5:].max(1)    # 类别
        pred = pred[pred[:, 4] >= self.conf_threshold]
        pred[:, 5] = np.argmax(pred[:, 5:], axis=-1)    # 置信度

        all_boxes = [[] for ix in range(self.class_num)]
        for ix in range(pred.shape[0]):
            box = [int(pred[ix, iy]) for iy in range(4)]
            box.append(int(pred[ix, 5]))
            box.append(pred[ix, 4])
            all_boxes[box[4]-1].append(box)

        return all_boxes

    def get_result(self, model_outputs, img_w, img_h):
        '''从模型输出中得到检测框'''

        num_channel = 3 * (self.class_num + 5)
        all_boxes = [[] for ix in range(self.class_num)]
        for ix in range(3):
            if not os.getenv("SKILL_NAME"):  # Studio-c76环境，模型输出格式为HWC
                hilens.info('HiLens Studio~~~')
                pred = model_outputs[2-ix].reshape(
                    (self.net_h // self.stride_list[ix], self.net_w // self.stride_list[ix], num_channel))
            else:  # Kit-c3x环境，模型输出格式为CHW，统一转换为HWC
                hilens.info('HiLens Kit~~~')
                pred = model_outputs[2-ix].reshape((num_channel, self.net_h // self.stride_list[ix],
                                                   self.net_w // self.stride_list[ix])).transpose((1, 2, 0))
            anchors = self.anchor_list[ix]
            boxes = self.decode_bbox(pred, anchors, img_w, img_h)
            all_boxes = [all_boxes[iy] + boxes[iy]
                         for iy in range(self.class_num)]

        res = self.apply_nms(all_boxes, self.iou_threshold)
        return res

    def draw_mask_info(self, img_data, bboxes):
        '''在图中画出口罩佩戴信息'''
        thickness = 2
        font_scale = 1
        text_font = cv2.FONT_HERSHEY_SIMPLEX
        for bbox in bboxes:
            label = int(bbox[4])
            x_min = int(bbox[0])
            y_min = int(bbox[1])
            x_max = int(bbox[2])
            y_max = int(bbox[3])

            if label == 2:  # person
                face_flag = False
                face_box = None
                for bbox2 in bboxes:
                    if int(bbox2[4]) == 0 and \
                            self.cover_ratio(bbox, bbox2) >= self.face_cover_threshold:
                        face_flag = True
                        face_box = bbox2
                if not face_flag:
                    cv2.rectangle(img_data, (x_min, y_min),
                                  (x_max, y_max), (255, 255, 0), thickness)
                    cv2.putText(img_data, 'unknown', (x_min, y_min-20),
                                text_font, font_scale, (255, 255, 0), thickness)
                else:
                    has_mask = False
                    mask_box = None
                    for bbox3 in bboxes:
                        if int(bbox3[4]) == 1 and \
                                self.cover_ratio(face_box, bbox3) >= self.mask_cover_threshold:
                            has_mask = True
                            mask_box = bbox3
                    if has_mask:
                        cv2.putText(img_data, 'has_mask', (x_min, y_min-20),
                                    text_font, font_scale, (0, 255, 0), thickness)
                        cv2.rectangle(img_data, (x_min, y_min),
                                      (x_max, y_max), (0, 255, 0), thickness)
                        cv2.rectangle(img_data, (face_box[0], face_box[1]),
                                      (face_box[2], face_box[3]), (0, 127, 127), thickness)
                        cv2.rectangle(img_data, (mask_box[0], mask_box[1]),
                                      (mask_box[2], mask_box[3]), (0, 255, 255), thickness)
                    else:
                        cv2.putText(img_data, 'no_mask', (x_min, y_min-20),
                                    text_font, font_scale, (255, 0, 0), thickness)
                        cv2.rectangle(img_data, (x_min, y_min),
                                      (x_max, y_max), (255, 0, 0), thickness)

        return img_data

    def draw_boxes(self, img_data, bboxes):
        '''在图中画出检测框'''
        thickness = 2
        for bbox in bboxes:
            x_min = int(bbox[0])
            y_min = int(bbox[1])
            x_max = int(bbox[2])
            y_max = int(bbox[3])
            # label = int(bbox[4])
            # score = bbox[5]
            cv2.rectangle(img_data, (x_min, y_min),
                          (x_max, y_max), (0, 0, 255), thickness)

        return img_data

    def convert_to_json(self, bboxes, frame_index):
        json_bbox = {'frame_id': frame_index}
        bbox_list = []
        for bbox in bboxes:
            bbox_info = {}
            bbox_info['x_min'] = int(bbox[0])
            bbox_info['y_min'] = int(bbox[1])
            bbox_info['x_max'] = int(bbox[2])
            bbox_info['y_max'] = int(bbox[3])
            bbox_info['label'] = int(bbox[4])
            bbox_info['score'] = int(bbox[5] * 100)
            bbox_list.append(bbox_info)
        json_bbox['bboxes'] = bbox_list
        return json_bbox
