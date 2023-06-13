# -*- coding: utf-8 -*-
# !/usr/bin/python3

import os
import cv2
import hilens
from yolo3 import Yolo3






def run(work_path):
    # 系统初始化，参数要与创建技能时填写的检验值保持一致
    hilens.init("hello")

    # 如果需要将检测结果保存到obs，flag_obs设置为True
    # 注意：保存obs这一操作只能在HiLens Studio执行，部署到HiLens Kit上无法执行


    # 初始化自带摄像头与HDMI显示器
    # hilens studio中VideoCapture如果不填写参数，则默认读取test/camera0.mp4文件
    # 在hilens kit中不填写参数则读取本地摄像头
    camera = hilens.VideoCapture()
    display = hilens.Display(hilens.HDMI)

    # 初始化模型
    #model_path = os.path.join(work_path, 'model/model.json')
    #mask_model = Yolo3(model_path)

    # 将检测结果存到json数据中
    frame_index = 0
    json_bbox_list = []

    while True:
        frame_index += 1
        #try:
        #input_yuv = camera.read()  # 读取一帧图片(YUV NV21格式)
        output_yuv = camera.read()  # 读取一帧图片(YUV NV21格式)
        # except Exception:
        #     hilens.info('last frame~~~')
        #     break

# """         img_rgb = cv2.cvtColor(input_yuv, cv2.COLOR_YUV2RGB_NV21)  # 转为RGB格式
#         bboxes = mask_model.infer(img_rgb)  # 获取检测结果

#         json_bbox = mask_model.convert_to_json(bboxes, frame_index)
#         json_bbox_list.append(json_bbox)  # 将检测结果转为json格式

#         img_rgb = mask_model.draw_mask_info(img_rgb, bboxes)  # 在图像上显示口罩佩戴信息
#         output_yuv = hilens.cvt_color(img_rgb, hilens.RGB2YUV_NV21) """
        display.show(output_yuv)  # 显示到屏幕上

    hilens.terminate()


if __name__ == "__main__":
    run(os.getcwd())
