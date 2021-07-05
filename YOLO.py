import time
from builtins import range

import cv2
import numpy as np
import check_segment
import os
import math
import pandas as pd
import json
import Common

images = []
folder = "01.2021-left"
class_name_path = 'yolov3.txt'
cfg_path = 'yolov4.cfg'
weights_path = 'yolov4.weights'
# result_path = 'F:/Artificial intelligence/MiAI_Yolo_1/yolo_beginner/Result 2'
result_path = 'F:/Artificial intelligence/MiAI_Yolo_1/yolo_beginner/Car_Detect_Result'


def rescale_frame(frame, scale):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimension = (width, height)
    return cv2.resize(frame, dimension, interpolation=cv2.INTER_AREA)


def bgr2rgb(color):
    return color[2], color[1], color[0]


my_scale = Common.my_scale
arr = Common.arr
listAxisPoint = [[]] * len(arr)
colorListLine = bgr2rgb((255, 255, 0))  # yellow
color_rectangle = bgr2rgb((255, 0, 0))
color_meters = bgr2rgb((0, 204, 0))
color_line_fragment = bgr2rgb((0, 102, 204))
count = 0
index = 0


def draw_circle(event, x, y, flags, param):
    img, arr = param
    global count
    for i in range(len(arr)):
        if abs(x - arr[i][0]) < 10 and event == cv2.EVENT_LBUTTONDOWN and arr[i][1] is False:
            cv2.circle(img, (arr[i][0], y), 8, colorListLine, -1)
            arr[i][1] = True
            listAxisPoint[i] = [arr[i][0], y]
            if i != 0 and arr[i - 1][1] is True:
                cv2.line(img, (listAxisPoint[i - 1][0], listAxisPoint[i - 1][1]),
                         (listAxisPoint[i][0], listAxisPoint[i][1]), colorListLine, 5)
                count = count + 1
            if i != len(arr) - 1 and arr[i + 1][1] is True:
                cv2.line(img, (listAxisPoint[i + 1][0], listAxisPoint[i + 1][1]),
                         (listAxisPoint[i][0], listAxisPoint[i][1]), colorListLine, 5)
                count = count + 1


def draw_circle_again(img):
    for i in range(len(arr)):
        cv2.circle(img, (listAxisPoint[i][0], listAxisPoint[i][1]), 8, colorListLine, -1)
        if i != 0:
            cv2.line(img, (listAxisPoint[i - 1][0], listAxisPoint[i - 1][1]),
                     (listAxisPoint[i][0], listAxisPoint[i][1]), colorListLine, 5)


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


list_sheet = [pd.DataFrame([[0]])] * (len(arr) -1)
list_coordinate_car = []


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    if label == "car" or label == "truck":
        for i in range(len(listAxisPoint) - 1):
            segment_one = [listAxisPoint[i], listAxisPoint[i + 1]]
            segment_two = [[x, y], [x, y_plus_h]]
            if check_segment.intersects(segment_one, segment_two):
                cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color_rectangle, 2)
                list_coordinate_car.append([x, y, x_plus_w, y_plus_h])
                cv2.putText(img, str(x_plus_w - x), (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_meters, 2)
                if (arr[i][0] <= x and x_plus_w <= arr[i + 1][0]) \
                        or (arr[i][0] <= x and (arr[i + 1][0] - x) > (x_plus_w - x) * 0.8):
                    a_row = pd.DataFrame([pd.Series([x_plus_w - x])])
                    list_sheet[i] = pd.concat([list_sheet[i], a_row], ignore_index=True)
                    print(x_plus_w - x, "fragment ", str(i + 1))
                if x > arr[i][0] and (x_plus_w - arr[i + 1][0]) > (x_plus_w - x) * 0.8:
                    a_row = pd.DataFrame([pd.Series([x_plus_w - x])])
                    list_sheet[i + 1] = pd.concat([list_sheet[i + 1], a_row], ignore_index=True)
                    print(x_plus_w - x, "fragment ", str(i + 2))


boxes = []  # boxes in for loop


def my_func_sort(e):
    return boxes[e][0]
    # boxes[indices[i][0]][0]


f1 = open("list_coordinate_car.txt", "w+")
f2 = open("list_axis_point.txt", "w+")
for filename in os.listdir(folder):
    start = time.time()
    index = index + 1
    if index % 35 != 1:
        continue
    image = cv2.imread(os.path.join(folder, filename))

    image = rescale_frame(image, my_scale)
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392
    with open(class_name_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    net = cv2.dnn.readNet(weights_path, cfg_path)
    blob = cv2.dnn.blobFromImage(image, scale, (608, 608), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    # Thực hiện xác định bằng HOG và SVM

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    header_img = "Image " + str(index)
    print(header_img)

    param = (image, arr)
    # for i in range(len(arr)):
    #     cv2.line(image, (arr[i][0], 0), (arr[i][0], math.floor(my_scale / 0.6 * 648)), color_line_fragment, 5)
    if index == 1:
        for i in range(len(arr)):
            cv2.line(image, (arr[i][0], 0), (arr[i][0], math.floor(my_scale / 0.6 * 648)), color_line_fragment, 5)
        cv2.namedWindow(winname=header_img)
        cv2.setMouseCallback(header_img, draw_circle, param=param)
        while True:
            cv2.imshow(header_img, image)
            k = cv2.waitKey(100)
            if k == 13 and count == len(listAxisPoint) - 1:  # 13 = enter
                a_line = json.dumps(listAxisPoint) + "\n"
                f2.write(a_line)
                break
    # else:
    #     draw_circle_again(image)

    arr_indices = []
    for i in indices:
        arr_indices.append(i[0])
    arr_indices.sort(key=my_func_sort)

    for i in arr_indices:
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

    a_line = json.dumps(list_coordinate_car) + "\n"
    f1.write(a_line)
    list_coordinate_car = []

    if index == 1:
        cv2.imshow(header_img, image)
        k = cv2.waitKey(0)

    result_file_name = str(index) + ".jpg"
    cv2.imwrite(os.path.join(result_path, result_file_name), image)
    end = time.time()
    print("YOLO Execution time: " + str(end - start))
    cv2.destroyAllWindows()
    # if index >= 1000:
    #     break
writer = pd.ExcelWriter('./length_of_car.xlsx', engine='xlsxwriter')
income_sheets = {}
for i in range(len(arr) - 1):
    list_sheet[i].columns = ['length of car']
    income_sheets["Fragment " + str(i + 1)] = list_sheet[i]
for sheet_name in income_sheets.keys():
    income_sheets[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)
writer.save()
f1.close()
f2.close()
