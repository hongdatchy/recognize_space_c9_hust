import json
import time
import cv2
import os
import Common


def bgr2rgb(color):
    return color[2], color[1], color[0]


result_path = 'F:/Artificial intelligence/MiAI_Yolo_1/yolo_beginner/Final_Result'

folder = "Car_Detect_Result"
# folder = "Result 2"
arr = Common.arr
f = open("list_coordinate_car.txt", "r")
lines = f.readlines()

f2 = open("list_axis_point.txt", "r")
list_point = json.loads(f2.read())

f3 = open("list_xl_xs.txt", "r")
list_xl_xs = json.loads(f3.read())

color_space = bgr2rgb((0, 204, 0))
color_meters = bgr2rgb((0, 204, 0))
color_final = bgr2rgb((102, 0, 102))
color_red = bgr2rgb((255, 0, 0))


def my_sort(e):
    return int(e.split(".")[0])


list_img_name = os.listdir(folder)
list_img_name.sort(key=my_sort)


def draw_space(n, index_arr, x, y, x_plus_w, y_plus_h):
    an_pha = 0.05  # % so với chiều cao của hình màu xanh
    A = int((y_plus_h - y) * (1 - (n + 1) * an_pha))
    x_final = x
    y_final = y
    xl_car = 0
    if x_final + list_xl_xs[index_arr - 1][1] < arr[index_arr][0]:
        xl_car = list_xl_xs[index_arr - 1][1]
    else:
        xl_car = int(arr[index_arr][0] + (1 - (arr[index_arr][0] - x_final) / list_xl_xs[index_arr - 1][1]) *
                     list_xl_xs[index_arr][1]) - x_final
    if n == 1 and x_plus_w - x >= xl_car:
        x_plus_w_final = x_final + xl_car
        y_final += int(an_pha * (y_plus_h - y))
        y_plus_h_final = y_final + A
        cv2.rectangle(image, (x_final, y_final), (x_plus_w_final, y_plus_h_final), color_final, 2)
        cv2.putText(image, "L", (x_final + 10, y_final + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_red, 2)
    else:
        x_plus_w_final = x_final
        for i in range(n):
            x_final = x_plus_w_final
            y_final += int(an_pha * (y_plus_h - y))
            if x_final + list_xl_xs[index_arr - 1][0] < arr[index_arr][0]:
                x_plus_w_final = x_final + list_xl_xs[index_arr - 1][0]
            else:
                x_plus_w_final = int(
                    arr[index_arr][0] + (1 - (arr[index_arr][0] - x_final) / list_xl_xs[index_arr - 1][0]) *
                    list_xl_xs[index_arr][0])
                index_arr += 1
            y_plus_h_final = y_final + A
            cv2.rectangle(image, (x_final, y_final), (x_plus_w_final, y_plus_h_final), color_final, 2)
            cv2.putText(image, "S", (x_final + 10, y_final + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_red, 2)


def draw_final(x, y, x_plus_w, y_plus_h):
    for i in range(len(arr) - 1):
        if arr[i][0] <= x and x_plus_w <= arr[i + 1][0]:  # space in one fragment
            if x_plus_w - x >= list_xl_xs[i][0]:
                n = int((x_plus_w - x) / list_xl_xs[i][0])
                draw_space(n, i + 1, x, y, x_plus_w, y_plus_h)
                # print("n" + str(n) + "i" + str(i))
        if x_plus_w > arr[i][0] > x >= arr[i - 1][0]:  # space in many fragment
            n = 0
            n += (arr[i][0] - x) / list_xl_xs[i - 1][0]
            for iii in range(i + 1, len(arr)):
                if arr[iii][0] > x_plus_w:
                    n += (x_plus_w - arr[iii - 1][0]) / list_xl_xs[iii - 1][0]
                    break
                else:
                    n += (arr[iii][0] - arr[iii - 1][0]) / list_xl_xs[iii - 1][0]
            n = int(n)
            # print("n" + str(n) + "i" + str(i))
            draw_space(n, i, x, y, x_plus_w, y_plus_h)


count = 0
for filename in list_img_name:
    start = time.time()
    image = cv2.imread(os.path.join(folder, filename))
    a_line = json.loads(lines[count])
    index = 0
    for coordinate_car in a_line:
        x = y = x_plus_w = y_plus_h = 0
        if index == 0 and coordinate_car[0] > 5:  # 5 pixel
            x = 0
            y = 2 * list_point[0][1] - coordinate_car[3]
            x_plus_w = coordinate_car[0]
            y_plus_h = coordinate_car[3]
            cv2.rectangle(image, (x, y), (x_plus_w, y_plus_h), color_space, 2)
            draw_final(x, y, x_plus_w, y_plus_h)
        if index == len(a_line) - 1 and arr[len(arr) - 1][0] - coordinate_car[2] > 5:  # 5 pixel
            x = coordinate_car[2]
            y = coordinate_car[1]
            x_plus_w = arr[len(arr) - 1][0]
            y_plus_h = 2 * list_point[len(list_point) - 1][1] - coordinate_car[1]
            cv2.rectangle(image, (coordinate_car[2], y), (x_plus_w, y_plus_h), color_space, 2)
            draw_final(x, y, x_plus_w, y_plus_h)
        if index != len(a_line) - 1:
            if a_line[index + 1][0] > a_line[index][2]:
                x = a_line[index][2]
                y = a_line[index][1]
                x_plus_w = a_line[index + 1][0]
                y_plus_h = a_line[index + 1][3]
                cv2.rectangle(image, (a_line[index][2], y), (x_plus_w, y_plus_h), color_space, 2)
                draw_final(x, y, x_plus_w, y_plus_h)
        index += 1

    cv2.imwrite(os.path.join(result_path, filename), image)
    count += 1
    end = time.time()
    print("Final Result Execution time:" + str(end - start))
    print("Image " + filename)
    cv2.namedWindow(filename, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(filename, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(filename, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
