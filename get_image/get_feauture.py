import dlib
from skimage import io
import numpy
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
import tensorflow as tf

output_dir='/Users/yicc/Desktop/code/可运行版本/fa_detect/get_image/faces/'

def create_testfile(num):
    dirPath = "/Users/yicc/Desktop/code/可运行版本/fa_detect/get_image/file_detect/"
    filename = str(num) + ".txt"
    file_path = dirPath + filename
    file = open(file_path,"w+")#(w+: 开头开始编辑，如不存在则创建)
    file.close()

def delete_testfile(num):
    dirPath = "/Users/yicc/Desktop/code/可运行版本/fa_detect/get_image/file_test/"
    filename = str(num) + ".txt"
    if(os.path.exists(output_dir+str(num)+'.jpg')):
        os.remove(dirPath + filename)

def file_exist(test_path):
    if os.path.exists(test_path):
        return 1
    else:
        return 0


def get_cut_images():
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    predictor_path = "/Users/yicc/Desktop/code/可运行版本/fa_detect/get_image/shape_predictor_68_face_landmarks.dat"

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # feel free to use any photo you want
    win = dlib.image_window()   #为什么有了x11还是不能用



    cwd = '/Users/yicc/Desktop/code/可运行版本/fa_detect/get_image/faces/'
    url = '/Users/yicc/Desktop/code/可运行版本/fa_detect/get_image/cut_images/'
    test_url = '/Users/yicc/Desktop/code/可运行版本/fa_detect/get_image/file_test/'
#   filenames=os.listdir(cwd)
#   filenames.sort(key=lambda x: int(x[:-4]))      #排序，这个很重要，一会取消注释
    name = 1
    index = 1
    while index > 0:
        active = True
        img_path = cwd + str(index) + ".jpg"
        test_path = test_url + str(index) + ".txt"
        while active:
            m = file_exist(test_path)
            if m == 1:

                img = io.imread(img_path)

                win.clear_overlay()
                win.set_image(img)

                # array of faces
                dets = detector(img, 1)
                print("In " + str(index) + ".jpg" + "," + "number of faces detected: {}".format(len(dets)))
                abc = plt.figure(num=1, figsize=(8, 5))
                for k, d in enumerate(dets):
                    shape = predictor(img, d)
                    np_shape = []
                    for i in shape.parts():
                        np_shape.append([i.x, i.y])
                    np_shape = numpy.array(np_shape)

                    plt.scatter(np_shape[:, 0], np_shape[:, 1], c='w', s=8)
                    # plt.plot(shape_graphed_np[:, 0], shape_graphed_np[:, 1], c='w')
                    print(np_shape)
                    ll_point = np_shape[36]
                    lr_point = np_shape[39]
                    rl_point = np_shape[42]
                    rr_point = np_shape[45]
                    ll_x = ll_point[0]
                    ll_y = ll_point[1]
                    lr_x = lr_point[0]
                    lr_y = lr_point[1]
                    rl_x = rl_point[0]
                    rl_y = rl_point[1]
                    rr_x = rr_point[0]
                    rr_y = rr_point[1]
                    cropped_l = img[ll_y - 50:lr_y + 50, ll_x - 10:lr_x + 10]  # 裁剪坐标为[y0:y1, x0:x1]
                    cropped_r = img[rl_y - 50:rr_y + 50, rl_x - 10:rr_x + 10]
                    cv2.imwrite(url + str(name) + '.jpg', cropped_l)
                    create_testfile(name)
                    name += 1
                    cv2.imwrite(url + str(name) + '.jpg', cropped_r)
                    create_testfile(name)
                    name += 1
                    win.add_overlay(shape)  # 绘制特征点

                    for idx, point in enumerate(np_shape):
                        pos = (point[0], point[1])
                        cv2.putText(img, str(idx), pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                                    fontScale=0.3, color=(0, 255, 0))
                        cv2.circle(img, pos, 3, color=(255, 255, 0))
                    win.set_image(img)
                active = False
                delete_testfile(index)
            else:
                continue
        index += 1


if __name__ == '__main__':
    get_cut_images()
    # plt.show()

#np_shape是一个列表，里面包含68个特征点的坐标数组，
#提取出特定点的坐标，给一定量的移动，剪切即可。
#36,39   42,45   ，两只眼的左右两边
#图片左上角为原点
