import cv2
import dlib
import os
import sys
import random
import threading

output_dir = '/Users/yicc/Desktop/code/可运行版本/fa_detect/get_image/faces/'
def create_testfile(num):
    dirPath = "/Users/yicc/Desktop/code/可运行版本/fa_detect/get_image/file_test/"
    filename = str(num) + ".txt"
    file_path = dirPath + filename
    file = open(file_path, "w+")#(w+: 开头开始编辑，如不存在则创建)
    file.close()


def get_face():
    lock = threading.Lock()

    size = 512  #原来是64

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


#使用dlib自带的frontal_face_detector作为我们的特征提取器
    detector = dlib.get_frontal_face_detector()
# 打开摄像头 参数为输入流，可以为摄像头或视频文件
    camera = cv2.VideoCapture(0)

    index = 1   #有参数时这里不要
    while True:
        if (index > 0):
            print('Being processed picture %s' % index)
        # 从摄像头读取照片
            success, img = camera.read()
        # 转为灰度图片
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 使用detector进行人脸检测
            dets = detector(gray_img, 1)

            for i, d in enumerate(dets):
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0

                face = img[x1:y1,x2:y2]
            # 调整图片的对比度与亮度， 对比度与亮度值都取随机数，这样能增加样本的多样性
#           face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))

                face = cv2.resize(face, (size,size))

                cv2.imshow('image', face)
                cv2.imwrite(output_dir+str(index)+'.jpg', face)
                if os.path.exists(output_dir+str(index)+'.jpg'):
                    create_testfile(index)

                index += 1
            key = cv2.waitKey(50) & 0xff    ####50ms循环一次，1s拍20张人脸照片，产生40个元素，取3s分析，有60张人脸照片，120个元素，2s闭眼则疲劳，即80个闭眼图则疲劳
            if key == 27:
                break
        else:
            print('Finished!')
            break

if __name__ == '__main__':
    get_face()


#[1,1,1,1,1,1,1]二个元素代表一个眼睛
