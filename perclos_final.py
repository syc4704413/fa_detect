#####这个代码读取cut_images里的图片，送入到模型中，得出结论，一定时间内多少张
#####这里可以import time 一下
###大概步骤是：测出结果，若为什么，则加1
import os
import numpy as np
import tensorflow as tf
import fa_detect.input_data
import fa_detect.model
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io

test_dir = '/Users/yicc/Desktop/code/可运行版本/fa_detect/get_image/cut_images/'
train_logs_dir = '/Users/yicc/Desktop/code/可运行版本/fa_detect/train_log'

def file_exist(test_path):
    if os.path.exists(test_path):
        return 1
    else:
        return 0

#def test_one_image():
def test_images():
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    #filenames=os.listdir(test_dir)
   # filenames.sort(key=lambda x: int(x[:-4]))      #排序，这个很重要，一会取消注释

    #这里创建一个列表，用来存储最后的值
    perclos = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
               1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
               1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,]       #这里要根据perclos判别方法来改，现在写了10个元素，对应5张照片的时间（双眼）

    index = 1
    while index >0:
        active = True
        text_path = '/Users/yicc/Desktop/code/可运行版本/fa_detect/get_image/file_detect/'
        txt_path = text_path+str(index)+".txt"
        while active:
            k = file_exist(txt_path)
            if k == 1:
                img_path = test_dir + str(index)+".jpg"  # 每一个图片的地址
                img = Image.open(img_path)  #或者用这种读取方法，应该只有速度上的区别
                plt.imshow(img)
                img = img.resize([32, 32])
                img = np.array(img)


                with tf.Graph().as_default():
                    BATCH_SIZE = 1
                    N_CLASSES = 2

                    image = tf.cast(img, tf.float32)
                    image = tf.image.per_image_standardization(image)
                    image = tf.reshape(image, [1, 32, 32, 3])
                    logit = fa_detect.model.inference(image, BATCH_SIZE, N_CLASSES)

                    logit = tf.nn.softmax(logit)

                    x = tf.placeholder(tf.float32, shape=[32, 32, 3])

                    saver = tf.train.Saver()

                    with tf.Session() as sess:

                        print("Reading checkpoints...")
                        ckpt = tf.train.get_checkpoint_state(train_logs_dir)
                        if ckpt and ckpt.model_checkpoint_path:
                            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                            saver.restore(sess, ckpt.model_checkpoint_path)
                            print('Loading success, global_step is %s' % global_step)
                        else:
                            print('No checkpoint file found')

                        prediction = sess.run(logit, feed_dict={x: img})
                        max_index = np.argmax(prediction)
                        if max_index==0:
                            print(str(index)+".jpg."+'This is a close eye with possibility %.6f' %prediction[:, 0])
                            perclos.append(0)
                            perclos.pop(0)
                        else:
                            print(str(index)+".jpg."+'This is a open eye with possibility %.6f' %prediction[:, 1])
                            perclos.append(1)
                            perclos.pop(0)
                        num = perclos.count(0)
                        print(perclos)
                        print("疲劳指数："+str(num/120))
                        if num >= 80:
                            print("疲劳")
                        else:
                            print("清醒")
                active = False
            else:
                continue
        index += 1

if __name__ == '__main__':
    test_images()
#这里检测perclos中0的个数，超过k个就报警
#在main函数里可以增加一个显示图像，出现疲劳就显示红色warning，并存在一定时间
