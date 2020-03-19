#清空所有文件夹
import os

def delete_all_file(dir):
    for i in os.listdir(dir):
        path_file = os.path.join(dir, i)
        os.remove(path_file)
dir1 = '/Users/yicc/Desktop/code/可运行版本/fa_detect/get_image/file_test/'
dir2 = '/Users/yicc/Desktop/code/可运行版本/fa_detect/get_image/file_detect/'
dir3 = '/Users/yicc/Desktop/code/可运行版本/fa_detect/get_image/faces/'
dir4 = '/Users/yicc/Desktop/code/可运行版本/fa_detect/get_image/cut_images/'

delete_all_file(dir1)
delete_all_file(dir2)
delete_all_file(dir3)
delete_all_file(dir4)