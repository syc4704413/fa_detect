import os
import numpy as np
import tensorflow as tf
import fa_detect.input_data
import fa_detect.model
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io
import time

import fa_detect.get_image.get_faces
import fa_detect.perclos_final
import threading
import fa_detect.get_image.get_feauture

#perclos_final.test_images()
#def main():
t1 = threading.Thread(target=fa_detect.get_image.get_faces.get_face)
t2 = threading.Thread(target=fa_detect.get_image.get_feauture.get_cut_images)
t3 = threading.Thread(target=fa_detect.perclos_final.test_images)

t3.start()
#time.sleep(60)

t1.start()
t2.start()



