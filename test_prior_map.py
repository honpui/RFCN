import cv2
import matplotlib.pyplot as plt
import MR


imgRoot = '/media/xyz/Files/data/datasets/VOC/VOCdevkit/SBDD/dataset/img/2008_000033.jpg'
img = cv2.imread(imgRoot)
mr_sal = MR.MR_saliency()
sal = mr_sal.saliency(img).astype(float) / 255.0
sal = 1-sal
plt.imshow(sal, cmap='gray')
plt.show()