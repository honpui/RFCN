import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage.segmentation import slic
from skimage import color, img_as_ubyte


def imsave(file_name, img):
    """
    save a torch tensor as an image
    :param file_name: 'image/folder/image_name'
    :param img: 3*h*w torch tensor
    :return: nothing
    """
    assert (type(img) == torch.FloatTensor,
            'img must be a torch.FloatTensor')
    ndim = len(img.size())
    assert (ndim == 2 or ndim == 3,
            'img must be a 2 or 3 dimensional tensor')
    img = img.numpy()
    if ndim == 3:
        plt.imsave(file_name, np.transpose(img, (1, 2, 0)))
    else:
        plt.imsave(file_name, img, cmap='gray')


def tensor2image(image):
    """
    convert a mean-0 tensor to float numpy image
    :param image: 
    :return: image
    """
    image = image.clone()
    image[0] = image[0] + 122.67891434
    image[1] = image[1] + 116.66876762
    image[2] = image[2] + 104.00698793
    image = image.numpy() / 255.0
    image = image.transpose((1, 2, 0))
    image = img_as_ubyte(image)
    return image


# def prior_map(img):
#     """
#     get RFCN prior map
#     :param img: numpy array (H*W*C, RGB), [0, 1], float
#     :return: pmap
#     """
#     # step 1 over segmentation into superpixels
#     sp = slic(img, n_segments=200, sigma=5)
#     sp_num = sp.max() + 1
#     sp = sp.astype(float)
#
#     # step 2 the mean lab color of the sps
#     mean_lab_color = np.zeros((sp_num, 3))
#     lab_img = color.rgb2lab(img)
#     for c in range(3):
#         for i in range(sp_num):
#             mean_lab_color[i, c] = lab_img[sp == i, c].mean()
#
#     # step 3, element uniqueness




    return pimg

