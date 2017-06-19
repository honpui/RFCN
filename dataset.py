#!/usr/bin/env python

import collections
import os

import numpy as np
import PIL.Image
import scipy.io
import torch
from torch.utils import data


class MyData(data.Dataset):
    """
    load data in a folder
    """
    mean_rgb = np.array([122.67891434, 116.66876762, 104.00698793])

    def __init__(self, root, transform=False):
        super(MyData, self).__init__()
        self.root = root
        self._transform = transform

        img_root = os.path.join(self.root, 'images')
        lbl_root = os.path.join(self.root, 'masks')
        file_names = os.listdir(img_root)
        self.img_names = []
        self.lbl_names = []
        for i, name in enumerate(file_names):
            if not name.endswith('.jpg'):
                continue
            self.lbl_names.append(
                os.path.join(lbl_root, name[:-4]+'.png')
            )
            self.img_names.append(
                os.path.join(img_root, name)
            )

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        # load image
        img_file = self.img_names[index]
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        # load label
        lbl_file = self.lbl_names[index]
        lbl = PIL.Image.open(lbl_file)
        lbl = np.array(lbl, dtype=np.int32)
        lbl[lbl != 0] = 1
        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl

    def transform(self, img, lbl):
        img = img.astype(np.float64)
        img -= self.mean_rgb
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).float()
        return img, lbl

    def untransform(self, img):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_rgb
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        return img


class MyTestData(data.Dataset):
    """
    load data in a folder
    """
    mean_rgb = np.array([122.67891434, 116.66876762, 104.00698793])

    def __init__(self, root, transform=False):
        super(MyTestData, self).__init__()
        self.root = root
        self._transform = transform

        img_root = self.root
        file_names = os.listdir(img_root)
        self.img_names = []
        self.names = []
        for i, name in enumerate(file_names):
            if not name.endswith('.jpg'):
                continue
            self.img_names.append(
                os.path.join(img_root, name)
            )
            self.names.append(name[:-4])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        # load image
        img_file = self.img_names[index]
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        if self._transform:
            return self.transform(img), self.names[index]
        else:
            return img, self.names[index]

    def transform(self, img):
        img = img.astype(np.float64)
        img -= self.mean_rgb
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        return img

    def untransform(self, img):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_rgb
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        return img


class VOCClassSegBase(data.Dataset):

    class_names = np.array([
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'potted plant',
        'sheep',
        'sofa',
        'train',
        'tv/monitor',
    ])
    mean_rgb = np.array([122.67891434, 116.66876762, 104.00698793])

    def __init__(self, root, year, split='train', transform=False):
        self.root = root
        self.split = split
        self._transform = transform

        dataset_dir = os.path.join(self.root, 'VOC/VOCdevkit/VOC%d' % year)
        self.files = collections.defaultdict(list)
        for split in ['train', 'val']:
            imgsets_file = os.path.join(
                dataset_dir, 'ImageSets/Segmentation/%s.txt' % split)
            for did in open(imgsets_file):
                did = did.strip()
                img_file = os.path.join(dataset_dir, 'JPEGImages/%s.jpg' % did)
                lbl_file = os.path.join(
                    dataset_dir, 'SegmentationClass/%s.png' % did)
                self.files[split].append({
                    'img': img_file,
                    'lbl': lbl_file,
                })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        # load label
        lbl_file = data_file['lbl']
        lbl = PIL.Image.open(lbl_file)
        lbl = np.array(lbl, dtype=np.int32)
        lbl[lbl == 255] = -1
        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl

    def transform(self, img, lbl):
        img = img.astype(np.float64)
        img -= self.mean_rgb
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def untransform(self, img):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_rgb
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        return img


class VOC2011ClassSeg(VOCClassSegBase):

    url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar'  # NOQA

    def __init__(self, root, split='train', transform=False):
        super(VOC2011ClassSeg, self).__init__(
            root, year=2011, split=split, transform=transform)
        pkg_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        imgsets_file = os.path.join(
            pkg_root, 'ext/fcn.berkeleyvision.org',
            'data/pascal/seg11valid.txt')
        dataset_dir = os.path.join(self.root, 'VOC/VOCdevkit/VOC2011')
        for did in open(imgsets_file):
            did = did.strip()
            img_file = os.path.join(dataset_dir, 'JPEGImages/%s.jpg' % did)
            lbl_file = os.path.join(dataset_dir, 'SegmentationClass/%s.png' % did)
            self.files['seg11valid'].append({'img': img_file, 'lbl': lbl_file})


class VOC2012ClassSeg(VOCClassSegBase):

    url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'  # NOQA

    def __init__(self, root, split='train', transform=False):
        super(VOC2012ClassSeg, self).__init__(
            root, year=2012, split=split, transform=transform)
        pkg_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        imgsets_file = os.path.join(
            pkg_root, 'ext/fcn.berkeleyvision.org',
            'data/pascal/seg11valid.txt')
        dataset_dir = os.path.join(self.root, 'VOC/VOCdevkit/VOC2012')
        for did in open(imgsets_file):
            did = did.strip()
            img_file = os.path.join(dataset_dir, 'JPEGImages/%s.jpg' % did)
            lbl_file = os.path.join(dataset_dir, 'SegmentationClass/%s.png' % did)
            self.files['seg12valid'].append({'img': img_file, 'lbl': lbl_file})


class SBDClassSeg(VOCClassSegBase):

    # XXX: It must be renamed to benchmark.tar to be extracted.
    url = 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz'  # NOQA

    def __init__(self, root, split='train', transform=False):
        self.root = root
        self.split = split
        self._transform = transform

        dataset_dir = os.path.join(self.root, 'VOC/VOCdevkit/SBDD/dataset')
        self.files = collections.defaultdict(list)
        for split in ['train', 'seg11valid']:
            imgsets_file = os.path.join(dataset_dir, '%s.txt' % split)
            for did in open(imgsets_file):
                did = did.strip()
                img_file = os.path.join(dataset_dir, 'img/%s.jpg' % did)
                lbl_file = os.path.join(dataset_dir, 'cls/%s.mat' % did)
                self.files[split].append({
                    'img': img_file,
                    'lbl': lbl_file,
                })

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        # load label
        lbl_file = data_file['lbl']
        mat = scipy.io.loadmat(lbl_file)
        lbl = mat['GTcls'][0]['Segmentation'][0].astype(np.int32)
        lbl[lbl == 255] = -1
        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl
