"""
RFCN
"""
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
import torch.nn.functional as functional

from dataset import SBDClassSeg, MyTestData
from transform import Colorize
from criterion import CrossEntropyLoss2d
from model import RFCN, FCN8s
from myfunc import imsave, tensor2image
import MR

import visdom
import numpy as np
import argparse
import os
import gc

parser = argparse.ArgumentParser()
parser.add_argument('--phase', type=str, default='train', help='train or test')
parser.add_argument('--param', type=str, default=None, help='path to pre-trained parameters')
parser.add_argument('--data', type=str, default='./train', help='path to input data')
parser.add_argument('--out', type=str, default='./out', help='path to output data')
opt = parser.parse_args()

opt.phase = 'train'
opt.data = '/media/xyz/Files/data/datasets'
opt.out = '/media/xyz/Files/data/models/torch/RFCN_pretrain'
opt.param = '/media/xyz/Files/data/models/torch/RFCN_pretrain/RFCN-epoch-4-step-11354.pth'

print(opt)

vis = visdom.Visdom()
win0 = vis.image(torch.zeros(3, 100, 100))
win1 = vis.image(torch.zeros(3, 100, 100))
win2 = vis.image(torch.zeros(3, 100, 100))
win22 = vis.image(torch.zeros(3, 100, 100))
win3 = vis.image(torch.zeros(3, 100, 100))
color_transform = Colorize()
"""parameters"""
iterNum = 30

"""data loader"""
# dataRoot = '/media/xyz/Files/data/datasets'
# checkRoot = '/media/xyz/Files/fcn8s-deconv'
dataRoot = opt.data
if not os.path.exists(opt.out):
    os.mkdir(opt.out)
if opt.phase == 'train':
    checkRoot = opt.out
    loader = torch.utils.data.DataLoader(
        SBDClassSeg(dataRoot, split='train', transform=True),
        batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
else:
    outputRoot = opt.out
    loader = torch.utils.data.DataLoader(
        MyTestData(dataRoot, transform=True),
        batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

"""nets"""
model = RFCN()
if opt.param is None:
    vgg16 = torchvision.models.vgg16(pretrained=True)
    model.copy_params_from_vgg16(vgg16, copy_fc8=False, init_upscore=True)
else:
    model.load_state_dict(torch.load(opt.param))

criterion = CrossEntropyLoss2d()
optimizer = torch.optim.Adam(model.parameters(), 0.0001, betas=(0.5, 0.999))

model = model.cuda()

mr_sal = MR.MR_saliency()
if opt.phase == 'train':
    """train"""
    for it in range(iterNum):
        epoch_loss = []
        for ib, data in enumerate(loader):
            # prior map
            _img = tensor2image(data[0][0])
            pmap = mr_sal.saliency(_img).astype(float) / 255.0
            pmap = 1.0 - pmap
            pmap = torch.unsqueeze(torch.FloatTensor(pmap), 0)
            pmap = torch.unsqueeze(pmap, 0)
            pmap = Variable(pmap).cuda()
            img = Variable(data[0]).cuda()

            # segmentation gt and bg&fg gt
            targets_S = Variable(data[1]).cuda()
            targets_G = torch.LongTensor(1, targets_S.size()[-2], targets_S.size()[-1]).fill_(0)
            targets_G[0][data[1] == 0] == 1
            targets_G = Variable(targets_G).cuda()

            model.zero_grad()
            loss = 0
            for ir in range(3):
                outputs = model(torch.cat((img, pmap.detach()), 1))  # detach or not?
                loss_S = criterion(outputs[:, :21, :, :], targets_S)
                loss_G = criterion(outputs[:, -2:, :, :], targets_G)
                _loss = loss_G + loss_S
                _loss.backward()
                loss += _loss.data[0]

                # update prior map
                del pmap
                gc.collect()
                pmap = functional.sigmoid(outputs[:, -1, :, :])
                pmap = torch.unsqueeze(pmap, 0)

                # visulize
                image = img[0].data.cpu()
                image[0] = image[0] + 122.67891434
                image[1] = image[1] + 116.66876762
                image[2] = image[2] + 104.00698793
                title = 'input (epoch: %d, step: %d, recurrent: %d)' % (it, ib, ir)
                vis.image(image, win=win1, env='fcn', opts=dict(title=title))
                title = 'output_c (epoch: %d, step: %d, recurrent: %d)' % (it, ib, ir)
                vis.image(color_transform(outputs[0, :21].cpu().max(0)[1].data),
                          win=win2, env='fcn', opts=dict(title=title))
                title = 'output_l (epoch: %d, step: %d, recurrent: %d)' % (it, ib, ir)
                bb = functional.sigmoid(outputs[0, -1:].cpu().data)
                vis.image(bb.repeat(3, 1, 1),
                          win=win22, env='fcn', opts=dict(title=title))
                title = 'target (epoch: %d, step: %d, recurrent: %d)' % (it, ib, ir)
                vis.image(color_transform(targets_S.cpu().data),
                          win=win3, env='fcn', opts=dict(title=title))

                del outputs
                gc.collect()

            # update the net
            optimizer.step()

            # show loss plot in this batch
            epoch_loss.append(loss)
            average = sum(epoch_loss) / len(epoch_loss)
            print('loss: %.4f (epoch: %d, step: %d)' % (loss, it, ib))
            epoch_loss.append(average)
            x = np.arange(1, len(epoch_loss) + 1, 1)
            title = 'loss'
            vis.line(np.array(epoch_loss), x, env='fcn', win=win0,
                     opts=dict(title=title))

            del img, targets_S, targets_G
            gc.collect()

        # save parameters in each iteration
        filename = ('%s/RFCN-epoch-%d-step-%d.pth' \
                    % (checkRoot, it, ib))
        torch.save(model.state_dict(), filename)
        print('save: (epoch: %d, step: %d)' % (it, ib))
else:
    for ib, data in enumerate(loader):
        print('testing batch %d' % ib)
        inputs = Variable(data[0]).cuda()
        outputs = model(inputs)
        hhh = color_transform(outputs[0].cpu().max(0)[1].data)
        imsave(os.path.join(outputRoot, data[1][0] + '.png'), hhh)
