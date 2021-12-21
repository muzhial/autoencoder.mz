import time

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch


class ViBe:

    def __init__(self,
                 num_sam=20,
                 min_match=2,
                 radiu=20,
                 rand_sam=16,
                 dist_type='ed',
                 device='cpu'):
        self.defaultNbSamples = num_sam
        self.defaultReqMatches = min_match
        self.defaultRadius = radiu
        self.defaultSubsamplingFactor = rand_sam

        self.dist_type = dist_type
        self.device = device

        self.background = 0
        self.foreground = 255

    def __buildNeighborArray(self, img):
        assert isinstance(img, torch.Tensor), f'img type must be torch.Tensor'
        assert len(img.size()) == 3, \
            f'img size must be (channel, height, width)'

        channel, height, width = img.size()
        img = img.to(self.device)
        self.samples = torch.zeros(
            (self.defaultNbSamples, channel, height, width))

        ramoff_xy = torch.randint(
            -1, 2, size=(2, self.defaultNbSamples, height, width))

        xr_ = torch.tile(torch.arange(width), (height, 1))
        yr_ = torch.tile(torch.arange(height), (width, 1)).t()

        xyr_ = torch.zeros(
            (2, self.defaultNbSamples, height, width))
        for i in range(self.defaultNbSamples):
            xyr_[1, i] = xr_
            xyr_[0, i] = yr_

        xyr_ = xyr_ + ramoff_xy

        xyr_[xyr_ < 0] = 0
        tpr_ = xyr_[1, :, :, -1]
        tpr_[tpr_ >= width] = width - 1
        tpb_ = xyr_[0, :, -1, :]
        tpb_[tpb_ >= height] = height - 1
        xyr_[0, :, -1, :] = tpb_
        xyr_[1, :, :, -1] = tpr_

        xyr = xyr_.long()
        # self.samples = img[xyr[0, :, :, :], xyr[1, :, :, :]]
        self.samples = img[:, xyr[0, :, :, :], xyr[1, :, :, :]
            ].permute(1, 0, 2, 3).contiguous()

    def ProcessFirstFrame(self, img):
        self.__buildNeighborArray(img)
        self.fgCount = torch.zeros(*img.shape[1:])
        self.fgMask = torch.zeros(*img.shape[1:])

    def Update(self, img):
        channel, height, width = img.size()
        img = img.to(self.device)
        # hist_feature(img, f'out/feat_hist.png')
        # heatmap_feature(img, f'out/heatmap_feature.png')
        if self.dist_type == 'cosine':
            img_tile = torch.tile(img, [self.samples.shape(0), 1, 1, 1])
            dist = 1 - (self.samples * img_tile).sum(dim=1) / (
                torch.norm(
                    self.samples, 2, dim=1
                ) * torch.norm(img_tile, 2, dim=1))
        elif self.dist_type == 'ed':
            dist = torch.sqrt(((self.samples - img) ** 2).sum(dim=1))
        elif self.dist_type == 'l1':
            dist = (self.samples.float() - img.float()
                ).abs().mean(dim=1)

        # hist_feature(dist, f'out/dist_hist.png')
        mask_bg = dist < self.defaultRadius
        mask_fg = dist >= self.defaultRadius
        dist[mask_bg] = 1
        dist[mask_fg] = 0

        matches = torch.sum(dist, dim=0)
        matches = matches < self.defaultReqMatches
        self.fgMask[matches] = self.foreground
        self.fgMask[~matches] = self.background
        self.fgCount[matches] = self.fgCount[matches] + 1
        self.fgCount[~matches] = 0
        # fakeFG = self.fgCount > 50
        # matches[fakeFG] = False
        upfactor = torch.randint(
            self.defaultSubsamplingFactor,
            size=img.shape[1:])
        upfactor[matches] = 100
        upSelfSamplesInd = torch.where(upfactor == 0)
        upSelfSamplesPosition = torch.randint(
            self.defaultNbSamples,
            size=upSelfSamplesInd[0].shape)
        samInd = (upSelfSamplesPosition, upSelfSamplesInd[0], upSelfSamplesInd[1])
        # self.samples[samInd] = img[upSelfSamplesInd]
        self.samples[samInd[0], :, samInd[1], samInd[2]] = \
            img[:, upSelfSamplesInd[0], upSelfSamplesInd[1]].T

        upfactor = torch.randint(
            self.defaultSubsamplingFactor,
            size=img.shape[1:])
        upfactor[matches] = 100
        upNbSamplesInd = torch.where(upfactor == 0)
        nbnums = upNbSamplesInd[0].shape[0]
        ramNbOffset = torch.randint(-1, 2, size=(2, nbnums))
        nbXY = torch.stack(upNbSamplesInd)
        nbXY += ramNbOffset
        nbXY[nbXY < 0] = 0
        nbXY[0, nbXY[0, :] >= height] = height - 1
        nbXY[1, nbXY[1, :] >= width] = width - 1
        nbSPos = torch.randint(self.defaultNbSamples, size=(nbnums, ))
        nbSamInd = (nbSPos, nbXY[0], nbXY[1])
        # self.samples[nbSamInd] = img[upNbSamplesInd]
        self.samples[nbSamInd[0], :, nbSamInd[1], nbSamInd[2]] = \
            img[:, upNbSamplesInd[0], upNbSamplesInd[1]].T

    def getFGMask(self):
        return self.fgMask
