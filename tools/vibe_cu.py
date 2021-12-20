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


def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    vc = cv2.VideoCapture("/dataset/mz/outside_data/vibe_data/seg_feat_out.avi")

    fw = vc.get(cv2.CAP_PROP_FRAME_WIDTH)
    fh = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = vc.get(cv2.CAP_PROP_FPS)
    f_num = vc.get(cv2.CAP_PROP_FRAME_COUNT)
    print(f'frame size: (w: {fw}, h: {fh})')
    print(f'frame rate: {fps}')
    print(f'frame count: {f_num}')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_video = cv2.VideoWriter(
        '/dataset/mz/outside_data/vibe_data/vibe_seg_out_test.avi',
        fourcc, fps, (1920, 1080))

    c = 0
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = torch.from_numpy(frame).to(device)
    # vibe = ViBe(num_sam=30,
    #             min_match=2,
    #             radiu=10,
    #             rand_sam=16)
    vibe = ViBe(device=device)
    vibe.ProcessFirstFrame(frame)
    print(f'video shape: {frame.shape}')
    # cv2.namedWindow("frame", 0)
    # cv2.resizeWindow("frame", frame.shape[1], frame.shape[0])
    # cv2.namedWindow("segMat", 0)
    # cv2.resizeWindow("segMat", frame.shape[1], frame.shape[0])

    while rval:
        rval, frame = vc.read()
        if not rval:
            break

        # 将输入转为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = torch.from_numpy(gray).to(device)
        # 输出二值图
        #(segMat, samples) = update(gray, samples)
        start_time = time.time()
        vibe.Update(gray)
        segMat = vibe.getFGMask()
        #　转为uint8类型
        segMat = segMat.cpu().numpy().astype(np.uint8)
        print(f'FPS: {1 / (time.time() - start_time)}')
        # 形态学处理模板初始化
        #kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        # 开运算
        #opening = cv2.morphologyEx(segMat, cv2.MORPH_OPEN, kernel1)
        # 形态学处理模板初始化
        #kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        # 闭运算
        #closed = cv2.morphologyEx(segMat, cv2.MORPH_CLOSE, kernel2)

        # 寻找轮廓
        #contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #for i in range(0, len(contours)):
        #        x, y, w, h = cv2.boundingRect(contours[i])
        #        print(w * h)
        #        if w * h > 400 and w * h < 10000:
        #            cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 255, 0), 2)

        # cv2.imshow("frame", frame)
        # cv2.imshow("segMat", segMat)
        out_video.write(cv2.cvtColor(segMat, cv2.COLOR_GRAY2BGR))
        #cv2.imwrite("./result/" + str(c) + ".jpg", frame,[int(cv2.IMWRITE_PNG_STRATEGY)])
        if 0xFF & cv2.waitKey(10) == 27:
            vc.release()
            cv2.destroyAllWindows()
            break

        c = c + 1


if __name__ == '__main__':
    main()
