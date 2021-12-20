import torch
import torch.nn as nn

from .memory_module import MemModule
from .losses import EntropyLossEncap, EntropyLoss


class AutoEncoderCov2DMem(nn.Module):
    def __init__(self,
                 chnum_in,
                 mem_dim,
                 feature_num=128,
                 feature_num_2=96,
                 shrink_thres=0.0025,
                 entropy_loss_weight=0.0002):
        super(AutoEncoderCov2DMem, self).__init__()

        self.chnum_in = chnum_in
        self.mem_dim = mem_dim
        self.feature_num = feature_num
        self.feature_num_2 = feature_num_2
        self.feature_num_x2 = feature_num * 2
        self.shrink_thres = shrink_thres
        self.entropy_loss_weight = entropy_loss_weight

        self.recon_loss = nn.MSELoss()
        self.memo_loss = EntropyLossEncap()

        self.encoder = nn.Sequential(
            nn.Conv2d(self.chnum_in,
                      self.feature_num_2,
                      kernel_size=3,
                      stride=2,
                      padding=1),  # (B, 96, 128, 128)
            nn.BatchNorm2d(self.feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.feature_num_2,
                      self.feature_num,
                      kernel_size=3,
                      stride=2,
                      padding=1),  # (B, 128, 64, 64)
            nn.BatchNorm2d(self.feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.feature_num,
                      self.feature_num_x2,
                      kernel_size=3,
                      stride=2,
                      padding=1),  # (B, 256, 32, 32)
            nn.BatchNorm2d(self.feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.feature_num_x2,
                      self.feature_num_x2,
                      kernel_size=3,
                      stride=2,
                      padding=1),  # (B, 256, 16, 16)
            nn.BatchNorm2d(self.feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.mem_rep = MemModule(mem_dim=self.mem_dim,
                                 fea_dim=self.feature_num_x2,
                                 shrink_thres=self.shrink_thres)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.feature_num_x2,
                               self.feature_num_x2,
                               3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(self.feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(self.feature_num_x2,
                               self.feature_num,
                               3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(self.feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(self.feature_num,
                               self.feature_num_2,
                               3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(self.feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(self.feature_num_2,
                               self.chnum_in,
                               3,
                               stride=2,
                               padding=1,
                               output_padding=1)
        )

    def get_loss(self, recon_res, att, target):
        recon_loss = self.recon_loss(recon_res, target)
        memo_loss = self.memo_loss(att)
        total_loss = recon_loss + self.entropy_loss_weight * memo_loss
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'memo_loss': memo_loss
        }

    def forward(self, x, return_loss=True):
        f = self.encoder(x)
        res_mem = self.mem_rep(f)
        f = res_mem['output']
        att = res_mem['att']
        recon_result = self.decoder(f)
        if return_loss:
            return self.get_loss(recon_result, att, x)
        else:
            return {
                'recon_result': recon_result
            }
