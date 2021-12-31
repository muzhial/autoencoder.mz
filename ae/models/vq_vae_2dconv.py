import torch
import torch.nn as nn

from .vq_vae import VQEmbeddingEMA


class VQVaeCov2D(nn.Module):
    def __init__(self,
                 chnum_in,
                 mem_dim,
                 n_embeddings,
                 feature_num=128,
                 feature_num_2=96,
                 commitment_cost=0.25,
                 decay=0.999,
                 epsilon=1e-5):
        super(VQVaeCov2D, self).__init__()

        self.chnum_in = chnum_in
        self.mem_dim = mem_dim
        self.n_embeddings = n_embeddings
        self.feature_num = feature_num
        self.feature_num_2 = feature_num_2
        self.feature_num_x2 = feature_num * 2
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        self.mse_loss = nn.MSELoss()

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

        self.vq_embedding = VQEmbeddingEMA(self.n_embeddings,
                                           self.mem_dim,
                                           self.commitment_cost,
                                           self.decay,
                                           self.epsilon)

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

    def forward(self, x, return_loss=True):
        f = self.encoder(x)
        z_quantized, commitment_loss, codebook_loss, perplexity = self.vq_embedding(f)
        x_hat = self.decoder(z_quantized)
        if return_loss:
            recon_loss = self.mse_loss(x_hat, x)
            total_loss = recon_loss + commitment_loss + codebook_loss
            return {
                'total_loss': total_loss,
                'recon_loss': recon_loss,
                'commitment_loss': commitment_loss,
                'codebook_loss': codebook_loss,
                'perplexity_loss': perplexity
            }
        else:
            return {
                'recon_result': x_hat
            }
