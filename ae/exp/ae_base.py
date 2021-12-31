import os
import random

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import optimizer
from torch.utils.data.dataloader import DataLoader

from ae.models import vq_vae_2dconv

from .base_exp import BaseExp


class Exp(BaseExp):
    def __init__(self):
        super().__init__()

        # ---------------- model config ---------------- #
        self.chnum_in = 1
        self.mem_dim = 2000
        self.feature_num = 128
        self.feature_num_2 = 96
        self.shrink_thres = 0.0025
        self.mem_type = 'mem'  # ['mem' | 'vq']

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        self.data_num_workers = 4
        self.input_size = (1280, 720)  # (w, h)
        # You can uncomment this line to specify a multiscale range
        # self.random_size = (14, 26)
        self.data_dir = '/dataset/mz/outside_data/fault_vid/imagedata/normal'
        # self.train_ann = "instances_train2017.json"
        # self.val_ann = "instances_val2017.json"

        # --------------  training config --------------------- #
        self.warmup_epochs = -1
        self.max_epoch = 100
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "warmcos"
        self.no_aug_epochs = 0
        self.min_lr_ratio = 0.05
        self.ema = False

        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 10
        self.eval_interval = 10
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # -----------------  testing config ------------------ #
        self.test_size = (1280, 720)

    def get_model(self):
        from ae.models import AutoEncoderCov2DMem, VQVaeCov2D
        from ae.utils import weights_init

        # def init_yolo(M):
        #     for m in M.modules():
        #         if isinstance(m, nn.BatchNorm2d):
        #             m.eps = 1e-3
        #             m.momentum = 0.03

        if getattr(self, "model", None) is None:
            if self.mem_type == 'mem':
                self.model = AutoEncoderCov2DMem(self.chnum_in,
                                                 self.mem_dim,
                                                 self.feature_num,
                                                 self.feature_num_2,
                                                 self.shrink_thres)
            elif self.mem_type == 'vq':
                self.model = VQVaeCov2D(self.chnum_in,
                                        self.mem_dim,
                                        self.n_embeddings,
                                        self.feature_num,
                                        self.feature_num_2,
                                        self.commitment_cost,
                                        self.decay,
                                        self.epsilon)

        self.model.apply(weights_init)
        return self.model

    def get_data_loader(
        self, batch_size, is_distributed,
        shuffle=True, round_up=True, cache_img=False
    ):
        from ae.data import (
            ImageDataset,
            TrainTransform,
            DistributedSampler,
            worker_init_reset_seed,
        )
        from ae.utils import (
            wait_for_the_master,
            get_local_rank,
            get_rank
        )

        local_rank = get_local_rank()
        rank = get_rank()

        with wait_for_the_master(local_rank):
            dataset = ImageDataset(
                data_dir=self.data_dir,
                preproc=TrainTransform(self.chnum_in, self.input_size))

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        if is_distributed:
            sampler = DistributedSampler(
                dataset,
                dist.get_world_size(),
                rank,
                shuffle=shuffle,
                round_up=round_up)
            shuffle = False
        else:
            sampler = None

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True
        }
        dataloader_kwargs["sampler"] = sampler
        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed
        dataloader_kwargs["shuffle"] = shuffle
        dataloader_kwargs["batch_size"] = batch_size

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    # def random_resize(self, data_loader, epoch, rank, is_distributed):
    #     tensor = torch.LongTensor(2).cuda()

    #     if rank == 0:
    #         size_factor = self.input_size[1] * 1.0 / self.input_size[0]
    #         if not hasattr(self, 'random_size'):
    #             min_size = int(self.input_size[0] / 32) - self.multiscale_range
    #             max_size = int(self.input_size[0] / 32) + self.multiscale_range
    #             self.random_size = (min_size, max_size)
    #         size = random.randint(*self.random_size)
    #         size = (int(32 * size), 32 * int(size * size_factor))
    #         tensor[0] = size[0]
    #         tensor[1] = size[1]

    #     if is_distributed:
    #         dist.barrier()
    #         dist.broadcast(tensor, 0)

    #     input_size = (tensor[0].item(), tensor[1].item())
    #     return input_size

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            # pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

            # for k, v in self.model.named_modules():
            #     if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            #         pg2.append(v.bias)  # biases
            #     if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            #         pg0.append(v.weight)  # no decay
            #     elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            #         pg1.append(v.weight)  # apply decay

            # optimizer = torch.optim.SGD(
            #     pg0, lr=lr, momentum=self.momentum, nesterov=True
            # )
            # optimizer.add_param_group(
            #     {"params": pg1, "weight_decay": self.weight_decay}
            # )  # add pg1 with weight_decay
            # optimizer.add_param_group({"params": pg2})

            # lr = 1e-4
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

            self.optimizer = optimizer

        return self.optimizer

    def get_lr_scheduler(self, lr, iters_per_epoch):
        from ae.utils import LRScheduler

        scheduler = LRScheduler(
            self.scheduler,
            lr,
            iters_per_epoch,
            self.max_epoch,
            warmup_epochs=self.warmup_epochs,
            warmup_lr_start=self.warmup_lr,
            no_aug_epochs=self.no_aug_epochs,
            min_lr_ratio=self.min_lr_ratio,
        )
        return scheduler

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        # from yolox.data import COCODataset, ValTransform

        # valdataset = COCODataset(
        #     data_dir=self.data_dir,
        #     json_file=self.val_ann if not testdev else "image_info_test-dev2017.json",
        #     name="val2017" if not testdev else "test2017",
        #     img_size=self.test_size,
        #     preproc=ValTransform(legacy=legacy),
        # )

        # if is_distributed:
        #     batch_size = batch_size // dist.get_world_size()
        #     sampler = torch.utils.data.distributed.DistributedSampler(
        #         valdataset, shuffle=False
        #     )
        # else:
        #     sampler = torch.utils.data.SequentialSampler(valdataset)

        # dataloader_kwargs = {
        #     "num_workers": self.data_num_workers,
        #     "pin_memory": True,
        #     "sampler": sampler,
        # }
        # dataloader_kwargs["batch_size"] = batch_size
        # val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        # return val_loader
        pass

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        # from yolox.evaluators import COCOEvaluator

        # val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        # evaluator = COCOEvaluator(
        #     dataloader=val_loader,
        #     img_size=self.test_size,
        #     confthre=self.test_conf,
        #     nmsthre=self.nmsthre,
        #     num_classes=self.num_classes,
        #     testdev=testdev,
        # )
        # return evaluator
        pass

    def eval(self, model, evaluator, is_distributed, half=False):
        # return evaluator.evaluate(model, is_distributed, half)
        pass
