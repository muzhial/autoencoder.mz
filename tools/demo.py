import argparse
import os
import time

from tqdm import tqdm
from loguru import logger
import cv2
import numpy as np
import torch

from ae.exp import get_exp
from ae.utils import get_model_info
from ae.data import TrainTransform, UnNormalize
from vibe_cu import ViBe


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('demo', default='image', help='demo type')
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument(
        "-n", "--name", type=str, default=None, help="model name")
    parser.add_argument(
        "--path", default="./assets/dog.jpg", help="path to images or video"
    )
    parser.add_argument(
        "--camid", type=int, default=0, help="webcam demo camera id")
    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your experiment description file",
    )
    parser.add_argument(
        "-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument(
        "--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )

    return parser


class Predictor(object):

    def __init__(
        self,
        model,
        exp,
        device='cpu',
        fp16=False):

        self.model = model
        self.exp = exp
        self.device = device
        self.fp16 = fp16

        self.train_transform = TrainTransform(
            self.exp.chnum_in,
            self.exp.test_size)

    @staticmethod
    def post_process(chnum_in, outputs):
        norm_mean, norm_std = ((0.5,) * chnum_in,
                               (0.5,) * chnum_in)
        unnorm_transform = UnNormalize(
            norm_mean, norm_std)

        unorm_recon_out = unnorm_transform(outputs['recon_result'])
        unorm_recon_out = unorm_recon_out.squeeze(0)
        recon_out = unorm_recon_out.cpu().numpy()
        recon_out = np.transpose(recon_out, (1, 2, 0))
        recon_out = (recon_out * 255).astype(np.uint8)
        # recon_frame = cv2.cvtColor(recon_out, cv2.COLOR_GRAY2BGR)
        return recon_out

    def inference(self, image):
        image = self.train_transform(image)
        img = image.unsqueeze(0)
        if self.device == 'gpu':
            img = img.cuda()
            if self.fp16:
                img = img.half()
        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img, return_loss=False)

        img = self.post_process(self.exp.chnum_in, outputs)
        # logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return img


def image_demo(predictor, image, save_dir):
    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    result = predictor.inference(img)
    cv2.imwrite(save_dir, result)


def vibe_demo(predictor, args, save_dir):
    device = 'cuda:0' if args.device == 'gpu' else 'cpu'
    vc = cv2.VideoCapture(args.path)
    fw = vc.get(cv2.CAP_PROP_FRAME_WIDTH)
    fh = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = vc.get(cv2.CAP_PROP_FPS)
    f_num = vc.get(cv2.CAP_PROP_FRAME_COUNT)
    logger.info(f'read video: {args.path}')
    logger.info(f'frame size: {fw}, {fh}')
    logger.info(f'frame rate: {fps}')
    logger.info(f'frame count: {f_num}')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_video = cv2.VideoWriter(
        save_dir, fourcc, fps, (int(fw), int(fh)))
    recon_video = cv2.VideoWriter(
        os.path.join(os.path.dirname(save_dir),
                     'recon_' + os.path.basename(save_dir)),
        fourcc, fps, (int(fw), int(fh)))

    assert vc.isOpened(), 'video not opened'
    init_process = False
    frame_idx = 0
    vibe = ViBe(
        num_sam=20,
        min_match=7,
        radiu=15,
        rand_sam=16,
        device=device)

    pbar = tqdm(total=int(f_num))
    while True:
        ret_val, frame = vc.read()
        if ret_val:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_result = predictor.inference(frame)
            recon_error = np.abs(frame[..., None] - frame_result)

            # save intermidate frame result
            recon_frame = cv2.cvtColor(recon_error, cv2.COLOR_GRAY2BGR)
            # cv2.imwrite(os.path.join(
            #                 os.path.dirname(save_dir), f'{frame_idx:04d}.png'),
            #             recon_frame)
            recon_video.write(recon_frame)

            frame_result = np.transpose(recon_error, (2, 0, 1))
            frame_result = torch.from_numpy(frame_result).to(device)
            if not init_process:
                vibe.ProcessFirstFrame(frame_result)
                init_process = True
            else:
                vibe.Update(frame_result)
                seg_mat = vibe.getFGMask()
                seg_mat = seg_mat.cpu().numpy().astype(np.uint8)
                out_video.write(cv2.cvtColor(seg_mat, cv2.COLOR_GRAY2BGR))
        else:
            break
        frame_idx += 1
        pbar.update(1)
    pbar.close()


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = os.path.join(file_name, "vis_res")
    os.makedirs(vis_folder, exist_ok=True)

    logger.info("Args: {}".format(args))

    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info(
        "Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()
    if args.ckpt is None:
        ckpt_file = os.path.join(file_name, "300e_ckpt.pth")
    else:
        ckpt_file = args.ckpt
    logger.info("loading checkpoint")
    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict
    model.load_state_dict({
        k.replace('module.', ''): v
        for k,v in ckpt["model"].items()
    })
    logger.info("loaded checkpoint done.")

    predictor = Predictor(model, exp, args.device)
    if args.demo == 'image':
        image_demo(predictor,
                   args.path,
                   os.path.join(
                       vis_folder, os.path.basename(args.path)))
    elif args.demo == 'video':
        pass
    elif args.demo == 'vibe':
        vibe_demo(predictor,
                  args,
                  os.path.join(vis_folder, os.path.basename(args.path)))


if __name__ == '__main__':
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
