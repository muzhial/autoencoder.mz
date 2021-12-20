import mmcv
from torchvision import transforms


class TrainTransform:

    def __init__(self, in_channel, resize):
        assert isinstance(in_channel, int), 'in_channel must be `int` type'
        self.norm_mean, self.norm_std = (0.5) * in_channel, (0.5) * in_channel
        self.resize = resize

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.norm_mean, self.norm_std)
        ])

    def __call__(self, image):
        image = mmcv.imresize(
            image, self.resize)
        image = self.transforms(image)
        return image


class UnNormalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, inp):
        # NCFHW or CFHW
        t_out = inp.clone()
        s = t_out.shape
        if(len(s)==5):
            channel_num = s[1]
            # TODO?: make efficient
            for i in range(channel_num):
                t_out[:, i, :, :, :] = t_out[:, i, :, :, :]*self.std[i] + self.mean[i]
        elif(len(s)==4):
            channel_num = s[0]
            for i in range(channel_num):
                t_out[i, :, :, :] = t_out[i, :, :, :]*self.std[i] + self.mean[i]
        return t_out
