from torchvision import transforms


class TrainTransform:

    def __init__(self, in_channel):
        assert isinstance(in_channel, int), 'in_channel must be `int` type'
        self.norm_mean, self.norm_std = [0.5] * in_channel, [0.5] * in_channel
        self.transforms = transforms.Compose(
            transforms.ToTensor(),
            transforms.Normalize(self.norm_mean, self.norm_std))

    def __call__(self, image):
        return self.transforms(image)
