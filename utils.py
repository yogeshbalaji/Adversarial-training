import torch


def _is_tensor_image(img):
    return torch.is_tensor(img)


def data_normalize(tensor, mean, std, inplace=False):
    """Normalize a tensor image with mean and standard deviation.
    .. note::
       This transform acts out of place by default, i.e., it does not mutates the input tensor.
    See :class:`~torchvision.transforms.Normalize` for more details.
    Args:
       tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
       mean (sequence): Sequence of means for each channel.
       std (sequence): Sequence of standard deviations for each channel.
       inplace(bool,optional): Bool to make this operation inplace.
    Returns:
       Tensor: Normalized Tensor image.
    """

    if not _is_tensor_image(tensor):
       raise TypeError('tensor is not a torch image.')

    if not inplace:
       tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return tensor


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ConfigMapper(object):
    def __init__(self, args):
        for key in args:
            self.__dict__[key] = args[key]
