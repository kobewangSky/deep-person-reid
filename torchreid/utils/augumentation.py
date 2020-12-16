import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
from math import sqrt
from PIL import Image
from torchvision.transforms import *
import imgaug.augmenters as iaa


class SSD_Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img, = t(img)
        return img


class SSD_Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)


class SSD_ConvertFromInts(object):
    def __call__(self, image):
        return image.astype(np.float32)






class SSD_Resize(object):
    """ If preserve_aspect_ratio is true, this resizes to an approximate area of max_size * max_size """


    def __init__(self, resize_gt):
        self.resize_gt = resize_gt

    def __call__(self, image):

        image = cv2.resize(image, (self.resize_gt[1], self.resize_gt[0]))

        return image


class SSD_RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image


class SSD_RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image


class SSD_RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image):
        # Don't shuffle the channels please, why would you do this

        # if random.randint(2):
        #     swap = self.perms[random.randint(len(self.perms))]
        #     shuffle = SwapChannels(swap)  # shuffle channels
        #     image = shuffle(image)
        return image


class SSD_ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif self.current == 'HSV' and self.transform == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        else:
            raise NotImplementedError
        return image


class SSD_RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image


class SSD_RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image


class SSD_ToCV2Image(object):
    def __call__(self, tensor, masks=None, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), masks, boxes, labels


class SSD_ToTensor(object):
    def __call__(self, cvimage, masks=None, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), masks, boxes, labels


class SSD_RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """

    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left + w), int(top + h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                :]


                return current_image


class SSD_Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image):
        if random.randint(2):
            return image

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width * ratio - width)
        top = random.uniform(0, height * ratio - height)

        expand_image = np.zeros(
            (int(height * ratio), int(width * ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
        int(left):int(left + width)] = image
        image = expand_image


        return image


class SSD_RandomMirror(object):
    def __call__(self, image
                 ):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
        return image


class SSD_RandomFlip(object):
    def __call__(self, image):
        height, _, _ = image.shape
        if random.randint(2):
            image = image[::-1, :]
        return image


class SSD_RandomRot(object):
    def __call__(self, image):
        np.rota
        old_height, old_width, _ = image.shape
        #TODO
        k = random.randint(4)
        image = np.rot90(image, k)

        return image


class SSD_SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image



class SSD_PhotometricDistort(object):
    def __init__(self, istest = False):
        self.rand_brightness = SSD_RandomBrightness()
        self.rand_light_noise = SSD_RandomLightingNoise()
        self.istest = istest

    def __call__(self, image):
        im = image.copy()
        pix = np.array(im, dtype='float32')



        pix = iaa.Fliplr(0.5).augment_image(pix)

        aug_ratate = iaa.OneOf([
            iaa.PerspectiveTransform(scale=(0, 0.015)),
            iaa.Rotate((-10, 10))
        ])

        pix = aug_ratate.augment_image(pix)

        if self.istest:
            return pix
        pix = self.rand_brightness(pix)
        if random.randint(2):
            self.aumentone(pix)
            #distort = SSD_Compose(self.pd[:-1])
        else:
            self.aumenttwo(pix)
            #distort = SSD_Compose(self.pd[1:])
        #pix = distort(pix)

        #pix = pix.astype('uint8')

        pix = np.where(pix > 255, 255, pix)
        pix = np.where(pix < 0, 0, pix)

        #im = Image.fromarray(pix, 'RGB')
        #self.rand_light_noise(pix)

        # aug = iaa.Sequential(iaa.SomeOf(2, [
        #     iaa.AdditiveGaussianNoise(scale=(0, 0.1*255)),
        #     iaa.Dropout(p=(0, 0.1), per_channel=0.5),
        #     iaa.SaltAndPepper(p=(0, 0.1), per_channel=True),
        #     iaa.GaussianBlur(sigma=(0.0, 3.0)),
        #     iaa.MotionBlur(k=15, angle=[-45, 45]),
        #     iaa.AverageBlur(k=((0, 5), (0, 5)))
        # ], random_order=True))

        aug_noise = iaa.OneOf([
            iaa.AdditiveGaussianNoise(scale=(0, 0.1 * 255)),
            iaa.Dropout(p=(0, 0.1), per_channel=0.5),
            iaa.SaltAndPepper(p=(0, 0.1), per_channel=True),
        ])
        aug_blur = iaa.OneOf([
            iaa.GaussianBlur(sigma=(0.0, 1.5)),
            iaa.MotionBlur(k=7, angle=[-20, 20]),
            iaa.AverageBlur(k=((0, 2.5), (0, 2.5)))
        ])

        pix = aug_noise.augment_image(pix)
        pix = aug_blur.augment_image(pix)

        return pix / 255.0

    def aumentone(self, img):

        img = SSD_RandomContrast()(img),
        img = SSD_ConvertColor(transform='HSV')(img[0]),
        img = SSD_RandomSaturation()(img[0]),
        img = SSD_RandomHue()(img[0]),
        img = SSD_ConvertColor(current='HSV', transform='RGB')(img[0]),
        return img

    def aumenttwo(self, img):

        img = SSD_ConvertColor(transform='HSV')(img),
        img = SSD_RandomSaturation()(img[0]),
        img = SSD_RandomHue()(img[0]),
        img = SSD_ConvertColor(current='HSV', transform='RGB')(img[0]),
        img = SSD_RandomContrast()(img[0])
        return img



class SSD_BaseTransform(object):
    """ Transorm to be used when evaluating. """

    def __init__(self, mean, std):
        self.augment = SSD_Compose([
            SSD_ConvertFromInts(),
            SSD_Resize(resize_gt=False),
        ])

    def __call__(self, img, masks=None, boxes=None, labels=None):
        return self.augment(img, masks, boxes, labels)


import torch.nn.functional as F


class SSD_FastBaseTransform(torch.nn.Module):
    """
    Transform that does all operations on the GPU for super speed.
    This doesn't suppport a lot of config settings and should only be used for production.
    Maintain this as necessary.
    """

    def __init__(self, MEANS, STD):
        super().__init__()

        self.mean = torch.Tensor(MEANS).float().cuda()[None, :, None, None]
        self.std = torch.Tensor(STD).float().cuda()[None, :, None, None]

    def forward(self, img):
        self.mean = self.mean.to(img.device)
        self.std = self.std.to(img.device)


        _, h, w, _ = img.size()
        img_size = (h * 1.125, w * 1.125)


        img = img.permute(0, 3, 1, 2).contiguous()
        img = F.interpolate(img, img_size, mode='bilinear', align_corners=False)

        if self.transform.normalize:
            img = (img - self.mean) / self.std
        elif self.transform.subtract_means:
            img = (img - self.mean)
        elif self.transform.to_float:
            img = img / 255

        if self.transform.channel_order != 'RGB':
            raise NotImplementedError

        img = img[:, (2, 1, 0), :, :].contiguous()

        # Return value is in channel order [n, c, h, w] and RGB
        return img


def SSD_do_nothing(img=None, masks=None, boxes=None, labels=None):
    return img, masks, boxes, labels


def SSD_enable_if(condition, obj):
    return obj if condition else SSD_do_nothing


class SSDAugmentation(object):
    """ Transform to be used when training. """

    def __init__(self, mean, std):
        self.augment = SSD_Compose([
            SSD_ConvertFromInts(),
            SSD_PhotometricDistort(),
            SSD_Expand(mean),
            SSD_RandomSampleCrop(),
            SSD_RandomMirror(),
            SSD_RandomFlip(),
            SSD_RandomRot(),
            SSD_Resize(),
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)
