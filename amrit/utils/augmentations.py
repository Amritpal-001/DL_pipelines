import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2

from torchvision import datasets, transforms

def get_augmentations(p=0.5, image_size=224):
    imagenet_stats = {"mean": [0.485, 0.456, 0.406],
                      "std": [0.229, 0.224, 0.225]}
    train_tfms = transforms.Compose([transforms.Resize((image_size,image_size)),
                                 transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

    train_tfms111 = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.ShiftScaleRotate(
                shift_limit=0.15, scale_limit=0.4, rotate_limit=45, p=p),
            A.Cutout(p=p),
            A.RandomRotate90(p=p),
            A.Flip(p=p),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2,
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50
                    ),
                ],
                p=p,
            ),
            A.CoarseDropout(max_holes=10, p=p),
            ToTensorV2(),
        ]
    )

    valid_tfms = transforms.Compose([transforms.Resize((image_size,image_size)),
                                 transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

    valid_tfms111 = A.Compose(
        [A.Resize(image_size, image_size),
         ToTensorV2()]
    )

    return train_tfms, valid_tfms


def get_tta(image_size=224):
    imagenet_stats = {"mean": [0.485, 0.456, 0.406],
                      "std": [0.229, 0.224, 0.225]}
    tta_tfms = A.Compose(
        [
            A.RandomResizedCrop(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
            ),
            ToTensorV2(),
        ],
        p=1.0,
    )
    return tta_tfms
