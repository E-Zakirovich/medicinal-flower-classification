import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms, datasets
import config


class Data:
    def __init__(self):

        self.transform_dataset = transforms.Compose([

            # make the size of the images same for cnn & nn
            transforms.Resize(
                (config.image_size, config.image_size)
                ),

            # flip & rotate images in order to avoid underfitting & overfitting
            transforms.RandomHorizontalFlip(p = config.flip_amount),
            transforms.RandomVerticalFlip(p = config.flip_amount),
            transforms.RandomRotation(degrees = config.rotation_amount),

            # make a tensor for cnn
            transforms.ToTensor(),

            # normalization
            transforms.Normalize(
                mean = (0.485, 0.456, 0.406),
                std = (0.229, 0.224, 0.225)
                )
            ]
        )

        self.validation_and_test_transformation = transforms.Compose([

            # make the size of the images same for cnn & nn
            transforms.Resize(
                (config.image_size, config.image_size)
            ),

            # make a tensor for cnn
            transforms.ToTensor(),

            # normalization
            transforms.Normalize(
                mean = (0.485, 0.456, 0.406),
                std = (0.229, 0.224, 0.225)
            )
        ])


    def __load_dataset(self):

        # train dataset transformation
        train_transform = datasets.ImageFolder(
            root = config.path,
            transform = self.transform_dataset
        )

        # validation and test transformation
        validation_and_test_transform = datasets.ImageFolder(
            root = config.path,
            transform = self.validation_and_test_transformation
        )

        # generate controllable random numbers
        generator = torch.Generator().manual_seed(config.seed)

        train_indices, validation_indices,  test_indices = random_split(
            train_transform,
            lengths=[config.train_split_amount, config.val_split_amount, config.test_split_amount],
            generator=generator
        )

        train_subset = Subset(
            train_transform,
            indices=train_indices.indices
        )

        validation_subset = Subset(
            validation_and_test_transform,
            indices=validation_indices.indices
        )

        test_subset = Subset(
            validation_and_test_transform,
            indices=test_indices.indices
        )

        train = DataLoader(
            train_subset,
            batch_size = config.batch_size,
            shuffle = True,
            num_workers = 2
        )

        validation = DataLoader(
            validation_subset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=2
        )

        test = DataLoader(
            test_subset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=2
        )

        return train, validation, test

    def loader(self):
        return self.__load_dataset()