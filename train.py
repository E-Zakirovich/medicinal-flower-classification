import torch
import torch.optim as optim
import config


class Train:
    def __init__(
            self,
            model,
            train_loader,
            validation_loader,
            device,
                 ):
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.device = device
        self.learning_rate = config.learning_rate


    def fit(self, epochs):
        ...
