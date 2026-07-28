import torch
from model.cnn import CNN
from dataset import Data
from train import Train
import config


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    train_data, validation_data, test_data = Data().loader()
    model = CNN()

    trainer = Train(model, train_data, validation_data, device)

    t_model = trainer.fit(config.epochs)

    torch.save(t_model.state_dict(), "trained_model.pth")

    print("Training completed!")
    print("Model saved as trained_model.pth")


if __name__ == '__main__':
    main()