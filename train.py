import torch
import torch.optim as optim

import config


class Train:
    def __init__(
            self,
            model,
            train_loader,
            validation_loader,
            device=torch.device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.device = device

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
        )


    def fit(self, epochs):

        for epoch in range(epochs):

            self.model.train()
            running_loss = 0
            correct = 0
            total = 0

            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                output = self.model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_loss = running_loss / total
            train_acc = correct / total

            self.model.eval()
            validation_loss = 0
            validation_correct = 0
            validation_total = 0

            with torch.no_grad():
                for images, labels in self.validation_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    output = self.model(images)
                    loss = self.criterion(output, labels)

                    validation_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(output.data, 1)
                    validation_correct += (predicted == labels).sum().item()
                    validation_total += labels.size(0)

            val_loss = validation_loss / validation_total
            val_acc = validation_correct / validation_total

            print(
                f"Epoch [{epoch + 1}/{epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.2%} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.2%}"
            )

        return self.model