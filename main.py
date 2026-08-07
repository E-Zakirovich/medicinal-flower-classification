from data import Dataset

def main():
    dataset = Dataset()
    train_loader, validation_loader, test_loader = dataset.load()

    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(validation_loader)}")
    print(f"Test batches: {len(test_loader)}")

    images, labels = next(iter(train_loader))
    print(f"Batch image shape: {images.shape}")
    print(f"Batch label shape: {labels.shape}")


if __name__ == '__main__':
    main()