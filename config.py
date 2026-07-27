# image settings
image_size = 128
flip_amount = 0.3
rotation_amount = 10
seed = 42
train_split_amount = 0.8
val_split_amount = 0.1
test_split_amount = 0.1

# dataset
path = "Data/"

# train settings
batch_size = 32
epochs = 40

# network settings
output_layer = 3
hidden_layer = 512
input_layer = 8 * 8 * 256
dropout = 0.5
learning_rate = 0.001