# some training parameters
EPOCHS = 50
BATCH_SIZE = 8
NUM_CLASSES = 5
image_height = 240
image_width = 320
channels = 3
save_model_dir = "saved_model/model"
dataset_dir = "datasets/"
train_dir = dataset_dir + "train"
valid_dir = dataset_dir + "valid"
test_dir = dataset_dir + "test"

# choose a network
# model = "resnet18"
# model = "resnet34"
# model = "resnet101"
# model = "resnet152"
model = "resnet50"
pool = "netvlad"
kernel = 3
