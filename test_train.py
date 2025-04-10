# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import gzip
import pickle
from struct import unpack

import matplotlib.pyplot as plt
import mynn as nn
import numpy as np
from draw_tools.plot import plot

# fixed seed for experiment
np.random.seed(309)

train_images_path = '.\dataset\MNIST\\train-images-idx3-ubyte.gz'
train_labels_path = '.\dataset\MNIST\\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28, 28, 1)

with gzip.open(train_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    train_labs = np.frombuffer(f.read(), dtype=np.uint8)


# choose 10000 samples from train set as validation set.
idx = np.random.permutation(np.arange(num))
# save the index.
with open('idx.pickle', 'wb') as f:
    pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# normalize from [0, 255] to [0, 1]
train_imgs = train_imgs.astype(np.float32) / train_imgs.max()
valid_imgs = valid_imgs.astype(np.float32) / valid_imgs.max()

# Convert to the format expected by the custom CNN model
train_imgs = train_imgs.transpose(0, 3, 1, 2) # (N, C, H, W)
valid_imgs = valid_imgs.transpose(0, 3, 1, 2) # (N, C, H, W)

model=nn.models.Model_CNN()
# optimizer = nn.optimizer.SGD(init_lr=0.06, model=linear_model)
optimizer = nn.optimizer.MomentGD(init_lr=0.001, model=model) # Adjusted learning rate
scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[2, 4], gamma=0.1) # Adjusted milestones and gamma
# scheduler = nn.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)
loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=train_labs.max()+1)

runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)

runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=5, log_iters=1, save_dir=r'./best_models')

_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)

plt.show()