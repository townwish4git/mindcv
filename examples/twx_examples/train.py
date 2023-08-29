import os
import numpy as np
import sys

import mindspore as ms
import mindcv
import mindspore.dataset as ds

from mindspore import Tensor, nn, ops

sys.path.append(r"D:\Code\Src\mindcv")

def read_train_dataset(root=r"D:\Code\Dataset\mnist", batch_size=16):
    dataset = ds.MnistDataset(dataset_dir=root, usage="train")
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataloader = dataset.create_tuple_iterator()
    return dataset, dataloader

batch_size = 512

dataset = ds.MnistDataset(dataset_dir=r"D:\Code\Dataset\mnist", usage="train")
dataset = dataset.batch(batch_size, drop_remainder=True)
ds_train, ds_valid = dataset.split([0.9, 0.1])
dataloader = ds_train.create_tuple_iterator()
dataloader_val = ds_valid.create_tuple_iterator()

model = mindcv.LeNet(in_channels=1)
loss = nn.CrossEntropyLoss(reduction="mean")
optimizer = nn.AdamWeightDecay(params=model.trainable_params(), learning_rate=1e-3)


def forward_fn(x, y):
    logits = model(x)
    losses = loss(logits, y)
    return losses, logits

grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)


def train_step(x, y):
    (losses, _), grads = grad_fn(x, y)
    optimizer(grads)
    return losses


def valid(dataloader=dataloader_val):
    model.set_train(False)
    correct_num = 0.
    total_num = 0.

    for data, label in dataloader:
        data = Tensor(data, dtype=ms.float32).resize((batch_size, 1, 28, 28))
        label = Tensor(label, dtype=ms.int32)
        logits = model(data)
        pred = logits.argmax(axis=1)
        correct_num += (pred == label).sum().asnumpy()
        total_num += label.shape[0]

    return correct_num / total_num


def epoch_train(max_epoch, epoch, dataloader=dataloader):
    model.set_train(True)
    _batch_size = dataset.get_dataset_size()

    for batch_index, (data, label) in enumerate(dataloader):
        data = Tensor(data, dtype=ms.float32).resize((batch_size, 1, 28, 28))
        label = Tensor(label, dtype=ms.int32)
        # print(data)
        losses = train_step(data, label)

        if batch_index % 5 == 0:
            losses = losses.asnumpy()
            print(f"loss:{losses:>7f}  [batch:{batch_index:>3d}/{_batch_size:>3d}][epoch:{epoch:>3d}/{max_epoch:>3d}]")


def train(max_epoch):
    best_acc = 0.0
    descend_epoch = 0
    for epoch in range(max_epoch):
        epoch_train(max_epoch ,epoch, dataloader)
        current_acc = valid(dataloader=dataloader_val)

        print("=" * 50)
        print("epoch:{}, accuracy:{}".format(epoch, current_acc))
        print("=" * 50)
        if current_acc > best_acc:
            best_acc = current_acc
            ms.save_checkpoint(save_obj=model, ckpt_file_name=r"D:\Code\model_ckpt\LeNet5.ckpt")
        else:
            descend_epoch += 1
            if descend_epoch >= 3:
                print(f"Accuracy on validation set hasn't beening increasing for a while, therefore the training is stopped.")
                break





if __name__ == "__main__":
    ckpt_dir = "./model_ckpt"
    train(max_epoch=50)