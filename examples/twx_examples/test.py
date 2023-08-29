import mindspore as ms
import mindcv
import mindspore.dataset as ds

from mindspore import Tensor, nn

def valid(dataloader):
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


model = mindcv.LeNet(in_channels=1)
ms.load_checkpoint(r"D:\Code\model_ckpt\LeNet5.ckpt", model)

batch_size = 512
dataset = ds.MnistDataset(r"D:\Code\Dataset\mnist", usage="test")
dataset = dataset.batch(batch_size, drop_remainder=True)
dataloader = dataset.create_tuple_iterator()

print(valid(dataloader))