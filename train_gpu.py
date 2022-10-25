import torch.optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
import torch
import time

# 准备数据集


train_data = torchvision.datasets.CIFAR10("datasetcifar10", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10("datasetcifar10", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# 训练 & 测试数据集的长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("The length of the train datasets：{}".format(train_data_size))
print("The length of the test datasets：{}".format(test_data_size))

# 利用DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
import torch
from torch import nn

# 搭建神经网络
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    tudui = Tudui()
    input = torch.ones(64, 3, 32, 32)
    output = tudui(input)
    print(output.shape)
tudui = Tudui()
tudui = tudui.cuda()

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()

#优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), learning_rate,)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练轮数
epoch = 300

# 添加tensorboard
writer = SummaryWriter('logs_train')

start_time = time.time()
for i in range(epoch):
    print("----------The {} epoch of training----------".format(i+1))

    # 训练步骤开始
    tudui.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()         # 优化器梯度清零
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print("training times： {}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print("total test loss： {}".format(total_test_loss))
    print("the accuracy of total test loss： {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1


    torch.save(tudui, "trained_models/tudui_{}_gpu.pth".format(i))
    print("model saved")

writer.close()
