import torch
import numpy as np
import os
from torch import nn
from torch.utils.data import DataLoader
from DataSet import MyDataset
from Net import RnnNet, Seq2seqNet
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class Train:
    def __init__(self, root):
        self.epoch = 100000
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.param_path = 'param/params.pt'

        # 加载训练集
        self.train_dataset = MyDataset(root)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=100, shuffle=True, num_workers=8)

        # 加载验证集
        self.test_dataset = MyDataset(root, is_train=False)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=10, shuffle=False, num_workers=0)

        # 定义网络
        # self.net = RnnNet().to(self.device)
        self.net = Seq2seqNet().to(self.device)

        # 加载参数
        if os.path.exists(self.param_path):
            self.net.load_state_dict(torch.load(self.param_path))

        # 定义优化器和损失
        self.optim = torch.optim.Adam(self.net.parameters())
        self.loss_func = nn.MSELoss()

    def __call__(self):
        self.summ = SummaryWriter('./logs')
        for epoch in range(self.epoch):
            loss_train_sum = 0
            for i, (img, label) in enumerate(tqdm(self.train_dataloader)):
                img = img.to(self.device)
                label = label.to(self.device)

                out = self.net(img)
                loss = self.loss_func(out, label)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                loss_train_sum += loss.detach().cpu().item()
            loss_train_avg = torch.true_divide(loss_train_sum, len(self.train_dataloader))

            loss_test_sum = 0
            acc = 0
            for i, (img, tage) in enumerate(tqdm(self.test_dataloader)):
                # 把数据的标签放入GPU进行计算
                input, test_tage = img.to(self.device), tage.to(self.device)
                test_output = self.net(input)

                perd = torch.argmax(test_output, 2).detach().cpu().numpy()
                label = torch.argmax(tage, 2).detach().cpu().numpy()

                loss = self.loss_func(test_output, test_tage)
                loss_test_sum += loss.cpu().item()
                acc += np.mean(np.all(perd == label, axis=1))

            loss_test_avg = torch.true_divide(loss_test_sum, len(self.test_dataloader))
            acc_avg = torch.true_divide(acc, len(self.test_dataloader))
            # add_scalars用来保存多个值， add_scalar只能保存一个
            self.summ.add_scalars("loss", {"train_avg_loss": loss_train_avg, "test_avg_loss": loss_test_avg},epoch)
            self.summ.add_scalar("acc", acc_avg, epoch)

            # 保存网络参数 w, b,不会自动创建文件 需要先将文件夹创建出来，按轮次保存，保存的格式为 .apk 或则 .t 文件  为二进制文件
            # 防止出现意外情况，保留参数
            torch.save(self.net.state_dict(), self.param_path)
            print(epoch, "训练损失",loss_train_avg.item(), "测试损失",loss_test_avg.item(), "得分",acc_avg.item())


if __name__ == '__main__':
    train = Train("img")
    train()
