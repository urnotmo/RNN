import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

data_trans = transforms.ToTensor()

class MyDataset(Dataset):
    def __init__(self, root, is_train = True):
        self.trans = data_trans
        self.data = []
        sub_dir = 'train' if is_train else 'test'
        root = os.path.join(root, sub_dir)
        for filename in os.listdir(root):
            img_path = os.path.join(root, filename) # 图片的地址
            trage = filename.split('.')[0] # 图片对应的标签
            y = self.one_hot(trage)
            self.data.append([img_path, y])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path,label = self.data[index]
        img_data = Image.open(img_path)
        img = self.trans(img_data)
        return img, label

    def one_hot(self, x):
        h = torch.zeros(4,10)
        for i in range(4):
            index = int(x[i])
            h[i][index] = 1
        return h

if __name__ == '__main__':
    data = MyDataset(r"D:\data\RNN\img", is_train=True)
    img, label = data[0]
    print(label)
    dataloadet = DataLoader(data, batch_size=100, shuffle=True)
    for i, (x,y) in enumerate(dataloadet):
        print(x.shape)
        print(y.shape)
