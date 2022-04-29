import os 
import torch
from models import ASTModel 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import torchvision.transforms as transforms
import random
import shutil

from PIL import Image
device = torch.device("cuda:0")
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

lang_label = {"english": 0, "french": 1,"german":3, "spanish":4}
class LangDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        rmb面额分类任务的Dataset
        :param data_dir: str, 数据集所在路径
        :param transform: torch.transform，数据预处理
        """
        self.label_name = {"english": 0, "french": 1,"german":3, "spanish":4}
        self.data_info = self.get_img_info(data_dir)  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')     # 0~255

        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等

        return img, label

    def __len__(self):
        return len(self.data_info)
    
    
    def get_img_info(self, data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            # 遍历类别
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.png'), img_names))

                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = lang_label[sub_dir]
                    data_info.append((path_img, int(label)))

        return data_info
 

def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


if __name__ == '__main__':

    random.seed(1)

    MAX_EPOCH = 10
    BATCH_SIZE = 2
    LR = 0.001
    log_interval = 10
    val_interval = 1
    
    dataset_dir = os.path.join(os.getcwd(), "../raw_data")
    split_dir = os.path.join(os.getcwd(), "../raw_data/", "../image_split")
    train_dir = os.path.join(split_dir, "train")
    valid_dir = os.path.join(split_dir, "valid")
    test_dir = os.path.join(split_dir, "test")

    train_pct = 0.6
    valid_pct = 0.2
    test_pct = 0.2
    for root, dirs, files in os.walk(dataset_dir):
        dirs = ['german', 'english', 'french', 'spanish']
        for sub_dir in dirs:

            imgs = os.listdir(os.path.join(root, sub_dir))
            imgs = list(filter(lambda x: x.endswith('.png'), imgs))
            random.shuffle(imgs)
            img_count = len(imgs)

            train_point = int(img_count * train_pct)
            valid_point = int(img_count * (train_pct + valid_pct))

            for i in range(img_count):
                if i < train_point:
                    out_dir = os.path.join(train_dir, sub_dir)
                elif i < valid_point:
                    out_dir = os.path.join(valid_dir, sub_dir)
                else:
                    out_dir = os.path.join(test_dir, sub_dir)

                makedir(out_dir)

                target_path = os.path.join(out_dir, imgs[i])
                src_path = os.path.join(dataset_dir, sub_dir, imgs[i])

                shutil.copy(src_path, target_path)

            print('Class:{}, train:{}, valid:{}, test:{}'.format(sub_dir, train_point, valid_point-train_point,
                                                                 img_count-valid_point))
        # train_dir = "/root/ast/image_split/train"
        train_data = LangDataset(data_dir=train_dir, transform=transform)
        valid_data = LangDataset(data_dir=valid_dir, transform=transform)
        print(train_data[0])
        # 构建DataLoder
        train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)
        
        model = ASTModel(label_dim=4, \
         fstride=10, tstride=10, \
         input_fdim=128, input_tdim=200, \
         imagenet_pretrain=True, audioset_pretrain=False, \
         model_size='base384')
        model.to(device)
        
        # model.initialize_weights()
        import torch.optim as optim

        # CrossEntropyLoss就是我们需要的损失函数
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        
        print("Start Training...")
        for epoch in range(10):
            # 我们用一个变量来记录每100个batch的平均loss
            loss100 = 0.0
            # 我们的dataloader派上了用场
            for i, data in enumerate(train_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device) # 注意需要复制到GPU
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                loss100 += loss.item()
                if i % 100 == 99:
                    print('[Epoch %d, Batch %5d] loss: %.3f' %
                          (epoch + 1, i + 1, loss100 / 100))
                    loss100 = 0.0

        print("Done Training!")
