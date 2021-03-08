import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
from torch.autograd import Variable
import os
import fnmatch
from datetime import datetime
import argparse




def NormalizeData(data):
    return 2*(data - np.min(data)) / (np.max(data) - np.min(data))-1

def Z_Score_NormalizeData(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean)/std

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

    
class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(input_size)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, data):
        data_flat = data.view(data.size(0), -1)
        validity = self.model(data_flat)

        return validity


class Generator(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                normlayer = nn.BatchNorm1d(out_feat, 0.8)
                layers.append(normlayer)
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(
            *block(input_size, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(input_size))),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


# Parameters
device = "cuda" if torch.cuda.is_available() else "cpu"
parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", type=int, default=10, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=3, help="batch size (cannot be 1 for training)")
parser.add_argument("--learning_rate", type=float, default=3e-4, help="learning rate")
parser.add_argument("--save_model", type=bool, default=False, help="Save model?")
parser.add_argument("--mode", default='Tran', help="Train or Test")
parser.add_argument("--dlv_path", default='data/spxw_call_dlv_0.csv', help="Load Path for Testing DLV")
parser.add_argument("--recursive", type=bool, default=True, help="Recursively Testing DLV?")
parser.add_argument("--recursive_length", type=int, default=2, help="Length of Recursively Testing DLV")
opt = parser.parse_args()

input_size = 3 * 8 * 1  # 24
min = -0.3406041746517267
max = 7.910383934523679
mean = 59.93618814995941
std = 175.9702100282392
if opt.mode == 'Train':
    # initialize networks
    disc = Discriminator(input_size).to(device)
    gen = Generator(input_size).to(device)
    print(disc)
    print(gen)
    optimizerD = optim.Adam(disc.parameters(), lr=opt.learning_rate)
    optimizerG = optim.Adam(gen.parameters(), lr=opt.learning_rate)
    adversarial_loss = nn.BCELoss()


    # Find all file names
    s = find('*.csv', 'data/')

    # load & normalize data
    dt_arry = np.zeros((len(s), 3, 8))
    i = 0
    for f in s:
        dt_arry[i,:,:] = np.loadtxt(open(f, "rb"), delimiter=",")
        i = i+1
    dt_arry = Z_Score_NormalizeData(dt_arry)
    dt_arry = NormalizeData(dt_arry)
    dt_tensor = torch.Tensor(dt_arry)
    my_dataset = TensorDataset(dt_tensor)
    dataloader = torch.utils.data.DataLoader(
        my_dataset,
        batch_size= opt.batch_size,
        shuffle=False,
        drop_last=True
    )

    # # load data (for dummy data)
    # result1 = np.loadtxt(open("E:/Northwestern/COMP_SCI 496 Adv Deep Learning/spxw_call_dlv.csv", "rb"), delimiter=",")
    # result2 = np.loadtxt(open("E:/Northwestern/COMP_SCI 496 Adv Deep Learning/2.csv", "rb"), delimiter=",")
    # result3 = np.loadtxt(open("E:/Northwestern/COMP_SCI 496 Adv Deep Learning/3.csv", "rb"), delimiter=",")
    # result4 = np.loadtxt(open("E:/Northwestern/COMP_SCI 496 Adv Deep Learning/4.csv", "rb"), delimiter=",")
    # result5 = np.loadtxt(open("E:/Northwestern/COMP_SCI 496 Adv Deep Learning/5.csv", "rb"), delimiter=",")
    # result6 = np.loadtxt(open("E:/Northwestern/COMP_SCI 496 Adv Deep Learning/6.csv", "rb"), delimiter=",")
    # result7 = np.loadtxt(open("E:/Northwestern/COMP_SCI 496 Adv Deep Learning/7.csv", "rb"), delimiter=",")
    # result8 = np.loadtxt(open("E:/Northwestern/COMP_SCI 496 Adv Deep Learning/8.csv", "rb"), delimiter=",")
    # result9 = np.loadtxt(open("E:/Northwestern/COMP_SCI 496 Adv Deep Learning/9.csv", "rb"), delimiter=",")
    # result = np.stack((result1, result2, result3, result4, result5, result6, result7, result8, result9))
    # tensor_x = torch.Tensor(result)
    # my_dataset = TensorDataset(tensor_x)
    # dataloader = torch.utils.data.DataLoader(
    #     my_dataset,
    #     batch_size= batch_size,
    #     shuffle=True,
    # )

    # initialize and normalize sigma_t for testing
    fixed_noise = dt_arry[0,:,:].reshape(1,24)
    fixed_noise = torch.Tensor(fixed_noise)
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    for epoch in range(opt.num_epochs):
        for batch_idx, data in enumerate(dataloader):
            # Adversarial ground truths
            valid = Variable(Tensor(data[0].size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(data[0].size(0), 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(data[0].type(Tensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizerG.zero_grad()

            # Sample noise as generator input
            z  = Variable(Tensor(np.random.normal(0, 1, (data[0].shape[0], 24))))

            # Generate a batch of images
            gen_dlvs = gen(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(disc(gen_dlvs), valid)

            g_loss.backward()
            optimizerG.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizerD.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(disc(real_imgs), valid)
            fake_loss = adversarial_loss(disc(gen_dlvs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizerD.step()


            if batch_idx == 0:
                print(
                    f"Epoch [{epoch}/{opt.num_epochs}] Batch {batch_idx}/{len(dataloader)} \
                        Loss D: {d_loss:.4f}, loss G: {g_loss:.4f}"
                )
    ## temp test after training
    with torch.no_grad():
        gen.eval()   #need to be eval mode since feeding one single sample to batchnorm layer
        fake = gen(fixed_noise).reshape(3, 8)
        fake = (fake+1)/2*(max-min)+min
        fake_in_right_scale = (fake * std) + mean
        np.savetxt("result/result.csv", fake_in_right_scale, delimiter=",")    

    if opt.save_model:
        print('model saving to: model/generater.pt')
        torch.save(gen.state_dict(), 'model/generater.pt')
else:
    with torch.no_grad():
        # load model and DLV with start time-series
        generator = Generator(input_size).to(device)
        generator.load_state_dict(torch.load('model/generater.pt'))
        generator.eval()
        test_dlv = np.loadtxt(open(opt.dlv_path, "rb"), delimiter=",")
        test_dlv= Z_Score_NormalizeData(test_dlv)
        test_dlv = NormalizeData(test_dlv)
        print('start with testing dlv:' +opt.dlv_path)
        if opt.recursive:
            print('start recursively testing')
            result = test_dlv 
            for i in range(opt.recursive_length):
                gaussian_noise = np.random.normal(0, 0.3, 24).reshape(1,24) #add gaussian noise
                if torch.is_tensor(result):
                    result = result.numpy()
                result = result.reshape(1,24) + gaussian_noise
                result = result.astype(float)
                tensor_x = torch.Tensor(result)
                result = generator(tensor_x).reshape(3, 8)
                fake = (result+1)/2*(max-min)+min
                fake_in_right_scale = (fake+1)/2*(max-min)+min
                filename = 'result/result' + str(i+1) + '.csv'   # 1 indexing
                print('result save to:'+filename)
                np.savetxt(filename, fake_in_right_scale, delimiter=",")
        else:
            print('start testing single dlv')
            tensor_x = torch.Tensor(test_dlv)
            result = generator(tensor_x).reshape(3, 8)
            fake = (result+1)/2*(max-min)+min
            fake_in_right_scale = (fake+1)/2*(max-min)+min
            print('result save to: result/result.csv')
            np.savetxt("result/result.csv", fake_in_right_scale, delimiter=",")
                