import argparse
import os
import numpy as np
import math
import sys
from PIL import Image

import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data

from torch.autograd import Variable
from torchvision.utils import save_image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd


import helper
import matplotlib.pyplot as plt
import numpy as np
from glob import glob


os.makedirs("images", exist_ok=True)
os.makedirs("fake_imgs", exist_ok=True)
os.makedirs("mdl_state", exist_ok=True)
os.makedirs("report", exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
#parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
#parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
#parser.add_argument("--channels", type=int, default=3, help="number of image channels")
#parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
#parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print("\n Defalu parameter ---------------")
print(opt)

img_size = 64
#img_shape = (opt.channels, opt.img_size, opt.img_size)
g_img_shape = (3, img_size, img_size)
cuda = True if torch.cuda.is_available() else False

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(g_img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *g_img_shape)
        return img

d_img_shape = (3, img_size, img_size)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(d_img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return

class MyDataset(data.Dataset):
    def __init__(self, root, transform=None):
        eval_lst = os.listdir(root)
        eval_lst.sort()
        img_p = [os.path.join(root, x) for x in eval_lst if is_image_file(x)]
        self.root = root
        self.transform = transform
        self.img_p = img_p

    def __getitem__(self, index):
        target = self.img_p[index]
        img = Image.open(target).convert('RGB')
        #print("MyDataset1: ", img.size)
        if self.transform:
            img = self.transform(img)
        img = np.array(img)
        #print("MyDataset2: ", img.shape)
        return img, target

    def __len__(self):
        return len(self.img_p)

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

# Configure data loader
train_dir = '../data/img_celeb_small'
#train_dir = '../data/img_celeb'

transform_train1 = transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    #transforms.ToPILImage(),
])

'''
        transforms.Resize(size=(resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
'''
#batch_nums = 64
#batch_nums = 3600
#batch_nums = 14400
#batch_nums = 21600
batch_nums = 43200


trainset      = MyDataset(root=train_dir, transform=transform_train1)
trainloader   = torch.utils.data.DataLoader(trainset, batch_size=batch_nums, shuffle=False)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=20, shuffle=False)


# Loss weight for gradient penalty
lambda_gp = 10

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor #if cuda else torch.FloatTensor

# UnNormalize
unorm = UnNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""

    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))

    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)

    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# ----------
#  Check image
# ----------

#img = Image.open('./img_celeb_small\\000114.jpg')
#img_t = transform_train1(img)

def check_set():
    print("\n --------- Check trainset  \n ")
    images = helper.get_batch(glob(os.path.join(train_dir, '*.jpg'))[:9], 112, 112, 'RGB')
    #plt.imshow(helper.images_square_grid(images))
    #plt.show()

    n_epochs = 1
    for epoch in range(n_epochs):
        for i, (imgs, _) in enumerate(trainloader, 0):
            imgs = imgs.cuda()
            if (epoch % 10 == 0 and i == 0):
                print ("batch size ", imgs.shape)
                c_imgs = np.ones((9, img_size, img_size, 3))
                imgs = imgs.cpu()
                for jj in range(9):
                    c_img = unorm(imgs[jj])
                    c_img = transforms.ToPILImage()(c_img)
                    c_imgs[jj] = c_img
                plt.imshow(helper.images_square_grid(c_imgs))
                plt.show()
                output_fig(c_imgs[:9], './images/input_img_chk.jpg')
            #real_validity = discriminator(imgs)



###################################################
def train_set(load_mdl, save_mdl, save_loss_csv):
    print('\n [INFO] Start Training ')
    n_epochs = 2000
    d_loss_vals = []
    g_loss_vals = []

    if(load_mdl):
        generator.load_state_dict(torch.load('./mdl_state/gemdl_save'))
        discriminator.load_state_dict(torch.load('./mdl_state/demdl_save'))

    generator.cuda()
    discriminator.cuda()
    generator.train()
    discriminator.train()

    for epoch in range(n_epochs):
        print("----------New Epoch ", epoch)
        d_run_loss = 0.0
        g_run_loss = 0.0
        for i, (imgs, _) in enumerate(trainloader):
            # Configure input
            real_imgs = imgs.cuda()
            optimizer_D.zero_grad()

            z = Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)))
            fake_imgs = generator(z)

            real_validity = discriminator(real_imgs)
            fake_validity = discriminator(fake_imgs)
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)

            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
            d_loss.backward()
            optimizer_D.step()
            optimizer_G.zero_grad()
            # Train the generator every n_critic steps
            if i % opt.n_critic == 0:
                fake_imgs = generator(z)
                fake_validity = discriminator(fake_imgs)
                g_loss = -torch.mean(fake_validity)
                g_loss.backward()
                optimizer_G.step()

            if (epoch % 50 == 49 and i == 0):
                print("[Epoch %d/%d] [D loss: %f] [G loss: %f]" %(epoch, n_epochs,  d_loss.item(), g_loss.item()) )
                fake_imgs = fake_imgs.cpu()
                f_imgs  = fake_imgs[:9]

                fidx = str(epoch).zfill(4)
                fpic_name = './images/epo_'+fidx+'.jpg'

                cfake_imgs = np.ones((9, img_size, img_size, 3))
                for idx in range(9):
                    f_imgs[idx] = unorm(f_imgs[idx])
                    cfake = transforms.ToPILImage()(f_imgs[idx])
                    cfake_imgs[idx] = cfake
                #save_image(cfake_imgs.data[:9], fpic_name, nrow=3, normalize=True)
                output_fig(cfake_imgs[:9], fpic_name)

            d_run_loss += d_loss.item()
            g_run_loss += g_loss.item()

        d_loss_vals.append(d_run_loss)
        g_loss_vals.append(g_run_loss)


    print('\n [INFO] Finished Training ')

    if(save_mdl):
        torch.save(generator.state_dict(), './mdl_state/gemdl_save')
        torch.save(discriminator.state_dict(), './mdl_state/demdl_save')

    if(save_loss_csv):
        fo = open('./report/train_loss.csv', "w")
        fo.write("epoch,n_epochs,d_loss,g_loss\n")
        for idx in range(n_epochs):
            line_str = str(idx)+","+str(n_epochs)+","+str(d_loss_vals[idx])+","+str(g_loss_vals[idx])+'\n'
            fo.write(line_str)
        fo.close()

###################################################
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    #npimg = img.numpy()
    npimg = img.detach().numpy()
    plt.imshow(npimg)
    plt.show()
#plt.imshow(helper.images_square_grid(images))

def output_fig(images_array, file_name="./results"):
    plt.figure(figsize=(6, 6), dpi=100)
    plt.imshow(helper.images_square_grid(images_array))
    plt.axis("off")
    plt.savefig(file_name+'.png', bbox_inches='tight', pad_inches=0)


def eval_generator(load_mdl):
    print('\n [INFO] Start Gen Fake ')
    #Tensor = torch.FloatTensor

    if(load_mdl):
        #generator.load_state_dict(torch.load('./mdl_state/gemdl_save'))
        generator.load_state_dict(torch.load('./mdl_state/gemdl_save_600'))
    generator.eval()
    generator.cuda()

    img_batch = 9
    grid_iters = 5
    for ii in range(grid_iters):
        z = Tensor(np.random.normal(0, 1, (img_batch, opt.latent_dim)))
        #z.cuda()
        g_imgs = generator(z)
        g_imgs = g_imgs.cpu()
        c_imgs = np.ones((img_batch, img_size, img_size, 3))

        for jj in range(img_batch):
            g_imgs[jj] = unorm(g_imgs[jj])
            c_imgs[jj] = transforms.ToPILImage()(g_imgs[jj])

        output_fig(c_imgs[:9], file_name="./fake_imgs/{}_image".format(str.zfill(str(ii+1), 3)))

    print('\n [INFO] Finished Gen Fake ')
###################################################
#check_set()
train_set(load_mdl=True, save_mdl=True, save_loss_csv=True)
#eval_generator(load_mdl=True)

