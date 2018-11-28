import os, time, pickle, argparse, models, utils, itertools
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import pdb
from torch.nn.init import xavier_uniform_
import numpy as np
from torchvision.utils import save_image


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='dataset', help='dataset')
    parser.add_argument('--in_ngc', type=int, default=3, help='input channel for generator')
    parser.add_argument('--out_ngc', type=int, default=3, help='output channel for generator')
    parser.add_argument('--in_ndc', type=int, default=3, help='input channel for discriminator')
    parser.add_argument('--out_ndc', type=int, default=1, help='output channel for discriminator')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--nb', type=int, default=8, help='the number of resnet block layers for generator')
    parser.add_argument('--img_size', type=int, default=64, help='input image size')
    parser.add_argument('--num_epoch', type=int, default=300)
    parser.add_argument('--lrD', type=float, default=0.0001, help='learning rate, default=0.0001')
    parser.add_argument('--lrG', type=float, default=0.0001, help='learning rate, default=0.0001')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    args = parser.parse_args()

    # Print args
    print('------------ Options -------------')
    for k, v in sorted(vars(args).items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    return args


# Create Dataset
class ClothingDataset(Dataset):
    def __init__(self, image_size, domain, mode):
        self.mode = mode
        self.image_array = self._create_dataset(image_size, domain, mode)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])
        
    def _create_dataset(self, image_size, domain, mode):
        data_dir = './data/dataset/'
        data_path = os.path.join(data_dir, domain + '_' + mode + '.npy')
        data = np.load(data_path)
            
        return data

    def __getitem__(self, idx):
        img = self.image_array[idx]
        return self.transform(Image.fromarray(img))

    def __len__(self):
        return len(self.image_array)


def dataloader_objects(args):

    A_front = ClothingDataset(args.img_size, 'A', 'front')
    A_back = ClothingDataset(args.img_size, 'A', 'back')
    A_side = ClothingDataset(args.img_size, 'A', 'side')
    B_front = ClothingDataset(args.img_size, 'B', 'front')
    B_back = ClothingDataset(args.img_size, 'B', 'back')
    B_side = ClothingDataset(args.img_size, 'B', 'side')

    A_front_loader = DataLoader(dataset=A_front, batch_size=args.batch_size, shuffle=True, drop_last=True)
    A_back_loader = DataLoader(dataset=A_back, batch_size=args.batch_size, shuffle=True, drop_last=True)
    A_side_loader = DataLoader(dataset=A_side, batch_size=args.batch_size, shuffle=True, drop_last=True)

    B_front_loader = DataLoader(dataset=B_front, batch_size=args.batch_size, shuffle=True, drop_last=True)
    B_back_loader = DataLoader(dataset=B_back, batch_size=args.batch_size, shuffle=True, drop_last=True)
    B_side_loader = DataLoader(dataset=B_side, batch_size=args.batch_size, shuffle=True, drop_last=True)

    
    dataloaders = [A_front_loader, A_back_loader, A_side_loader, B_front_loader, B_back_loader, B_side_loader]
    return dataloaders


def initialize_models(args, device):
    # network
    En_A = models.encoder(in_nc=args.in_ngc, nf=args.ngf, img_size=args.img_size).to(device)
    En_B = models.encoder(in_nc=args.in_ngc, nf=args.ngf, img_size=args.img_size).to(device)
    De_A = models.decoder(out_nc=args.out_ngc, nf=args.ngf).to(device)
    De_B = models.decoder(out_nc=args.out_ngc, nf=args.ngf).to(device)
    Disc_A = models.discriminator(in_nc=args.in_ndc, out_nc=args.out_ndc, nf=args.ndf, img_size=args.img_size).to(device)
    Disc_B = models.discriminator(in_nc=args.in_ndc, out_nc=args.out_ndc, nf=args.ndf, img_size=args.img_size).to(device)

    print('---------- models initialized -------------')
    utils.print_network(En_A)
    utils.print_network(En_B)
    utils.print_network(De_A)
    utils.print_network(De_B)
    utils.print_network(Disc_A)
    utils.print_network(Disc_B)
    print('-----------------------------------------------')

    # Parallelize code
    En_A = nn.DataParallel(En_A)
    En_B = nn.DataParallel(En_B)
    De_A = nn.DataParallel(De_A)
    De_B = nn.DataParallel(De_B)
    Disc_A = nn.DataParallel(Disc_A)
    Disc_B = nn.DataParallel(Disc_B)

    all_models = [En_A, En_B, De_A, De_B, Disc_A, Disc_B]
    return all_models


def setup():
    train_hist = {}
    train_hist['Disc_A_front_loss'] = []
    train_hist['Disc_A_back_loss'] = []
    train_hist['Disc_A_side_loss'] = []
    train_hist['Disc_B_front_loss'] = []
    train_hist['Disc_B_back_loss'] = []
    train_hist['Disc_B_side_loss'] = []
    train_hist['Gen_front_loss'] = []
    train_hist['Gen_back_loss'] = []
    train_hist['Gen_side_loss'] = []
    train_hist['per_epoch_time'] = []
    train_hist['total_time'] = []

    print('Setup complete!')
    return train_hist


def loader_epoch(A_loader, B_loader, device, En_A, En_B, De_A, De_B, Disc_A, Disc_B, real, fake, Disc_A_optimizer, Disc_B_optimizer, BCE_loss, L1_loss, train_hist, Disc_A_losses, Disc_B_losses, Gen_losses, angle):

    for A, B in zip(A_loader, B_loader):
        A, B = A.to(device), B.to(device)

        # train Disc_A & Disc_B
        # Disc real loss
        Disc_A_real = Disc_A(A)
        Disc_A_real_loss = BCE_loss(Disc_A_real, real)

        Disc_B_real = Disc_B(B)
        Disc_B_real_loss = BCE_loss(Disc_B_real, real)

        # Disc fake loss
        in_A, sp_A = En_A(A)
        in_B, sp_B = En_B(B)
      
        # De_A == B2A decoder, De_B == A2B decoder
        B2A = De_A(in_B + sp_A)
        A2B = De_B(in_A + sp_B)

        Disc_A_fake = Disc_A(B2A)
        Disc_A_fake_loss = BCE_loss(Disc_A_fake, fake)

        Disc_B_fake = Disc_B(A2B)
        Disc_B_fake_loss = BCE_loss(Disc_B_fake, fake)

        Disc_A_loss = Disc_A_real_loss + Disc_A_fake_loss
        Disc_B_loss = Disc_B_real_loss + Disc_B_fake_loss

        Disc_A_optimizer.zero_grad()
        Disc_A_loss.backward(retain_graph=True)
        Disc_A_optimizer.step()

        Disc_B_optimizer.zero_grad()
        Disc_B_loss.backward(retain_graph=True)
        Disc_B_optimizer.step()

        train_hist['Disc_A_' + angle + '_loss'].append(Disc_A_loss.item())
        train_hist['Disc_B_' + angle + '_loss'].append(Disc_B_loss.item())
        Disc_A_losses.append(Disc_A_loss.item())
        Disc_B_losses.append(Disc_B_loss.item())

        # train Generator
        Gen_A_fake_loss = BCE_loss(Disc_A_fake, real)
        Gen_B_fake_loss = BCE_loss(Disc_B_fake, real)

        # Generator Dual loss
        in_A_hat, sp_B_hat = En_B(A2B)
        in_B_hat, sp_A_hat = En_A(B2A)

        A_hat = De_A(in_A_hat + sp_A)
        B_hat = De_B(in_B_hat + sp_B)

        Gen_gan_loss = Gen_A_fake_loss + Gen_B_fake_loss
        Gen_dual_loss = (L1_loss(A_hat, A.detach()) ** 2 )+ (L1_loss(B_hat, B.detach()) ** 2)
        Gen_in_loss = (L1_loss(in_A_hat, in_A.detach()) ** 2) + (L1_loss(in_B_hat, in_B.detach()) ** 2)
        Gen_sp_loss = (L1_loss(sp_A_hat, sp_A.detach()) ** 2) + (L1_loss(sp_B_hat, sp_B.detach()) ** 2)

        Gen_loss = Gen_A_fake_loss + Gen_B_fake_loss + Gen_dual_loss + Gen_in_loss + Gen_sp_loss

        Gen_optimizer.zero_grad()
        Gen_loss.backward()
        Gen_optimizer.step()

        train_hist['Gen_' + angle + '_loss'].append(Gen_loss.item())
        Gen_losses.append(Gen_loss.item())

    # Save Images
    imgs_save = [A, B, A2B, B2A]
    image_dir = os.path.join(args.dataset + '_results', 'img')
    #save_image(imgs_save, image_dir + '/epoch_%d.png' % (epoch), nrow=2, normalize=True)
    result = torch.cat((A[0], B[0], A2B[0], B2A[0]), 2)
    path = os.path.join(args.dataset + '_results', 'img', str(epoch+1) + '_'+ angle +'_epoch.png')
    plt.imsave(path, (result.detach().cpu().numpy().transpose(1, 2, 0) + 1) / 2)

    return Disc_A_losses, Disc_B_losses, Gen_losses, train_hist


def training(args, epoch, device, dataloaders, all_models, BCE_loss, L1_loss, Gen_optimizer, Disc_A_optimizer, Disc_B_optimizer, train_hist):
    En_A, En_B, De_A, De_B, Disc_A, Disc_B = all_models
    A_front_loader, A_back_loader, A_side_loader, B_front_loader, B_back_loader, B_side_loader = dataloaders

    En_A.train()
    En_B.train()
    De_A.train()
    De_B.train()
    Disc_A.train()
    Disc_B.train()

    real = torch.ones(args.batch_size, 1, 1, 1).to(device)
    fake = torch.zeros(args.batch_size, 1, 1, 1).to(device)
    
    Disc_A_losses = []
    Disc_B_losses = []
    Gen_losses = []

    epoch_start_time = time.time()

    # Front loaders
    Disc_A_losses, Disc_B_losses, Gen_losses, train_hist = loader_epoch(A_front_loader, B_front_loader, device, En_A, En_B, De_A, De_B, Disc_A, Disc_B, real, fake, Disc_A_optimizer, Disc_B_optimizer, BCE_loss, L1_loss, train_hist, Disc_A_losses, Disc_B_losses, Gen_losses, 'front')

    # Back loaders
    Disc_A_losses, Disc_B_losses, Gen_losses, train_hist = loader_epoch(A_back_loader, B_back_loader, device, En_A, En_B, De_A, De_B, Disc_A, Disc_B, real, fake, Disc_A_optimizer, Disc_B_optimizer, BCE_loss, L1_loss, train_hist, Disc_A_losses, Disc_B_losses, Gen_losses, 'back')

    # Side loaders
    Disc_A_losses, Disc_B_losses, Gen_losses, train_hist = loader_epoch(A_side_loader, B_side_loader, device, En_A, En_B, De_A, De_B, Disc_A, Disc_B, real, fake, Disc_A_optimizer, Disc_B_optimizer, BCE_loss, L1_loss, train_hist, Disc_A_losses, Disc_B_losses, Gen_losses, 'side')

    per_epoch_time = time.time() - epoch_start_time
    train_hist['per_epoch_time'].append(per_epoch_time)
    print(
        '[%d/%d] - time: %f, Disc A loss: %f, Disc B loss: %f, Gen loss: %f' % (
            (epoch + 1), args.num_epoch, per_epoch_time, torch.mean(torch.FloatTensor(Disc_A_losses)),
            torch.mean(torch.FloatTensor(Disc_B_losses)), torch.mean(torch.FloatTensor(Gen_losses)),))

    # Save Images
    imgs_save = [A, B, A2B, B2A]
    image_dir = os.path.join(args.dataset + '_results', 'img')
    #save_image(imgs_save, image_dir + '/epoch_%d.png' % (epoch), nrow=2, normalize=True)
    result = torch.cat((A[0], B[0], A2B[0], B2A[0]), 2)
    path = os.path.join(args.dataset + '_results', 'img', str(epoch+1) + '_epoch.png')
    plt.imsave(path, (result.detach().cpu().numpy().transpose(1, 2, 0) + 1) / 2)

    return train_hist


def main():
    args = parse_arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = True

    # results save path
    if not os.path.isdir(os.path.join(args.dataset + '_results', 'img')):
        os.makedirs(os.path.join(args.dataset + '_results', 'img'))
    if not os.path.isdir(os.path.join(args.dataset + '_results', 'model')):
        os.makedirs(os.path.join(args.dataset + '_results', 'model'))

    # Get train and test dataloader objects
    dataloaders = dataloader_objects(args)

    # initialize models
    all_models = initialize_models(args, device)
    En_A, En_B, De_A, De_B, Disc_A, Disc_B = all_models

    # loss
    BCE_loss = nn.BCELoss().to(device)
    L1_loss = nn.L1Loss().to(device)

    # Adam optimizer
    Gen_optimizer = optim.Adam(itertools.chain(En_A.parameters(), De_A.parameters(), En_B.parameters(), De_B.parameters()), lr=args.lrG, betas=(args.beta1, args.beta2))
    Disc_A_optimizer = optim.Adam(Disc_A.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
    Disc_B_optimizer = optim.Adam(Disc_B.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

    # Begin model training
    train_hist = setup()

    start_time = time.time()

    for epoch in range(args.num_epoch):
        print('====================================================================================================')
        print("Epoch = ", epoch)
        epoch_start_time = time.time()

        # Train
        train_hist = training(args, epoch, device, dataloaders, all_models, BCE_loss, L1_loss, Gen_optimizer, Disc_A_optimizer, Disc_B_optimizer, train_hist)

    print('====================================================================================================')
    total_time = time.time() - start_time
    train_hist['total_time'].append(total_time)

    print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_time'])), args.num_epoch, total_time))

    print("Training Completed!")
    with open(os.path.join(args.dataset + '_results',  'train_hist.pkl'), 'wb') as f:
        pickle.dump(train_hist, f)


if __name__ == '__main__':
    main()
