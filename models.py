import utils
import torch.nn as nn

class encoder(nn.Module):
    # initializers
    def __init__(self, in_nc, nf=32, img_size=64):
        super(encoder, self).__init__()
        self.input_nc = in_nc
        self.nf = nf
        self.img_size = img_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_nc, out_channels=nf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=nf, out_channels=nf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=nf * 2, out_channels=nf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.independent_feature = nn.Sequential(
            nn.Conv2d(in_channels=nf * 4, out_channels=nf * 8, kernel_size=4, stride=2, padding=1),
        )
        self.specific_feature = nn.Sequential(
            nn.Linear(in_features=(nf * 4) * (img_size // 8) * (img_size // 8), out_features=nf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=nf * 8, out_features=nf * 8),
        )

        utils.initialize_weights(self)

    # forward method
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        i = self.independent_feature(x)
        f = x.view(-1, (self.nf * 4) * (self.img_size // 8) * (self.img_size // 8))
        s = self.specific_feature(f)
        s = s.unsqueeze(2)
        s = s.unsqueeze(3)

        return i, s


class decoder(nn.Module):
    # initializers
    def __init__(self, out_nc, nf=32):
        super(decoder, self).__init__()
        self.output_nc = out_nc
        self.nf = nf

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nf * 8, out_channels=nf * 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nf * 4, out_channels=nf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(nf * 2),
            nn.ReLU(inplace=True),
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nf * 2, out_channels=nf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(nf),
            nn.ReLU(inplace=True),
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nf, out_channels=out_nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

        utils.initialize_weights(self)

    # forward method
    def forward(self, x):
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)

        return x


class discriminator(nn.Module):
    # initializers
    def __init__(self, in_nc, out_nc, nf=32, img_size=64):
        super(discriminator, self).__init__()
        self.input_nc = in_nc
        self.output_nc = out_nc
        self.nf = nf
        self.img_size = img_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_nc, out_channels=nf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=nf, out_channels=nf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=nf * 2, out_channels=nf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=nf * 4, out_channels=nf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=(nf * 8) * (img_size // 16) * (img_size // 16), out_features=nf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=nf * 8, out_features=out_nc),
            nn.Sigmoid(),
        )

        utils.initialize_weights(self)

    # forward method
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        f = x.view(-1, (self.nf * 8) * (self.img_size // 16) * (self.img_size // 16))
        d = self.fc(f)
        d = d.unsqueeze(2)
        d = d.unsqueeze(3)

        return d
