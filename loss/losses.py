import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal, Independent, kl
from torch.autograd import Variable
CE = torch.nn.BCELoss(reduction='sum')

class cross_entropy(nn.Module):
    def __init__(self, weight=None, reduction='mean',ignore_index=256):
        super(cross_entropy, self).__init__()
        self.weight = weight
        self.ignore_index =ignore_index
        self.reduction = reduction


    def forward(self,input, target):
        target = target.long()
        if target.dim() == 4:
            target = torch.squeeze(target, dim=1)
        if input.shape[-1] != target.shape[-1]:
            input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

        return F.cross_entropy(input=input, target=target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)


class Mutual_info_reg(nn.Module):
    def __init__(self, input_channels, channels, latent_size=2):
        super(Mutual_info_reg, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        # self.bn1 = nn.BatchNorm2d(channels)
        self.layer2 = nn.Conv2d(3*input_channels, channels, kernel_size=4, stride=2, padding=1)
        # self.bn2 = nn.BatchNorm2d(channels)
        self.layer3 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)
        self.channel = channels
        #self.dimension = int(size / 4 )
        #outsize2 = int(outsize1*0.5+1)
        self.fc1_rgb3 = nn.LazyLinear(latent_size)
        self.fc2_rgb3 = nn.LazyLinear(latent_size)
        self.fc1_depth3 = nn.LazyLinear(latent_size)
        self.fc2_depth3 = nn.LazyLinear(latent_size)

        self.leakyrelu = nn.LeakyReLU()
        self.tanh = torch.nn.Tanh()

    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        return kl_div

    def reparametrize(self, mu, logvar):
        device = mu.device  # 获取 mu 的设备
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_().to(device=device)
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, rgb_feat, depth_feat):
        #print(rgb_feat.shape[1]*rgb_feat.shape[2]*rgb_feat.shape[3],"ininininininininininininini")
        rgb_feat = self.layer3(self.leakyrelu(self.layer1(rgb_feat)))
        depth_feat = self.layer4(self.leakyrelu(self.layer2(depth_feat)))
        a, b, c, d = rgb_feat.shape
        rgb_feat = rgb_feat.view(a, -1)

        depth_feat = depth_feat.view(a, -1)
        mu_rgb = self.fc1_rgb3(rgb_feat)
        logvar_rgb = self.fc2_rgb3(rgb_feat)
        mu_depth = self.fc1_depth3(depth_feat)
        logvar_depth = self.fc2_depth3(depth_feat)

        mu_depth = self.tanh(mu_depth)
        mu_rgb = self.tanh(mu_rgb)
        logvar_depth = self.tanh(logvar_depth)
        logvar_rgb = self.tanh(logvar_rgb)

        z_rgb = self.reparametrize(mu_rgb, logvar_rgb) #z_rgb

        dist_rgb = Independent(Normal(loc=mu_rgb, scale=torch.exp(logvar_rgb)), 1)
        z_depth = self.reparametrize(mu_depth, logvar_depth)
        dist_depth = Independent(Normal(loc=mu_depth, scale=torch.exp(logvar_depth)), 1)
        bi_di_kld = torch.mean(self.kl_divergence(dist_rgb, dist_depth)) + torch.mean(
            self.kl_divergence(dist_depth, dist_rgb))
        z_rgb_norm = torch.sigmoid(z_rgb) #z_rgb
        z_depth_norm = torch.sigmoid(z_depth)
        ce_rgb_depth = CE(z_rgb_norm, z_depth_norm.detach())
        ce_depth_rgb = CE(z_depth_norm, z_rgb_norm.detach())
        latent_loss = ce_rgb_depth + ce_depth_rgb - bi_di_kld
        # latent_loss = torch.abs(cos_sim(z_rgb,z_depth)).sum()

        return latent_loss 
    
class Mutual_info_regsa(nn.Module):
    def __init__(self, input_channels, channels, latent_size = 4):
        super(Mutual_info_regsa, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        # self.bn1 = nn.BatchNorm2d(channels)
        self.layer2 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        # self.bn2 = nn.BatchNorm2d(channels)
        self.layer3 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

        self.channel = channels

        # self.fc1_rgb1 = nn.Linear(channels * 1 * 16 * 16, latent_size)
        # self.fc2_rgb1 = nn.Linear(channels * 1 * 16 * 16, latent_size)
        # self.fc1_depth1 = nn.Linear(channels * 1 * 16 * 16, latent_size)
        # self.fc2_depth1 = nn.Linear(channels * 1 * 16 * 16, latent_size)
        # 
        # self.fc1_rgb2 = nn.Linear(channels * 1 * 22 * 22, latent_size)
        # self.fc2_rgb2 = nn.Linear(channels * 1 * 22 * 22, latent_size)
        # self.fc1_depth2 = nn.Linear(channels * 1 * 22 * 22, latent_size)
        # self.fc2_depth2 = nn.Linear(channels * 1 * 22 * 22, latent_size)

        self.fc1_rgb3 = nn.Linear(channels * 1 * 32 * 32, latent_size)
        self.fc2_rgb3 = nn.Linear(channels * 1 * 32 * 32, latent_size)
        self.fc1_depth3 = nn.Linear(channels * 1 * 32 * 32, latent_size)
        self.fc2_depth3 = nn.Linear(channels * 1 * 32 * 32, latent_size)

        self.leakyrelu = nn.LeakyReLU()
        self.tanh = torch.nn.Tanh()

    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        return kl_div

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, rgb_feat, depth_feat):
        rgb_feat = self.layer3(self.leakyrelu(self.layer1(rgb_feat)))
        depth_feat = self.layer4(self.leakyrelu(self.layer2(depth_feat)))
        # print(rgb_feat.size())
        # print(depth_feat.size())
        # if rgb_feat.shape[2] == 16:
        #     rgb_feat = rgb_feat.view(-1, self.channel * 1 * 16 * 16)
        #     depth_feat = depth_feat.view(-1, self.channel * 1 * 16 * 16)
        #
        #     mu_rgb = self.fc1_rgb1(rgb_feat)
        #     logvar_rgb = self.fc2_rgb1(rgb_feat)
        #     mu_depth = self.fc1_depth1(depth_feat)
        #     logvar_depth = self.fc2_depth1(depth_feat)
        # elif rgb_feat.shape[2] == 22:
        #     rgb_feat = rgb_feat.view(-1, self.channel * 1 * 22 * 22)
        #     depth_feat = depth_feat.view(-1, self.channel * 1 * 22 * 22)
        #     mu_rgb = self.fc1_rgb2(rgb_feat)
        #     logvar_rgb = self.fc2_rgb2(rgb_feat)
        #     mu_depth = self.fc1_depth2(depth_feat)
        #     logvar_depth = self.fc2_depth2(depth_feat)
        # else:
        rgb_feat = rgb_feat.view(-1, self.channel * 1 * 32 * 32)
        depth_feat = depth_feat.view(-1, self.channel * 1 * 32 * 32)
        mu_rgb = self.fc1_rgb3(rgb_feat)
        logvar_rgb = self.fc2_rgb3(rgb_feat)
        mu_depth = self.fc1_depth3(depth_feat)
        logvar_depth = self.fc2_depth3(depth_feat)

        mu_depth = self.tanh(mu_depth)
        mu_rgb = self.tanh(mu_rgb)
        logvar_depth = self.tanh(logvar_depth)
        logvar_rgb = self.tanh(logvar_rgb)
        z_rgb = self.reparametrize(mu_rgb, logvar_rgb)
        dist_rgb = Independent(Normal(loc=mu_rgb, scale=torch.exp(logvar_rgb)), 1)
        z_depth = self.reparametrize(mu_depth, logvar_depth)
        dist_depth = Independent(Normal(loc=mu_depth, scale=torch.exp(logvar_depth)), 1)
        bi_di_kld = torch.mean(self.kl_divergence(dist_rgb, dist_depth)) + torch.mean(
            self.kl_divergence(dist_depth, dist_rgb))
        z_rgb_norm = torch.sigmoid(z_rgb)
        z_depth_norm = torch.sigmoid(z_depth)
        ce_rgb_depth = CE(z_rgb_norm,z_depth_norm.detach())
        ce_depth_rgb = CE(z_depth_norm, z_rgb_norm.detach())
        latent_loss = ce_rgb_depth+ce_depth_rgb-bi_di_kld
        # latent_loss = torch.abs(cos_sim(z_rgb,z_depth)).sum()

        return latent_loss