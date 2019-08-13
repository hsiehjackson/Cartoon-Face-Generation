import torch.nn as nn
import torch

class ACGenerator(nn.Module):
    def __init__(self, opt, n_filters):
        super(ACGenerator, self).__init__()
        
        self.image_size = opt.img_size
        self.init_size = opt.img_size // 2 ** opt.n_layers
        self.n_layers = opt.n_layers
        self.n_filters = n_filters
        self.project = nn.Linear(opt.latent_dim + opt.all_classes_dim, n_filters[0] * (self.init_size) ** 2)
        self.input_layer = nn.Sequential(
        	nn.BatchNorm2d(n_filters[0]), 
        	nn.ReLU())    
        self.deconvolution_block = nn.ModuleList(\
            [nn.Sequential(
            	nn.ConvTranspose2d(n_filters[i] + opt.all_classes_dim, n_filters[i + 1], kernel_size = 4, stride = 2, padding = 1),
            	nn.BatchNorm2d(n_filters[i + 1]), nn.ReLU())
            for i in range(opt.n_layers - 1)])
        self.output_layer = nn.Sequential(
        	nn.ConvTranspose2d(n_filters[-1], opt.channels, kernel_size = 4, stride = 2, padding = 1), 
        	nn.Tanh())

    def forward(self, x, condition):
        x = torch.cat([x, condition], 1)
        x = self.project(x).view(-1, self.n_filters[0], self.init_size, self.init_size)
        x = self.input_layer(x)   
        condition = condition.view(condition.size(0), condition.size(1), 1, 1)
        for i in range(self.n_layers - 1):
            x = torch.cat([x, condition.expand(-1, -1, x.size(2), x.size(3))], 1)
            x = self.deconvolution_block[i](x)
        x = self.output_layer(x)
        return x

class ACGenerator_noconcat(nn.Module):
    def __init__(self, opt, n_filters):
        super(ACGenerator_noconcat, self).__init__()
        
        self.image_size = opt.img_size
        self.init_size = opt.img_size // 2 ** opt.n_layers
        self.n_layers = opt.n_layers
        self.n_filters = n_filters
        self.project = nn.Linear(opt.latent_dim + opt.all_classes_dim, n_filters[0] * (self.init_size) ** 2)
        self.input_layer = nn.Sequential(
            nn.BatchNorm2d(n_filters[0]), 
            nn.ReLU())    
        self.deconvolution_block = nn.ModuleList(\
            [nn.Sequential(
                nn.ConvTranspose2d(n_filters[i], n_filters[i + 1], kernel_size = 4, stride = 2, padding = 1),
                nn.BatchNorm2d(n_filters[i + 1]), nn.ReLU())
            for i in range(opt.n_layers - 1)])
        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(n_filters[-1], opt.channels, kernel_size = 4, stride = 2, padding = 1), 
            nn.Tanh())

    def forward(self, x, condition):
        x = torch.cat([x, condition], 1)
        x = self.project(x).view(-1, self.n_filters[0], self.init_size, self.init_size)
        x = self.input_layer(x)   
        for i in range(self.n_layers - 1):
            x = self.deconvolution_block[i](x)
        x = self.output_layer(x)
        return x



class ACDiscriminator(nn.Module):
    def __init__(self, opt, n_filters):
        super(ACDiscriminator, self).__init__()
 
        self.image_size = opt.img_size
        self.last_size = opt.img_size // 2 ** opt.n_layers
        self.n_layers = opt.n_layers
        self.n_filters = n_filters

        self.input_layer = nn.Sequential(
        	nn.Conv2d(3, n_filters[0], kernel_size = 4, stride = 2, padding = 1), 
        	nn.LeakyReLU(0.2))

        self.convolution_block = nn.ModuleList(\
            [nn.Sequential(
            	nn.Conv2d(n_filters[i], n_filters[i + 1], kernel_size = 4, stride = 2, padding = 1),
            	nn.BatchNorm2d(n_filters[i + 1]), 
                nn.LeakyReLU(0.2))
            for i in range(opt.n_layers - 1)])

        self.score_layer = nn.Linear((self.last_size) ** 2 * n_filters[-1] + opt.all_classes_dim, 1)
        
        self.logit_layer = nn.Sequential(
        	nn.Linear((self.last_size) ** 2 * n_filters[-1], opt.all_classes_dim), 
        	nn.Sigmoid())

    def forward(self, x, condition):
        x = self.input_layer(x)
        for i in range(self.n_layers - 1):
            x = self.convolution_block[i](x)
        x = x.view(-1, (self.last_size) ** 2 * self.n_filters[-1])
        score = self.score_layer(torch.cat([x, condition], 1))
        logit = self.logit_layer(x)
        return score, logit