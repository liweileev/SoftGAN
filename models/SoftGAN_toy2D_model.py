import os
import time
import torch
import datetime
import math
import numpy as np
from tqdm import tnrange

import torch.nn as nn
from torchsummary import summary
from sklearn.metrics import pairwise_distances as pdist

from utils.utils import *
import matplotlib.pyplot as plt
from networks.toy2D_network import Generator, Discriminator

class Trainer(object):
    def __init__(self, config):
        
        # Model hyper-parameters
        self.z_dim = config.z_dim
        self.h_dim = config.h_dim
        self.beta = config.beta
        self.mix_coeffs = config.mix_coeffs
        self.mean = config.mean
        self.cov = config.cov
        self.num_samples = config.num_samples
        self.parallel = config.parallel
        
        self.total_step = config.total_step
        self.batch_size = config.batch_size
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.lr_decay = config.lr_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model
        
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        self.loss_save_path = config.loss_save_path
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.version = config.version
        self.printnet = config.printnet

        self.build_model()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

    def train(self):
        
        names = self.__dict__

        # Fixed input for debugging
        fixed_z = torch.randn(self.batch_size, self.z_dim).cuda()

        # Start with trained model
        if self.pretrained_model:
            start = self.pretrained_model + 1
        else:
            start = 0
            
        # Storage losses for plot
        if self.pretrained_model:
            G_losses = np.load(os.path.join(self.loss_save_path, 'G_losses.npy')).tolist()
            D_losses = np.load(os.path.join(self.loss_save_path, 'D_losses.npy')).tolist()
        else:
            G_losses = []
            D_losses = []

        # Start time
        start_time = time.time()
        
        for step in tnrange(start, self.total_step):
            
            self.D.train()
            self.G.train()

            real_data = gmm_sample(self.batch_size, self.mix_coeffs, self.mean, self.cov)
            
            real_data = real_data.cuda()
            z = torch.randn(self.batch_size, self.z_dim).cuda()       

            # ================== Train D ================== #
            d_loss_real = 0
            d_loss_fake = 0
            
            with torch.no_grad():
                fake_data = self.G(z)

            # D_real = torch.sigmoid(self.D(real_data))
            D_real = self.D(real_data)
            entropy_real = self.entropy(D_real)
            d_loss_real = torch.mean(D_real + self.beta * entropy_real)

            # D_fake = torch.sigmoid(self.D(fake_data))
            D_fake = self.D(fake_data)
            entropy_fake = self.entropy(D_fake)
            d_loss_fake = torch.mean(-D_fake + self.beta * entropy_fake)

            # Backward + Optimize
            d_loss = - (d_loss_real + d_loss_fake)
            D_losses.append(d_loss.item())
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()
            
            # ================== Train G  ================== #
            fake_data = self.G(z)
            g_loss = torch.mean(-self.D(fake_data))
            
            # Backward + Optimize
            self.reset_grad()
            G_losses.append(g_loss.item())
            g_loss.backward()
            self.g_optimizer.step()
            
            # ================== log and save  ================== #
            # Print out log info
            if (step + 1) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print("Elapsed [{}], step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}".format(elapsed, step + 1, self.total_step, d_loss.item(), g_loss.item()))

            # Sample images
            if step==0 or (step + 1) % self.sample_step == 0:
                fake_data = self.G(fixed_z)
                self.save_image(step)
                save_loss(G_losses, self.loss_save_path, "G")
                save_loss(D_losses, self.loss_save_path, "D")
            
            # save models
            if (step+1) % self.model_save_step==0:
                torch.save(self.D.state_dict(), os.path.join(self.model_save_path, 'D_step{}.pth'.format(step+1)))
                torch.save(self.G.state_dict(), os.path.join(self.model_save_path, 'G_step{}.pth'.format(step+1)))
                np.save(os.path.join(self.loss_save_path, 'G_losses'), G_losses)
                np.save(os.path.join(self.loss_save_path, 'D_losses'), D_losses)

    def build_model(self):
        names = self.__dict__
        
        self.D = Discriminator(self.h_dim).cuda()
        self.G = Generator(self.z_dim, self.h_dim).cuda()
        print("Initialization parameters of Generator & Discriminator successfully.")
            
        if self.parallel:
            self.D = nn.DataParallel(self.D)
            self.G = nn.DataParallel(self.G)
            print("Parallel computing started.")

       # Loss and optimizer
        self.d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), self.d_lr, [self.beta1, self.beta2])
        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr, [self.beta1, self.beta2])
        print("Initialization optimizers of Generator & Discriminator successfully.")
            
        if self.printnet:
            print("\n=============================\nG summary:")
            summary(self.G, (self.z_dim, 1))
            print("D summary:")
            summary(self.D, (2, 1))
            self.G.cuda()
            self.D.cuda()
            print("\n=============================\n")

    def load_pretrained_model(self):
        names = self.__dict__
        self.D.load_state_dict(torch.load(os.path.join(self.model_save_path, 'D_step{}.pth'.format(self.pretrained_model))))
        self.G.load_state_dict(torch.load(os.path.join(self.model_save_path, 'G_step{}.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def reset_grad(self):
        names = self.__dict__
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()
    
    def save_image(self, step):
        names = self.__dict__
        real_data_num_samples = gmm_sample(self.num_samples, self.mix_coeffs, self.mean, self.cov)
        z = torch.randn(self.num_samples, self.z_dim).cuda()            
        fake_data_num_samples = self.G(z).detach()
        
        # plot & save
        plt.figure()
        plt.scatter(real_data_num_samples[:, 0], real_data_num_samples[:, 1], marker='+', c='r', label='real data')
        plt.scatter(fake_data_num_samples[:, 0], fake_data_num_samples[:, 1], marker='o', c='b', label='generated data')
        plt.legend(loc=[1.05, 0])
        plt.savefig(os.path.join(self.sample_path, '{}.png'.format(step + 1)), bbox_inches='tight')
        plt.close()

    def entropy(self, intensor):
        EPS = 1e-8
        pos = intensor.clamp(min=EPS, max=1)
        neg = 1 - intensor
        neg.clamp_(min=EPS, max=1)
        
        return -(pos * pos.log() + neg * neg.log())