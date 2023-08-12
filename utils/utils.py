import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def make_folder(path):
        if not os.path.exists(path):
            os.makedirs(path)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)
    
def save_loss(losses, path, name):
    plt.figure(figsize=(10,5))
    plt.title(name + " Loss During Training")
    plt.plot(losses,label=name)
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(path, name))        
    plt.close()
    
def showAnimation(path, interval=200):
    print(path)
    files = sorted([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))], key=lambda f: int(''.join(filter(str.isdigit, f))))
    print(str(len(files)) + ' samples')
    img_list = []
    for file in files:
        img_list.append(plt.imread(os.path.join(path, file)))
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(i, animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=interval, repeat=False)
    return ani

def gmm_sample(num_samples, mix_coeffs, mean, cov):
    z = np.random.multinomial(num_samples, mix_coeffs)
    samples = np.zeros(shape=[num_samples, len(mean[0])])
    i_start = 0
    for i in range(len(mix_coeffs)):
        i_end = i_start + z[i]
        samples[i_start:i_end, :] = np.random.multivariate_normal(
            mean=np.array(mean)[i, :],
            cov=np.diag(np.array(cov)[i, :]),
            size=z[i])
        i_start = i_end    
    return torch.from_numpy(samples).float()

class Configuration(object):
    # The class "Configuration" - It's actually an initializer 
    def __init__(self, initdict):
        self.z_dim = initdict['z_dim']
        self.parallel = initdict['parallel']
        self.total_step = initdict['total_step']
        self.batch_size = initdict['batch_size']
        self.g_lr = initdict['g_lr']
        self.d_lr = initdict['d_lr']
        self.lr_decay = initdict['lr_decay']
        self.beta1 = initdict['beta1']
        self.beta2 = initdict['beta2']
        self.pretrained_model = initdict['pretrained_model']
        self.model_save_path = initdict['model_save_path']
        self.sample_path = initdict['sample_path']
        self.log_step = initdict['log_step']
        self.sample_step = initdict['sample_step']
        self.model_save_step = initdict['model_save_step']
        self.version = initdict['version']
        self.printnet = initdict['printnet']
        
        if 'd_num' in initdict.keys():
            self.d_num = initdict['d_num']
        if 'network' in initdict.keys():
            self.network = initdict['network']
        if 'g_conv_dim' in initdict.keys():
            self.g_conv_dim = initdict['g_conv_dim']
        if 'd_conv_dim' in initdict.keys():
            self.d_conv_dim = initdict['d_conv_dim']
        if 'imsize' in initdict.keys():
            self.imsize = initdict['imsize']
        if 'channel' in initdict.keys():
            self.channel = initdict['channel']
        if 'loss_save_path' in initdict.keys():
            self.loss_save_path = initdict['loss_save_path']
        if 'topo_style' in initdict.keys():
            self.topo_style = initdict['topo_style']        
        if 'h_dim' in initdict.keys():
            self.h_dim = initdict['h_dim']          
        if 'mix_coeffs' in initdict.keys():
            self.mix_coeffs = initdict['mix_coeffs']          
        if 'mean' in initdict.keys():
            self.mean = initdict['mean']          
        if 'cov' in initdict.keys():
            self.cov = initdict['cov']          
        if 'num_samples' in initdict.keys():
            self.num_samples = initdict['num_samples']
        if 'beta' in initdict.keys():
            self.beta = initdict['beta']
        if 'use_tensorboard' in initdict.keys():
            self.use_tensorboard = initdict['use_tensorboard']