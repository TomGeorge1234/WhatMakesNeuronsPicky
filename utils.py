# =============================================================================
# DESCRIPTION: 
#This file contain utility functions for training and analysing the simple_network and MNIST_network model classes in networks.py

#Functions are as follows (described below)
# simple_network() functions:
#  • plot_training()
#  • plot_RI()
#  • plot_lesion_test()
#  • train_and_analyse()
# MNIST_network() functions:
#  • MNIST_subset()
#  • EWC()
# =============================================================================

from copy import deepcopy
import random
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import datasets
from torch.utils.data import Dataset
from tqdm import tqdm
import time 
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt

# =============================================================================
# =============================================================================
# SIMPLE_NETWORK ANALYSIS FUNCTIONS
# =============================================================================
# =============================================================================

# The following three functions all take a list of trained simple_network models and prefrom analysis on them 
# =============================================================================
# plot_training plots the testing abs error throughout training for all models (and shows an average) 
# =============================================================================
def plot_training(models, title=None):  #only works for simple_network lists, not MNIST_networks
    fig, ax = plt.subplots(figsize = (2,1.5))
    task1 = np.zeros(len(np.array(models[0].hist)))
    task2 = np.zeros(len(np.array(models[0].hist)))    
    for i in range(len(models)):
        ax.plot(np.array(models[i].hist)[:,0], linewidth=0.5, alpha=0.2, c='C0')
        task1 += np.array(models[i].hist)[:,0]
        ax.plot(np.array(models[i].hist)[:,1], linewidth=0.5, alpha=0.2, c='C1')
        task2 += np.array(models[i].hist)[:,1]
    task1, task2 = task1/len(models), task2/len(models)
    ax.plot(task1, linewidth=1, c='C0', label = r'Task 1: %s' %models[-1].task1_description )
    ax.plot(task2, linewidth=1, c='C1', label =r'Task 2: %s' %models[-1].task2_description)
    ax.set_ylabel('Absolute test error')
    ax.set_xlabel('Epochs')
    if models[-1].train_mode == 'replay':
        ax.axvspan(0, models[-1].epochs,color='C0', alpha=0.1)  #vertical shading
        ax.axvspan(models[-1].epochs, 2*models[-1].epochs,color='C1', alpha=0.1)  #vertical shading
    ax.legend()
    if title != None:
        fig.suptitle("%s" %title)
    plt.show()
    return 

# =============================================================================
# plot_RI plots the fractional task variance throughout the layers 
# =============================================================================
def plot_RI(models, show_threshold=False, title=None):  #only works for simple_network lists, not MNIST_networks
    RI = [[],[],[],[],[]]
    for model in models:
        for i in range(len(RI)):
            RI[i].extend(list(model.RI[i]))
    fig, axs = plt.subplots(1,4,sharey = True, figsize = (4,0.8))
    for i in range(4):
        n, bins, patches = axs[i].hist(RI[i], weights=np.ones(len(RI[i])) / len(RI[i]),bins=np.linspace(-1,1,11))
        bin_centre = [(bin_right + bin_left)/2 for (bin_right, bin_left) in zip(list(bins[1:]),list(bins[:-1]))]
        col = (bin_centre - min(bin_centre))/(max(bin_centre) - min(bin_centre))
        cm = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap',['C1','C0'], N=1000)
        for c, p in zip(col, patches):
                plt.setp(p, 'facecolor', cm(c))
        plt.gca().yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
        axs[i].set_title("Hidden layer %g" %(i+1))
        axs[i].set_xlim([-1,1])
        axs[i].set_xlabel(r'$\mathcal{RI}$')
        if i == 0:
            axs[i].set_ylabel('Proportion')            
        if i == 3 and show_threshold == True: 
            axs[i].axvline(0.9,color='r',linestyle='--',linewidth=0.8)
            axs[i].axvline(-0.9,color='r',linestyle='--',linewidth=0.8)
    for i in range(4):
        axs[i].text(0.51,axs[i].get_ylim()[-1]*0.92, r"+ %g%%" %int((100*(np.sum(np.isnan(np.array(RI[i])))/len(RI[i])))), fontdict = {'color':'grey', 'fontsize':4})
    if title != None:
        fig.suptitle("%s" %title)
    plt.show()
    return 

# =============================================================================
# plot_lesion_test lesions (sets to zero) penultimate layer neurons with high or low RI then performs a test and plots the performance drop.
# =============================================================================
def plot_lesion_test(models, RI_threshold=0.9, title=None): #only works for simple_network lists, not MNIST_networks
    
    derr_h1_large, derr_h1_small, derr_h2_large, derr_h2_small = [], [], [], []
    
    for model in models:
        initial_error = model.abs_error()
        h1 = model.forward(model.x_task1_test, mode='other')[-2]
        h2 = model.forward(model.x_task2_test, mode='other')[-2]
        
        h1_large = h1 * torch.Tensor(np.less(np.array(model.RI[-2]),RI_threshold))
        h1_small = h1 * torch.Tensor(np.greater(np.array(model.RI[-2]),-RI_threshold))
        h2_large = h2 * torch.Tensor(np.less(np.array(model.RI[-2]),RI_threshold))
        h2_small = h2 * torch.Tensor(np.greater(np.array(model.RI[-2]),-RI_threshold))
        
        derr_h1_large.append(((nn.functional.relu(model.fc5(h1_large)) - model.y_task1_test).abs().mean(dim=0).item() - initial_error[0])/initial_error[0])
        derr_h1_small.append(((nn.functional.relu(model.fc5(h1_small)) - model.y_task1_test).abs().mean(dim=0).item() - initial_error[0])/initial_error[0])
        derr_h2_large.append(((nn.functional.relu(model.fc5(h2_large)) - model.y_task2_test).abs().mean(dim=0).item() - initial_error[1])/initial_error[1])
        derr_h2_small.append(((nn.functional.relu(model.fc5(h2_small)) - model.y_task2_test).abs().mean(dim=0).item() - initial_error[1])/initial_error[1])

    fig, axs = plt.subplots(1, 2, figsize = (1.5,0.8), sharey = True)
    axs[0].bar([-0.5,0.5],[100*np.mean(np.array(derr_h1_small)),100*np.mean(np.array(derr_h2_small))],width=1,color=['C0','C1'])
    axs[1].bar([-0.5,0.5],[100*np.mean(np.array(derr_h1_large)),100*np.mean(np.array(derr_h2_large))],width=1, color=['C0','C1'])
    axs[0].set_xticks([-0.5,0.5])
    axs[0].set_xticklabels([r'Task 1:''\n'r'%s' %(models[-1].task1_description),r'Task 2:''\n'r'%s' %(models[-1].task2_description)])
    axs[1].set_xticks([-0.5,0.5])
    axs[1].set_xticklabels([r'Task 1:''\n'r'%s' %(models[-1].task1_description),r'Task 2:''\n'r'%s' %(models[-1].task2_description)])
    axs[0].set_title(r'Lesion $\mathcal{RI}$ < %.2f' %-RI_threshold)
    axs[1].set_title('Lesion $\mathcal{RI}$ > %.2f' %RI_threshold)
    axs[0].set_ylabel(r'$\Delta$ Error (%)')
    if title != None:
        fig.suptitle("%s" %title)
    plt.show()
    return 
    

# =============================================================================
# train_multiple trains N_models simple_networks and returns the trained networkas a list
# a network is only accepted as 'trained' if the abs_error on BOTH tasks is < 0.05 after training 
# if a network fails to train 10 times in a row the training is aborted and an error returned
# =============================================================================
def train_multiple(model_class, hyperparameters = None,  N_models=20, ):
    models = []    
    print("Training %g models" %N_models); time.sleep(.5)
    for _ in tqdm(range(N_models)):
        fail_count = 0
        current_model_successful = False
        while current_model_successful == False:
            if fail_count >= 10:
                print("\n This model doesn't train well, aborting")
                return models
            model = model_class(hyperparameters)
            model.train_model()
            if model.abs_error()[0]<0.05 and model.abs_error()[1]<0.05:
                model.get_RI()
                models.append(model)
                current_model_successful = True
            else:
                fail_count += 1
    return models





# =============================================================================
# =============================================================================
# MNIST_NETWORK ANALYSIS FUNCTIONS
# =============================================================================
# =============================================================================

# =============================================================================
# MNIST_subset defines a pytorch dataset containing subsets of the MNIST dataset labelled as defined in task_sets

#__init__(task_sets) initialises the dataset             
    #task_sets defines all the tasks. Here's an example of how we'd write 3 seperate tasks which would all train on the same model:
    #task 1: digits 0 and 7 --> 0, digits 1 and 8 --> 1
    #task 2: digits 0 and 8 --> 0, digits 1 and 7 --> 1
    #task 3: digits 0, 1, 2, 3 and 4 --> 0, digits 5,6,7,8 and 9 --> 1
    #would give: 
    #task_sets = [  [[0,7],[1,8]]  ,  [[0,8],[1,7]]  ,  [[0,1,2,3,4],[5,6,7,8,9]] ]
    # note all tasks must have the same number of output classes (here 2 but could be more, or len(task_sets[0]))

# __getitem__() returns a flattened mnist image to which is appended the task context.
# i.e. the first 784 digits are the image, the remainder is the context label (a one-hot vector).
# =============================================================================
class MNIST_subset(Dataset):
    def __init__(self, task_sets = [[0,2,4,6,8],[1,3,5,7,9]], train=True, task_id=0, task_count=1):
        global trainset, testset

        try: 
            trainset
        except NameError:
            print("Downloading MNIST datasets")
            trainset = datasets.MNIST('./data', train=True, download=True)
            testset = datasets.MNIST('./data', train=False, download=True)
        else: 
            pass 
        
        if train == True:
            dataset = trainset
        elif train == False:
            dataset = testset
        for i in range(len(task_sets)):
            for j in range(len(task_sets[i])):
                if i == 0 and j == 0: 
                    self.data = dataset.data[dataset.targets==task_sets[i][j]]
                    self.targets = i*torch.ones(sum(dataset.targets==task_sets[i][j])).byte()
                else:
                    self.data = torch.cat((self.data,dataset.data[dataset.targets==task_sets[i][j]]))
                    self.targets = torch.cat((self.targets,i*torch.ones(sum(dataset.targets==task_sets[i][j])).byte()))
                
        idx = torch.randperm(len(self.targets)) 
        self.data = self.data[idx].float()/255
        self.targets = self.targets[idx].long() 
        
        #add in context:
        context = torch.zeros((len(self.data),task_count))
        context[:,task_id] = torch.ones(len(self.data))
        context = context.float()
        
        self.data = torch.cat((self.data.reshape(len(self.data),-1),context),dim=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        return img, target
    
    def get_sample(self, sample_size):
        sample_idx = random.sample(range(len(self)), sample_size)
        return [img for img in self.data[sample_idx]]



# =============================================================================
# EWC is used to compute the EWC penaly for elastic weights training. 

#__init__ takes the current model and stores a deepcopy of it's state when initialised.
# This is done so we know how far the weights have drifted from thiswhen calculating the penalty
#it also takes a dataset of previous task examples in order to calculate the fischer information matrix
# =============================================================================
class EWC(object):
    def __init__(self, model: nn.Module, dataset: list):

        self.model = model
        self.dataset = dataset

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = Variable(p.data)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = Variable(p.data)

        self.model.eval()
        for input in self.dataset:
            self.model.zero_grad()
            input = Variable(input)
            input = input.view(-1,input.shape[0])
            output = self.model.forward(input).view(1, -1)
            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss



