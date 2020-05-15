# =============================================================================
# DESCRIPTION:
# This file  contains two pytorch neural network classes. 
# •simple_network is a basic FCNN performing variants of a task where a two dimensional input is mapped to a single output (y = x1 + x2)
# •MNIST_network is a more complex network performing variants of the MNIST classification task 

# USE: 
# •Initialise oth networks can as x_network(hyperparameters) where hyperparameters is a python dictionary (see bwlow for exactly what is required)
#    hyperparameter contains everything you can, and would want to, vary for your network 
# •Train both networks by trained by calling x_network.train_model()
# •Analyse an MNIST_network by calling MNIST_network.plot_training(), MNIST_network.plot_RI() 
# •Analysing simple_network models is done slightly differently. For this multiple models must be trained and 
#    stored in a list models = [model1, model2, ...] which can be passed to plot_training(models), plot_RI(models) and plot_lesion_test(models) in utils.py
# =============================================================================

import numpy as np
import matplotlib 
from matplotlib import rcParams
import matplotlib.pyplot as plt
from cycler import cycler 
plt.style.use("seaborn")
rcParams['figure.dpi']= 300
rcParams['axes.labelsize']=5
rcParams['axes.labelpad']=2
rcParams['axes.titlepad']=3
rcParams['axes.titlesize']=5
rcParams['axes.xmargin']=0
rcParams['axes.ymargin']=0
rcParams['xtick.labelsize']=4
rcParams['ytick.labelsize']=4
rcParams['grid.linewidth']=0.5
rcParams['legend.fontsize']=4
rcParams['lines.linewidth']=1
rcParams['xtick.major.pad']=2
rcParams['xtick.minor.pad']=2
rcParams['ytick.major.pad']=2
rcParams['ytick.minor.pad']=2
rcParams['xtick.color']='grey'
rcParams['ytick.color']='grey'
rcParams['figure.titlesize']='medium'
rcParams['axes.prop_cycle']=cycler('color', ['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854','#ffd92f','#e5c494','#b3b3b3'])
import random
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch import optim
from tqdm.autonotebook import tqdm
import time
from torch.utils.data import DataLoader, ConcatDataset

from utils import EWC, MNIST_subset 




class simple_network(nn.Module):
    def __init__(self, hyperparameters=None):
        if hyperparameters == None:
            hyperparameters = self.get_default_hyperparameters()
        self.hidden_size = hyperparameters['hidden_size']
        self.context_location = hyperparameters['context_location']
        if self.context_location == 'start':
            super(simple_network, self).__init__()
            self.fc1 = nn.Linear(4, self.hidden_size)
            self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
            self.fc3 = nn.Linear(self.hidden_size, self.hidden_size)
            self.fc4 = nn.Linear(self.hidden_size, self.hidden_size)
            self.fc5 = nn.Linear(self.hidden_size, 1)
        # if context_loaction = 'end' then the context vector is only passed into the penultimate hidden layer 
        elif self.context_location == 'end':
            super(simple_network, self).__init__()
            self.fc1 = nn.Linear(2, self.hidden_size)
            self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
            self.fc3 = nn.Linear(self.hidden_size, self.hidden_size)
            self.fc4 = nn.Linear(self.hidden_size+2, self.hidden_size)
            self.fc5 = nn.Linear(self.hidden_size, 1)
            
        #initialise hyperparameter
        self.N_train = hyperparameters['N_train']
        self.N_test = hyperparameters['N_test']
        self.epochs = hyperparameters['epochs']
        self.lr = hyperparameters['lr']
        self.batch_size = hyperparameters['batch_size']
        self.train_mode = hyperparameters['train_mode']
        self.second_task = hyperparameters['second_task']
        self.fraction = hyperparameters['fraction']
        #initialises some attributes
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.type_of_network = 'simple_network'
        #arrays for later analysis
        self.hist = []
        self.RI = [[],[],[],[],[]]
        self.Itask1 = [[],[],[],[],[]]
        self.Itask2 = [[],[],[],[],[]]
        #training and testing data 
        self.set_data()
        self.initialise_biases()
        self.hist.append(self.abs_error())  
        
        self.task1_description = '$x_0 + x_1$'
        if self.second_task == 'prod':
            self.task2_description = '$x_0 \cdot x_1$'
        elif self.second_task == 'add1.5':
            self.task2_description = '$x_0 + 1.5 x_1$'
            

            

    #a forward pass of the network. If mode is 'normal' only the output is returned,
    #otherwise all hidden layers are returned. Context location determines whether 
    # task context is passed in at the first or penultimate layer
    # at the first layer, if mode is '
    def forward(self, input, mode='normal'):
        if self.context_location == 'start':
            x = nn.functional.relu(self.fc1(input))        
            x1 = nn.functional.relu(self.fc2(x))        
            x2 = nn.functional.relu(self.fc3(x1))
            x3 = nn.functional.relu(self.fc4(x2))
            x4 = nn.functional.relu(self.fc5(x3))
            if mode == 'normal':
                return x4
            elif mode == 'other':
                return x, x1, x2, x3, x4
        elif self.context_location == 'end':
            x = nn.functional.relu(self.fc1(input[:,:2]))        
            x1 = nn.functional.relu(self.fc2(x))        
            x2 = nn.functional.relu(self.fc3(x1))
            x2_ = torch.cat((x2,input[:,2:]),axis=1)
            x3 = nn.functional.relu(self.fc4(x2_))
            x4 = nn.functional.relu(self.fc5(x3))
            if mode == 'normal':
                return x4
            elif mode == 'other':
                return x, x1, x2, x3, x4
    
    #MAKE TRAINING DATA
    def set_data(self):
        self.N_task1 = int(self.fraction*self.N_train)
        self.N_task2 = self.N_train - self.N_task1
        
        self.x_task1_train = np.concatenate((np.random.uniform(0,1,(self.N_task1,2)),np.ones((self.N_task1,1)),np.zeros((self.N_task1,1))),axis=1)
        self.x_task2_train = np.concatenate((np.random.uniform(0,1,(self.N_task2,2)),np.zeros((self.N_task2,1)),np.ones((self.N_task2,1))),axis=1)
        self.x_task1_test = np.concatenate((np.random.uniform(0,1,(self.N_test,2)),np.ones((self.N_test,1)),np.zeros((self.N_test,1))),axis=1)
        self.x_task2_test = np.concatenate((np.random.uniform(0,1,(self.N_test,2)),np.zeros((self.N_test,1)),np.ones((self.N_test,1))),axis=1)

        self.y_task1_train = np.sum(self.x_task1_train[:,:2],axis=1)
        self.y_task1_test = np.sum(self.x_task1_test[:,:2],axis=1)
        if self.second_task == 'prod':
            self.y_task2_train = np.prod(self.x_task2_train[:,:2],axis=1)
            self.y_task2_test = np.prod(self.x_task2_test[:,:2],axis=1)
        elif self.second_task == 'add1.5':
            self.y_task2_train = np.add(self.x_task2_train[:,0],1.5*self.x_task2_train[:,1])
            self.y_task2_test = np.add(self.x_task2_test[:,0],1.5*self.x_task2_test[:,1])
        
        self.x_train = np.concatenate((self.x_task1_train,self.x_task2_train))
        self.y_train = np.concatenate((self.y_task1_train,self.y_task2_train))
        
        self.x_train = torch.from_numpy(self.x_train).float()
        self.x_task1_train = torch.from_numpy(self.x_task1_train).float()
        self.x_task2_train = torch.from_numpy(self.x_task2_train).float()
        self.x_task1_test = torch.from_numpy(self.x_task1_test).float()
        self.x_task2_test = torch.from_numpy(self.x_task2_test).float()
        
        self.y_train = torch.from_numpy(self.y_train).float().unsqueeze(1)
        self.y_task1_train = torch.from_numpy(self.y_task1_train).float().unsqueeze(1)
        self.y_task2_train = torch.from_numpy(self.y_task2_train).float().unsqueeze(1)
        self.y_task1_test = torch.from_numpy(self.y_task1_test).float().unsqueeze(1)
        self.y_task2_test = torch.from_numpy(self.y_task2_test).float().unsqueeze(1)

    def abs_error(self):
        task1_error = (self.forward(self.x_task1_test) - self.y_task1_test).abs().mean(dim=0).item()
        task2_error = (self.forward(self.x_task2_test) - self.y_task2_test).abs().mean(dim=0).item()
        return [task1_error, task2_error]

    def train_model(self):
        if self.train_mode == 'random':
            for epoch in range(self.epochs):
                for i in range(int(self.N_train/self.batch_size)): #input.shape == [2]
                    idx = np.random.choice(self.N_train,self.batch_size,replace=False)
                    self.do_train_step(idx)
                self.eval() #test/evaluation model 
                with torch.no_grad():
                    self.hist.append(self.abs_error())     

        if self.train_mode == 'replay':
            N_task1 = int(self.N_train*self.fraction)
            for epoch in range(self.epochs):
                for i in range(int(N_task1/self.batch_size)): 
                    idx = np.random.choice(N_task1,self.batch_size,replace=False)
                    self.do_train_step(idx)
                self.eval() #test/evaluation model 
                with torch.no_grad():
                    self.hist.append(self.abs_error())     
            for epoch in range(self.epochs):
                for i in range(int((self.N_train-N_task1)/self.batch_size)): #input.shape == [2]
                    if (i+1)%10 == 0: 
                        idx = np.random.choice(N_task1,self.batch_size,replace=False)
                    else:    
                        idx = np.random.choice(range(N_task1,self.N_train),self.batch_size,replace=False)
                    self.do_train_step(idx)
                self.eval() #test/evaluation model 
                with torch.no_grad():
                 self.hist.append(self.abs_error())                      
                 
    def do_train_step(self, idx):
        sample=self.x_train[idx] 
        self.train() # we move the model to train regime because some models have different train/test behavior, e.g., dropout.
        self.optimizer.zero_grad()
        output = self.forward(sample)
        loss = F.mse_loss(output,self.y_train[idx])
        loss.backward()
        self.optimizer.step()
        
    def get_RI(self):
        self.RI = [[],[],[],[],[]]
        self.Itask1 = [[],[],[],[],[]]
        self.Itask2 = [[],[],[],[],[]]

        hidden_task1 = self.forward(self.x_task1_test, mode='other')        
        hidden_task2 = self.forward(self.x_task2_test, mode='other') 
        
        error_task1 = F.mse_loss(self.y_task1_test, hidden_task1[-1])
        error_task2 = F.mse_loss(self.y_task2_test, hidden_task2[-1])
        
        for i in range(len(hidden_task1)):
            hidden_task1[i].retain_grad()
            hidden_task2[i].retain_grad()
            
        error_task1.backward()
        error_task2.backward()    

        for i in range(len(hidden_task1)):
            Itask1 = (((hidden_task1[i] * hidden_task1[i].grad)**2).mean(0)).detach().numpy()
            Itask2 = (((hidden_task2[i] * hidden_task2[i].grad)**2).mean(0)).detach().numpy()
            Itask1[Itask1 < np.mean(Itask1)/10]=0
            Itask2[Itask2 < np.mean(Itask2)/10]=0
            RI_ = (Itask1 - Itask2) / (Itask1 + Itask2)
            self.Itask1[i].extend(list(Itask1)) 
            self.Itask2[i].extend(list(Itask2))
            self.RI[i].extend(list(RI_))
            
    def initialise_biases(self): #probably unneccesary, set biases initially to zero
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.zeros_(self.fc2.bias)
        torch.nn.init.zeros_(self.fc3.bias)
        torch.nn.init.zeros_(self.fc4.bias)
        torch.nn.init.zeros_(self.fc5.bias)     
    
    def get_default_hyperparameters(self):
        hps = {'N_train' : 1000, #size of training dataset 
               'N_test' : 100, #size of test set x
               'lr' : 0.001, #SGD learning rate 
               'epochs' : 10, #training epochs
               'batch_size' : 10,  #batch size (large will probably fail)           
               'context_location' : 'start',  #where the feed in the task context 'start' vs 'end'
               'train_mode' : 'random', #training mode 'random' vs 'replay' 
               'second_task' : 'prod', #first task adds x+y, second task 'prod' = xy or 'add1.5' = x+1.5y
               'fraction' : 0.50, #fraction of training data for tasks 1 vs task 2
               'hidden_size' : 100} #hidden layer width 
        return hps





# =============================================================================
# Deep Neural Network Class for learning multiple tasks. 
# USE: >>> model = MNIST_network()
#     >>> model.train_model()
#     >>> model.plot_training()
#     >>> model.RI_plot()

# DESCRIPTION: This class is initiated with 'task_sets' (specifying what permutations of the MNIST task to learn simultaneously)
# Important attribute functions include:
    #forward() - a forward pass of the network. Can return just the output or the penultimate layer as well
    #set_data() - takes the task_sets and creates pytorch dataloaders from which to train/test
    #train_model() - trains the model 
    #test() - tests on a test dataset and returns the accuracy
    #training_plot() - plots how loss and accuracy changes throughout training
    #get_RI() - calculates I and RI across all tasks and task pairs  
    #RI_plot() - plots a matrix of histograms of the fractional tast variance (aka. relative importance) for penultimate layer neurons in all pairs or  tasks
# =============================================================================
class MNIST_network(nn.Module):    
    #initialise the class, collate the train/test data and build the deep model 
    def __init__(self, hyperparameters):      
        # print("Initialising model")
        # print("Loading and partitioning MNIST data")
        
        self.task_sets = hyperparameters['task_sets']
        self.hidden_size = hyperparameters['hidden_size']
        self.is_CNN = hyperparameters['is_CNN']
        self.epochs = hyperparameters['epochs']
        self.train_mode = hyperparameters['train_mode']
        self.sample_size = hyperparameters['sample_size']
        self.importance = hyperparameters['importance']
        self.batch_size = hyperparameters['batch_size']
        self.lr = hyperparameters['lr']
        
        self.task_count = len(self.task_sets)
        self.num_classes = len(self.task_sets[0])
        self.loss, self.acc, self.ewc = {}, {}, {} 
        self.set_data()  
        self.type_of_network = 'MNIST_network'


        if self.is_CNN == True: #this can be a CNN...
            super(MNIST_network, self).__init__()
            self.conv1 = nn.Conv2d(1,4,kernel_size=3)
            self.conv2 = nn.Conv2d(4,12,kernel_size=3)
            self.fc1 = nn.Linear(5*5*12 + self.task_count, self.hidden_size)
            self.fc2 = nn.Linear(self.hidden_size, self.num_classes)
        elif self.is_CNN == False: #...or just a fully connected NN
            super(MNIST_network, self).__init__()
            self.fc1 = nn.Linear(28 * 28 + self.task_count, self.hidden_size)
            self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
            self.fc3 = nn.Linear(self.hidden_size, self.num_classes)
        
        self.optimizer = optim.Adam(params=self.parameters(), lr=self.lr)

     
    #forward pass
    def forward(self, input, mode = 'normal'):
        if self.is_CNN == False:
            image = input[:,:784]
            context = input[:,784:]
            x = torch.cat((image,context),1)
            x1 = F.relu(self.fc1(x))
            x2 = F.relu(self.fc2(x1))
            x3 = F.softmax(self.fc3(x2),dim=1)
            if mode == 'normal':
                return x3
            elif mode == 'other':
                return x2, x3
        
        elif self.is_CNN == True:
            image = input[:,:784].view(-1,1,28,28)
            context = input[:,784:]
            x1 = F.max_pool2d(F.relu(self.conv1(image)),kernel_size=2)
            x2 = F.max_pool2d(F.relu(self.conv2(x1)),kernel_size=2)
            x3 = x2.view(x2.shape[0],-1)
            x3 = torch.cat((x3,context),1)
            x4 = F.relu(self.fc1(x3))
            x5 = F.softmax(self.fc2(x4),dim=1)
            if mode == 'normal':
                return x5
            elif mode == 'other':
                return x4, x5
            
    #task_sets defines all the tasks. Here's an example of how we'd write 3 seperate tasks which would all train on the same model:
    #task 1: digits 0 and 7 --> 0, digits 1 and 8 --> 1
    #task 2: digits 0 and 8 --> 0, digits 1 and 7 --> 1
    #task 3: digits 0, 1, 2, 3 and 4 --> 0, digits 5,6,7,8 and 9 --> 1
    #would give: 
    #task_sets = [  [[0,7],[1,8]]  ,  [[0,8],[1,7]]  ,  [[0,1,2,3,4],[5,6,7,8,9]] ]
    # note all tasks must have the same number of output classes (here 2 but could be more, or len(task_sets[0]))
    def set_data(self):
        self.train_datasets = []
        self.test_datasets = []
        for i in range(self.task_count):
            self.train_datasets.append(MNIST_subset(task_sets=self.task_sets[i],train=True,task_id=i, task_count=self.task_count))
            self.test_datasets.append(MNIST_subset(task_sets=self.task_sets[i],train=False,task_id=i, task_count=self.task_count))

        self.train_loaders = [DataLoader(dataset, batch_size=self.batch_size, shuffle=True) for dataset in self.train_datasets]
        self.test_loaders = [DataLoader(dataset, batch_size=self.batch_size, shuffle=True) for dataset in self.test_datasets]

        self.train_dataset = ConcatDataset(self.train_datasets) #a concatenated version used for when we train randomly on all tasks
        self.train_loader = DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=True)

    #train the model (notice different regimes depending on whether we are training sequentially, randomly or using elastic weights consolidation )
    def train_model(self):
        
        if self.train_mode == 'random':
            self.total_epochs = self.epochs
        else:
            self.total_epochs = self.epochs*self.task_count
            
        for task in range(self.task_count):
                self.acc[task] = []
                self.acc[task].append(self.test(self.test_loaders[task]))

        if self.train_mode == 'sequential':
            for task in tqdm(range(self.task_count),desc='Task',leave=False):
                self.loss[task] = []
                self.loss[task].append(self.do_train_epoch(self.train_loaders[task],do_train=False))
                for _ in tqdm(range(self.epochs),desc='Epoch',leave=False):
                    self.loss[task].append(self.do_train_epoch(self.train_loaders[task]))
                    for sub_task in range(task + 1):
                        self.acc[sub_task].append(self.test(self.test_loaders[sub_task]))

        elif self.train_mode == 'random':
            self.loss[0] = []
            self.loss[0].append(self.do_train_epoch(self.train_loader,do_train=False))
            for _ in tqdm(range(self.epochs),desc='Epoch',leave=False):
                self.loss[0].append(self.do_train_epoch(self.train_loader))
                for sub_task in range(self.task_count):
                    self.acc[sub_task].append(self.test(self.test_loaders[sub_task]))
    
        elif self.train_mode == 'ewc':
            for task in tqdm(range(self.task_count),desc='Task',leave=False):
                self.loss[task] = []
                if task == 0:
                    self.loss[task].append(self.do_train_epoch(self.train_loaders[task],do_train=False))
                    for _ in tqdm(range(self.epochs),desc='Epoch',leave=False):
                        self.loss[task].append(self.do_train_epoch(self.train_loaders[task]))
                        self.acc[task].append(self.test(self.test_loaders[task]))
                else:
                    old_tasks = []
                    for sub_task in range(task):
                        old_tasks = old_tasks + self.train_loaders[sub_task].dataset.get_sample(self.sample_size)
                    old_tasks = random.sample(old_tasks, k=self.sample_size)
                    self.loss[task].append(self.do_train_epoch(self.train_loaders[task],do_train=False))
                    EWC_now = EWC(self, old_tasks)
                    for _ in tqdm(range(self.epochs),desc='Epoch',leave=False):
                        self.loss[task].append(self.do_train_epoch(self.train_loaders[task], ewc=EWC_now, importance=self.importance))
                        for sub_task in range(task + 1):
                            self.acc[sub_task].append(self.test(self.test_loaders[sub_task]))
            
        self.get_RI()
    #function to do one epoch of training
    def do_train_epoch(self, data_loader, ewc=None, importance=0, do_train=True):
        self.train()
        epoch_loss = 0
        for input, target in data_loader:
            input, target = Variable(input), Variable(target)
            self.optimizer.zero_grad()
            output = self.forward(input)
            if self.train_mode == 'ewc' and importance > 0:
                loss = F.cross_entropy(output, target) + importance * ewc.penalty(self)
            else:
                loss = F.cross_entropy(output, target)                  
            epoch_loss += loss.item()
            if do_train == True: 
                loss.backward()
                self.optimizer.step()
        #print ratio of ewc loss to real loss for debugging
        # if self.train_mode == 'ewc' and importance > 0:
        #     print("%.2f" %((F.cross_entropy(output, target) / (importance*ewc.penalty(self))).item()))
        #     print(importance)
        return epoch_loss / len(data_loader)
    
    #test a dataloader and return accuracy
    def test(self, data_loader):       
        self.eval()
        correct = 0
        for input, target in data_loader:
            input, target = Variable(input), Variable(target)
            output = self.forward(input)
            correct += (F.softmax(output, dim=1).max(dim=1)[1] == target).data.sum()
        accuracy = correct.item() / len(data_loader.dataset)
        return accuracy
    
    #plot graphs for loss and accuracy throughout training
    def plot_training(self):
        fig, axs = plt.subplots(1,2,figsize=(4,1.5))
        if self.train_mode != 'random':
            for t, v in self.loss.items():
                axs[0].plot(list(range(t * self.epochs, (t + 1) * self.epochs + 1)), v, label="Task %g" %(t+1))
        else:
            for t, v in self.loss.items():
                axs[0].plot(list(range(t * self.epochs, (t + 1) * self.epochs + 1)), v, label="All tasks", color='C6')
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Cross entropy loss")
        axs[0].legend(loc = 1)
        
        for t, v in self.acc.items():
            if self.train_mode == 'random':
                axs[1].plot(list(range(self.total_epochs + 1)), v, label="Task %g" %(t+1))
            else:
                axs[1].plot(list(range(t * self.epochs, self.total_epochs + 1)), v, label="Task %g" %(t+1))
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Test accuracy")
        axs[1].set_ylim(0,1)
        axs[1].legend(loc=4)
        
        if self.train_mode != 'random':
            for i in range(self.task_count):
                axs[0].axvspan(i*self.epochs, (i+1)*self.epochs,color='C%g' %i, alpha=0.1)  #vertical shading
                axs[1].axvspan(i*self.epochs, (i+1)*self.epochs,color='C%g' %i, alpha=0.1)  #vertical shading
        plt.show()
   
    #calculates the relative importance for the penultimate hidden layer 
    def get_RI(self):        
        self.I = {}
        for task in range(len(self.task_sets)):
            importance = None
            for input, target in self.test_loaders[task]:
                input, target = Variable(input, requires_grad = True), Variable(target)
                hidden = self.forward(input,mode='other')
                loss = F.cross_entropy(hidden[-1], target)
                for j in range(len(hidden)):
                    hidden[j].retain_grad()
                loss.backward()
                if importance is None: 
                    importance = (hidden[-2].grad * hidden[-2])**2
                else:
                    importance = torch.cat((importance,(hidden[-2].grad * hidden[-2])**2),0)
            I = (importance.mean(dim=0)).detach().numpy()
            I[I < np.mean(I)/10] = 0
            self.I[task] = I
        
        if self.task_count == 1:
            print("No RI to plot: only 1 task")
            return 
        
        else:
            self.RI = {}
            for i in range(self.task_count):
                for j in range(self.task_count):
                    self.RI[i,j] = (self.I[i] - self.I[j]) / (self.I[i] + self.I[j])

        
    #plot relative importance matrix over task pairs 
    def plot_RI(self): #gets importances for penultimate layer                     
        fig, axs = plt.subplots(self.task_count-1, self.task_count-1, figsize=(1*(self.task_count-1),1*(self.task_count-1)), sharex=True, sharey=True)
        # fig.suptitle("RI histograms across task pairs")
        for i in range(self.task_count-1):
            for j in range(i):
                axs[i][j].axis("off")
            for j in range(i,self.task_count-1):
                n, bins, patches = axs[i][j].hist(self.RI[i,j+1],  weights=np.ones(len(self.RI[i,j+1])) / len(self.RI[i,j+1]), bins=np.linspace(-1,1,11))
                bin_centres = [(bin_right + bin_left)/2 for (bin_right, bin_left) in zip(list(bins[1:]),list(bins[:-1]))]
                col = [(bin_centre + 1) / 2 for bin_centre in bin_centres]
                cm = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap',['C%g' %(j+1),'C%g' %i], N=1000)
                for c, p in zip(col, patches):
                        plt.setp(p, 'facecolor', cm(c))
                plt.gca().yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
                colors = {}
                for color in range(self.task_count):
                    colors[color] = matplotlib.colors.to_rgba_array('C%g' %color)
                    colors[color][0][-1] = 0.3
                if i == 0:
                    # axs[i][j].set_title("Task %g" %(j+2), backgroundcolor=colors[j+1][0])
                    axs[i][j].set_title("Task %g" %(j+2), color='C%g' %(j+1))
                    if j == 0:
                        axs[i][j].set_xlim([-1,1])
                        axs[i][j].set_xticks([-1,0,1])
                        axs[i][j].set_xticklabels(['-1','0','1'])
                if j == i:
                    # axs[i][j].set_ylabel("Task %g" %(j+1), backgroundcolor=colors[j][0])
                    axs[i][j].set_ylabel("Task %g" %(j+1), color='C%g' %(j))
        for i in range(self.task_count-1):
            for j in range(i,self.task_count-1):
                    axs[i][j].text(0.5,axs[i][j].get_ylim()[-1]*0.9, r"+ %g%%" %int((100*(np.sum(np.isnan(np.array(self.RI[i,j+1])))/len(self.RI[i,j+1])))),fontsize=4, color='grey')
        
        return