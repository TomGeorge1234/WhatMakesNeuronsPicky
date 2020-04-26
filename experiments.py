from utils import plot_training, plot_RI, plot_lesion_test, train_multiple
from networks import simple_network, MNIST_network
import numpy as np
np.seterr(all='ignore')

simple_hyperparameters = {'N_train' : 1000, #size of training dataset 
                          'N_test' : 100, #size of test set x
                          'lr' : 0.001, #SGD learning rate 
                          'epochs' : 10, #training epochs
                          'batch_size' : 10,  #batch size (large will probably fail)           
                          'context_location' : 'start',  #where the feed in the task context 'start' vs 'end'
                          'train_mode' : 'random', #training mode 'random' vs 'replay' 
                          'second_task' : 'prod', #first task adds x+y, second task 'prod' = xy or 'add1.5' = x+1.5y
                          'fraction' : 0.50, #fraction of training data for tasks 1 vs task 2
                          'hidden_size' : 100 #hidden layer width 
                          }

models1 = train_multiple(simple_network, simple_hyperparameters, N_models=100) 
plot_training(models1)
plot_RI(models1, show_threshold=False)
plot_lesion_test(models1)

# simple_hyperparameters['second_task'] = 'add1.5'
# models2 = train_multiple(simple_network, simple_hyperparameters, N_models=100) 
# plot_training(models2)
# plot_RI(models2)
# plot_lesion_test(models2)

# simple_hyperparameters['second_task'] = 'prod'
# simple_hyperparameters['hidden_size'] = 5
# models3 = train_multiple(simple_network, simple_hyperparameters, N_models=100) 
# plot_training(models3)
# plot_RI(models3)
# plot_lesion_test(models3)

simple_hyperparameters['hidden_size'] = 100
simple_hyperparameters['context_location'] = 'end'
models4 = train_multiple(simple_network, simple_hyperparameters, N_models=100) 
plot_training(models4)
plot_RI(models4,show_threshold=True)
plot_lesion_test(models4)

simple_hyperparameters['context_location'] = 'start'
simple_hyperparameters['second_task'] = 'add1.5'
simple_hyperparameters['train_mode'] = 'replay'
models5 = train_multiple(simple_network, simple_hyperparameters, N_models=100) 
plot_training(models5)
plot_RI(models5)
plot_lesion_test(models5)

simple_hyperparameters['train_mode'] = 'random'
simple_hyperparameters['second_task'] = 'add1.5'
simple_hyperparameters['fraction'] = 0.2
models6 = train_multiple(simple_network, simple_hyperparameters, N_models=100) 
plot_training(models6)
plot_RI(models6)
plot_lesion_test(models6)

simple_hyperparameters['fraction'] = 0.8
models7 = train_multiple(simple_network, simple_hyperparameters, N_models=100) 
# plot_training(models7)
plot_RI(models7)
# plot_lesion_test(models7)



MNIST_hyperparameters = {'hidden_size' : 100,
                          'epochs' : 5,    
                          'task_sets' :  [[[0,1,2,3,4],[5,6,7,8,9]] , 
                                          [[0,2,4,6,8],[1,3,5,7,9]] ,
                                          [[2,3,5,7],[0,1,4,6,8,9]] ,
                                          [[6,7,2,8,4],[5,0,1,3,9]] ,
                                          [[5,6,7,8,9],[0,1,2,3,4]]
                                         ],
                          'is_CNN' : True,
                          'train_mode' : 'random',
                          'sample_size' : 200,
                          'importance' : 1000,
                          'batch_size' : 64,
                          'lr' : 0.001}   
             
model1 = MNIST_network(MNIST_hyperparameters)
model1.train_model()
model1.plot_training()
model1.plot_RI()

MNIST_hyperparameters['train_mode'] = 'ewc'
model2 = MNIST_network(MNIST_hyperparameters)
model2.train_model()
model2.plot_training()
model2.plot_RI()


