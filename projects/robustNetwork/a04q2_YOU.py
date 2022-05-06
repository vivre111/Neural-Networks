#!/usr/bin/env python
# coding: utf-8

# # A04-Q2: Combatting Overfitting

# ## Preliminaries

# In[1]:


# Standard imports
import numpy as np
import torch
import matplotlib.pylab as plt
import copy


# In[ ]:




torch.nn.Dropout
# # Dataset

# In[2]:

class DividedPlane(torch.utils.data.Dataset):
    def __init__(self, n=100, noise=0.1, seed=None):
        a = torch.tensor([-0.4, 0.5, 0.15]) #torch.rand((3,))
        def myfunc(x):
            y = a[0]*x[:,0] + a[1]*x[:,1] + a[2]
            return y
        self.x = torch.rand((n,2))*2. - 1.
        y = myfunc(self.x) + noise*torch.normal( torch.zeros((len(self.x))) )
        self.y = (y>0.).type(torch.float)
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    def inputs(self):
        return self.x
    
    def targets(self):
        return self.y.reshape( (len(self.y),1) )
    
    def plot(self, labels=None, *args, **kwargs): 
        X = self.inputs()
        if labels is None:
            labels = self.targets()
        colour_options = ['y', 'r', 'g', 'b', 'k']
        if len(labels[0])>1:
            # one-hot labels
            cidx = torch.argmax(labels, axis=1)
        else:
            # binary labels
            cidx = (labels>0.5).type(torch.int)
        colours = [colour_options[k] for k in cidx]
        plt.scatter(X[:,0].detach(), X[:,1].detach(), color=colours, marker='.')


# In[3]:


train = DividedPlane(n=100, noise=0.2, seed=165)
test = DividedPlane(n=5000)
plt.figure(figsize=(9,4))
plt.subplot(1,2,1); train.plot(); plt.title(f'Training Set');
plt.subplot(1,2,2); test.plot(); plt.title(f'Test Set');


# # A: `Dropout` layer

# In[ ]:


class Dropout(torch.nn.Module):
    '''
     lyr = Dropout()
     
     Creates a dropout layer in which each node is set to zero
     with probability lyr.dropprob.
     
     Usage:
       lyr = Dropout()
       lyr.set_dropprob(p) # set the dropout probability to p
       y = lyr(z)          # sets each node to 0 with probability p
       
     The input, z, contains one sample per row. So the dropout is
     performed independently on each row of z.
    '''
    def __init__(self):
        super().__init__()
        self.dropprob = 0.
        
    def set_dropprob(self, p):
        self.dropprob = p
        
    def forward(self, z):
        # Drop nodes with prob dropprob
        
        #===== YOUR CODE HERE =====
        y = z  # replace this line
        
        return y


# In[ ]:


# Test for Dropout layer
z = torch.ones((3,1000))
drop_layer = Dropout()
drop_layer.set_dropprob(0.75)
y = drop_layer(z)
drop_fraction = (torch.sum(y==0.)*100.)/torch.numel(y)
print(f'Dropped {drop_fraction:.1f}%')
print(f'Expected output is {torch.sum(y)}, which should be close to {torch.sum(z)}')


# # B: `RobustNetwork`
# * Implement regularization by weight decay <br>
# * Integrate the dropout layer into learning

# In[ ]:


class RobustNetwork(torch.nn.Module):
    def __init__(self, nodes=100):
        super().__init__()
        self.lyrs = torch.nn.ModuleList()
        self.lyrs.append(torch.nn.Linear(2, nodes))
        self.lyrs.append(torch.nn.ReLU())
        self.lyrs.append(torch.nn.Linear(nodes, nodes))
        self.lyrs.append(torch.nn.Sigmoid())
        self.drop_lyr = Dropout()    # <-- Create Dropout layer
        self.lyrs.append(self.drop_lyr)  # Add it to the list
        self.lyrs.append(torch.nn.Linear(nodes, 1))
        self.lyrs.append(torch.nn.Sigmoid())
        self.loss_fcn = torch.nn.BCELoss(reduction='mean')
    
    def forward(self, x):
        y = x
        for lyr in self.lyrs:
            y = lyr(y)
        return y

    
    def learn(self, x, t, epochs=100, lr=0.1, weight_decay=0., dropprob=0.):
        losses = []
        for epoch in range(epochs):
            y = self(x)
            
            loss = self.loss_fcn(y.squeeze(), t.squeeze())
            
            losses.append(loss.item())
            self.zero_grad()
            loss.backward()
            with torch.no_grad():
                for p in self.parameters():
                    p -= lr*p.grad
        plt.plot(np.array(losses))
        plt.yscale('log'); plt.xlabel('Epochs'); plt.ylabel('Log Loss');
        print(f'Final loss = {loss}')
        return losses


# # Train and test

# In[ ]:





# In[ ]:


sum(y>0)


# In[ ]:


net_orig = RobustNetwork(nodes=250)

# Duplicate the network for apples-to-apples comparison
net = copy.deepcopy(net_orig)
rnet = copy.deepcopy(net_orig)
dnet = copy.deepcopy(net_orig)

# Set come common parameters
lr = 0.5
n_epochs = 5000


# ### Saving and loading models
# You might find it helpful to save and load your networks. The lines below save the network, including the connection weights and biases.
# 
# Note that the pertinent classes have to be declared before you can load an object of that class.

# In[ ]:


#torch.save(net, 'simple_net.pt')
#net = torch.load('simple_net.pt')


# ### Train the models

# In[ ]:


# No effort to guard against overfitting
losses = net.learn(train.inputs(), train.targets(), epochs=n_epochs, lr=lr)


# In[ ]:


# L2 regularization
rlosses = rnet.learn(train.inputs(), train.targets(), epochs=n_epochs, lr=lr, weight_decay=0.004)


# In[ ]:


# Dropout
dlosses = dnet.learn(train.inputs(), train.targets(), epochs=n_epochs, lr=lr, dropprob=0.9)


# ### Test the models
# #### Let's see what the decision boundaries look like.

# In[ ]:


test.inputs().shape


# In[ ]:


# Compute test loss
y = net(test.inputs()); test_loss = net.loss_fcn(y, test.targets())
ry = rnet(test.inputs()); rtest_loss = rnet.loss_fcn(ry, test.targets())
dy = dnet(test.inputs()); dtest_loss = dnet.loss_fcn(dy, test.targets())

# Display the results
plt.figure(figsize=(15,4))
plt.subplot(1,3,1)
test.plot(labels=net(test.inputs())); plt.title(f'Orig Test Loss = {test_loss:.3f}')
plt.subplot(1,3,2)
test.plot(labels=rnet(test.inputs())); plt.title(f'Weight Decay Test Loss = {rtest_loss:.3f}')
plt.subplot(1,3,3)
test.plot(labels=dnet(test.inputs())); plt.title(f'Dropout Test Loss = {dtest_loss:.3f}');


# In[ ]:





# In[ ]:




