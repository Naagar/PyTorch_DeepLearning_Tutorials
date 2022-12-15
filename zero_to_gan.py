# import torch
# import numpy as np
# import torchvision
# import torch.nn.functional as F
# import torch.nn as nn
# from torchvision.datasets import MNIST
# from torchvision.datasets import CIFAR10
# from torchvision.transforms import ToTensor
# from torch.utils.data.sampler import SubsetRandomSampler
# from torch.utils.data.dataloader import DataLoader

# dataset  = MNIST(root='data/',
#                  download=True,
#                  transform=ToTensor())

# # dataset  = CIFAR10(root='data/',
# #                  download=True,
# #                  transform=ToTensor())


# def split_indices(n, val_pct):
#     # determine size of validation set 
#     n_val = int(val_pct*n)

#     # creat random permutation od 0 to n-1
#     idxs = np.random.permutation(n)

#     # pick first n_val indices for validation set 
#     return idxs[n_val:], idxs[:n_val]



# train_indices, val_indices = split_indices(len(dataset), val_pct=0.2)
# print(len(train_indices), len(val_indices))
# print('sample_val indices:', val_indices[:20])




# batch_size = 500


# # Traning sampler and data loader
# train_sampler = SubsetRandomSampler(train_indices)
# train_d1 = DataLoader(dataset,
#                       batch_size,
#                        sampler = train_sampler)


# #  validation sampler and data loader 
# valid_sampler = SubsetRandomSampler(val_indices)
# valid_d1 = DataLoader(dataset,
#                       batch_size,
#                        sampler = valid_sampler)




# class MnistModel(nn.Module):
#     # feed foward nn with 1 hidden layer
#     def __init__(self, in_size, hidden_size_1, hidden_size_2, out_size):
#         super().__init__()
#         # hidden layer
#         self.linear1 = nn.Linear(in_size, hidden_size_1)

#         self.linear11 = nn.Linear(hidden_size_1, hidden_size_2)

#         # output layer
#         self.linear2 = nn.Linear(hidden_size_2, out_size)

#     def forward(self, xb):
#         # Flaten the imahe tensor
#         xb = xb.view(xb.size(0), -1)

#         # get intermediat outputs using hidden layer
#         out = self.linear1(xb)
#         out = self.linear11(out)

#         # apply activation function   or ignoring the negatives values
#         out = F.relu(out)

#         # Get predctions using the output layer
#         out = self.linear2(out)
#         return out



# for xb, yb in train_d1:
#   xb= xb.view(xb.size(0), -1)
#   print(xb.shape)
#   break




# input_size = 784
# num_classes = 10

# model = MnistModel(input_size, hidden_size_1=128, hidden_size_2=32, out_size=num_classes)




# print(model)


# for t in model.parameters():
#     print(t.shape)


# torch.cuda.is_available()



# print(model.parameters)




# # for images, labels in train_d1:
# #     print(images.shape)
# #     output = model(images)
# #     loss = F.cross_entropy(output, labels)
# #     print('Loss:', loss.item())
# #     break


# torch.cuda.is_available()


# def get_default_device():
#     # pic GPU if avaliable , else cpu
#     if torch.cuda.is_available():
#         return torch.device('cuda')
#     else:
#         return torch.device('cpu') 

# device  = get_default_device()
# device


# def to_device(data, device):
#     # move tensor to chossen device 
#     if isinstance(data, (list,tuple)):
#         return [to_device(x, device) for x in data] 
#     return data.to(device, non_blocking=True)


# for images, labels in train_d1:
#     print(images.shape)
#     images = to_device(images, device)
#     print(images.device)
#     break



# class DeviceDataLoader1():
#   # wrap a data loader to move data to a device
#   def __init__(self, d1, device):
#     self.d1 = d1
#     self.device = device

#   def __iter__(self):
#     # yield a batch of data after moving to device 
#     for b in self.d1:
#       yield to_device(b, self.device)

#   def __len__(self):
#     # no of batches 
#     return len(self.d1)





# train_d1 = DeviceDataLoader1(train_d1, device)

# valid_d1 = DeviceDataLoader1(valid_d1, device)

# for xb, yb in valid_d1:
#   print('xb.device', xb.device)
#   print('yb', yb)
#   break



# def loss_batch(model, loss_fn, xb, yb, opt=None, metric=None):
#     # genertes predctions
#     preds = model(xb)

#     # calculate loss
#     loss = loss_fn(preds, yb)

#     if opt is not None:
#         # compute gradients
#         loss.backward()

#         # update parameters
#         opt.step()

#         # reset gradients
#         opt.zero_grad()

#     metric_result = None

#     if metric is not None:
#         # compute the metric
#         metric_result = metric(preds, yb)
    
#     return loss.item(), len(xb), metric_result

# def evaluate(model, loss_fn, valid_d1, metric=None):
#   with torch.no_grad():
#     # pass each batch through the model 
#     results = [loss_batch(model, loss_fn, xb, yb, metric=metric)
#                 for xb, yb in valid_d1]

#     # seprate losses , counts, metrics
#     losses, nums, metrics = zip(*results)

#     # total size of the dataset
#     total = np.sum(nums)

#     # Avg. loss across batches
#     avg_loss = np.sum(np.multiply(losses, nums)) / total
#     avg_metric = None
#     if metric is not None:
#       # Avg. of metric accross batches
#       avg_metric = np.sum(np.multiply(metrics, nums)) / total

#   return avg_loss, total, avg_metric


# def fit(epochs, lr, model, loss_fn, train_d1, valid_d1, metric=None, opt_fn=None ):
# 	    losses, metrics = [], []
# 	    if opt_fn is None: opt_fn = torch.optim.SGD
# 	    opt = torch.optim.SGD(model.parameters(), lr)

# 	    for epoch in range(epochs):
# 	        # rtraning 
# 	        for xb, yb in train_d1:
# 	            loss, _, _ = loss_batch(model, loss_fn, xb, yb, opt)

# 	        # Evaluation 
# 	        results  = evaluate(model, loss_fn, valid_d1, metric)
# 	        val_loss, total, val_metric = results

# 	        # record the loss and metric
# 	        losses.append(val_loss)
# 	        metrics.append(val_metric)

# 	        # print progress 
# 	        if metric is None:
# 	            print('epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, val_loss))
# 	        else:
# 	          print('epoch [{}/{}], Loss: {:.4f}, {}: {:.4f}'.format(epoch+1, epochs, val_loss, metric.__name__, val_metric))

# 	    return losses, metrics



# def accuracy(outputs, labels):
#     _, preds = torch.max(outputs, dim=1)
#     return torch.sum(preds == labels).item() / len(preds)


# #  model (on GPU)
# model = MnistModel(input_size, hidden_size_1=64, hidden_size_2=32, out_size=num_classes)
# to_device(model, device)




# val_loss, total, val_acc = evaluate(model, F.cross_entropy, 
#                                     valid_d1, metric=accuracy)
# print('Loss: {:.4f}, Accuracy: {:.4f}'.format(val_loss, val_acc))




################
###############   Image classification using CNN (CIFAR10)




import os
import torch
import torchvision
import tarfile
from torchvision.datasets.utils import download_url

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader


from torchvision.utils import make_grid

##  Download dataset 
dataset_url = "http://files.fast.ai/data/cifar10.tgz"
download_url(dataset_url, '.')



# extract from archive
with tarfile.open('./cifar10.tgz', 'r:gz') as tar:
  def is_within_directory(directory, target):
      
      abs_directory = os.path.abspath(directory)
      abs_target = os.path.abspath(target)
  
      prefix = os.path.commonprefix([abs_directory, abs_target])
      
      return prefix == abs_directory
  
  def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
  
      for member in tar.getmembers():
          member_path = os.path.join(path, member.name)
          if not is_within_directory(path, member_path):
              raise Exception("Attempted Path Traversal in Tar File")
  
      tar.extractall(path, members, numeric_owner=numeric_owner) 
      
  
  safe_extract(tar, path="./data")


data_dir = './data/cifar10'
print(os.listdir(data_dir))
classes = os.listdir(data_dir + '/train')
print(classes)



airplane_files = os.listdir(data_dir + "/train/airplane" )
print('No of test exaples for ship:', len(airplane_files))
print(airplane_files[:50])


dataset = ImageFolder(data_dir+'/train', transform=ToTensor())



img, labels = dataset[5999]
print(img.shape, labels)
img

print(dataset.classes)





def show_example(img, labels):
    print('labels:', dataset.classes[labels], '('+str(labels)+')' )
    plt.imshow(img.permute(1, 2, 0))  #  permuting the image index 0 1 2 to 1 2 0


# show_example(*dataset[100])


import numpy as np

def split_indices(n, val_pct=0.1, seed=99):
    # determine size of validation set
    n_val = int(val_pct*n)

    # set the random seed ( for reproducibility)
    np.random.seed(seed)

    #Creat random permutation of 0 to n-1
    idxs = np.random.permutation(n)

    # pick first n_val indices for the validation set 
    return idxs[n_val:], idxs[:n_val]


val_pct = 0.2
rand_seed = 42   # udes for random no generator

train_indices, val_indices = split_indices(len(dataset), val_pct, rand_seed)
print(len(train_indices), len(val_indices))
print('sample validation indices:', val_indices[:9])



batch_size  = 200

# traning data sampler and data loader
train_sampler = SubsetRandomSampler(train_indices)
train_d1 = DataLoader(dataset, 
                      batch_size,
                      sampler = train_sampler)
# validation data sampler and loader 

val_sampler = SubsetRandomSampler(val_indices)
val_d1 = DataLoader(dataset, 
                      batch_size,
                      sampler = val_sampler)



def show_batch(d1):
    for images, labels in d1:
        fig, ax = plt.subplots(figsize=(10,10))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images, 10).permute(1, 2, 0))   ## channels to the end
        break 
# show_batch(train_d1)


## defining the cnn

import torch.nn as nn
import torch.nn.functional as F 


sample_model = nn.Sequential(
    nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(2,2)
)




for images, labels in train_d1:
    print('images_shape:', images.shape)
    out = sample_model(images)
    print('out_shape:', out.shape)
    break
