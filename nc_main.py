import numpy as np
from numpy.random import default_rng
rng = default_rng() # set RNG
max_sample = 1e10  # set max number of pixels
np.random.seed(0)   # set seed

def sampler2D(image1):
  size = 1
  for s in image1.shape: size *= s

  if size > max_sample:
    max = int(np.floor(np.sqrt(max_sample / image1.shape[1])))
    pixels_x = rng.choice(image1.shape[2], size=max, replace=False)
    pixels_y = rng.choice(image1.shape[3], size=max, replace=False)
    return pixels_x, pixels_y
  else:
    pixels_x = np.arange(image1.shape[2])
    pixels_y = np.arange(image1.shape[3])
    return pixels_x, pixels_y

def sampler1D(image1):
  if image1.shape[0] * image1.shape[1] > max_sample:
    pixels = rng.choice(image1.shape[1], size=max_sample, replace=False)
    return pixels
  else:
    pixels = np.arange(image1.shape[1])
    return pixels

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
import gc
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.models as models
from tqdm import tqdm
from collections import OrderedDict
from scipy.sparse.linalg import svds
from torchvision import datasets, transforms
import pickle
import importlib
from nc_models import get_model
import nc_models


#===============================================================================================
################################################################################################
path = '/scratch/gpfs/eonal/0NCResearch/experiments/' # Save folder
# also need a 'data' folder in current directory


debug = False # Only runs 20 batches per epoch for debugging
train_only = False # Avoids running any analysis

model_name = 'MLP6'
dataset = 'CIFAR100' # 'CIFAR10' / 'CIFAR100' / 'MNIST' / 'FashionMNIST' / 'SVHN' / 'TinyImageNet'
loss_name = 'MSE' # 'CE' / 'MSE'
activation = 'ReLU' # 'ReLU' / 'LeakyReLU' / 'Tanh' / 'Sigmoid' / 'SiLU',

epochs              = 3
epoch_list          = [1, 150] # forced epochs for analysis
acc_targets = [0.90, 0.95, 0.975, 0.99] # target accuracies for analysis

batch_size          = 16
analysis_batch_size = 1
momentum            = 0.9
weight_decay        = 0
################################################################################################
#===============================================================================================
settings = '-'.join([model_name, dataset, loss_name, activation])
print('Will save to path', path + settings)

assert dataset in our_models.official_names['datasets']
assert model_name in our_models.official_names['models']
assert loss_name in our_models.official_names['losses']
assert activation in our_models.official_names['activations']
print(f' Model {model_name}\n Dataset {dataset}\n Loss {loss_name}\n Activation {activation}')

if dataset in ['CIFAR10', 'CIFAR100']:
    mean = (0.491, 0.482, 0.447)
    std = (0.247, 0.243, 0.262)
    im_size = 32
    padded_im_size = im_size + 4
    input_ch = 3
    transform = transforms.Compose([transforms.Pad((padded_im_size - im_size) // 2),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])

    if dataset == 'CIFAR10':
        C = 10
        train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True)
        analysis_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=True, download=True, transform=transform), batch_size=analysis_batch_size, shuffle=True)
    else:
        C = 100
        train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('data', train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True)
        analysis_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('data', train=True, download=True, transform=transform), batch_size=analysis_batch_size, shuffle=True)

elif dataset == 'SVHN':
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    im_size             = 32
    padded_im_size      = im_size + 4
    C                   = 10
    input_ch            = 3
    transform = transforms.Compose([transforms.Pad((padded_im_size - im_size) // 2),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])
    train_loader = torch.utils.data.DataLoader(
        datasets.SVHN('data', split='train', download=True, transform=transform),
        batch_size=batch_size, shuffle=True)
    analysis_loader = torch.utils.data.DataLoader(
        datasets.SVHN('data', split='train', download=True, transform=transform),
        batch_size=analysis_batch_size, shuffle=True)
elif dataset == 'MNIST':
    mean = 0.1307
    std = 0.3081
    im_size = 28
    padded_im_size = im_size + 4
    C = 10
    input_ch = 1
    transform = transforms.Compose([transforms.Pad((padded_im_size - im_size) // 2),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True)
    analysis_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transform),
        batch_size=analysis_batch_size, shuffle=True)
elif dataset == 'FashionMNIST':
    mean = 0.2859
    std = 0.3530
    im_size = 28
    padded_im_size = im_size + 4
    C = 10
    input_ch = 1
    transform = transforms.Compose([transforms.Pad((padded_im_size - im_size) // 2),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True)
    analysis_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('data', train=True, download=True, transform=transform),
        batch_size=analysis_batch_size, shuffle=True)
elif dataset == 'TinyImageNet':
    mean= (0.485, 0.456, 0.406)
    std= (0.229, 0.224, 0.225)
    im_size             = 64
    padded_im_size      = im_size + 4
    C                   = 200
    input_ch            = 3

    transform = transforms.Compose([transforms.Pad((padded_im_size - im_size) // 2),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])
    TRAIN_DIR = 'data/tiny-imagenet-200/train'
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    analysis_loader = torch.utils.data.DataLoader(train_dataset, batch_size=analysis_batch_size, shuffle=True)


# Optimization Criterion
if loss_name == 'CE':
  lr = 0.1                  # Max LR for OneCycleLR
  criterion = nn.CrossEntropyLoss()
  criterion_summed = nn.CrossEntropyLoss(reduction='sum')
elif loss_name == 'MSE':
  lr = 0.1                 # Max LR for OneCycleLR
  criterion = nn.MSELoss()
  criterion_summed = nn.MSELoss(reduction='sum')

model = get_model(model_name, num_classes = C, activation=activation).to(device)
optimizer = optim.SGD(model.parameters(),lr=lr,momentum=momentum,weight_decay=weight_decay)
lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer,max_lr=lr,epochs=epochs,steps_per_epoch=len(train_loader), div_factor=10, final_div_factor=50000.0)


def train(model, criterion, device, num_classes, train_loader, optimizer, lr_scheduler, epoch):
    model.train()

    pbar = tqdm(total=len(train_loader), position=0, leave=True)

    glob_accuracy = 0
    glob_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader, start=1):
        if data.shape[0] != batch_size:
            continue

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        out = model(data)
        if str(criterion) == 'CrossEntropyLoss()':
          loss = criterion(out, target)
        elif str(criterion) == 'MSELoss()':
          loss = criterion(out, F.one_hot(target, num_classes=num_classes).float())

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        accuracy = torch.mean((torch.argmax(out,dim=1)==target).float()).item()
        glob_accuracy += accuracy
        glob_loss += loss.item()
        if debug and batch_idx > 20:
          break

    glob_accuracy /= (len(train_loader)-1)
    glob_loss /= (len(train_loader)-1)
    pbar.update(1)
    pbar.set_description(
            'Train\t\tEpoch: {} [{}/{} ({:.0f}%)] \t'
            'Batch Loss: {:.6f} \t'
            'Batch Accuracy: {:.6f}'.format(
                epoch,
                batch_idx,
                len(train_loader),
                100. * batch_idx / len(train_loader),
                glob_loss,
                glob_accuracy))
    pbar.close()
    return glob_accuracy


def modded_analysis(graphs, model, criterion_summed, device, num_classes, loader, features, value):
    model.eval()

    N             = [0 for _ in range(C)]
    mean          = [0 for _ in range(C)]
    Sw            = 0

    loss          = 0
    net_correct   = 0
    NCC_match_net = 0

    # choose pixel values
    for batch_idx, (data, target) in enumerate(loader, start=1):
      data, target = data.to(device), target.to(device)
      #output = model(data).detach()
      output = model(data)
      #h = features[value].data.detach()
      h = features[value].data
      
      if len(h.shape) == 4:
        pixels_x, pixels_y = sampler2D(h)

      if len(h.shape) == 2:
        pixels = sampler1D(h)

      break

    # start computation
    for computation in ['Mean','Cov']:
        pbar = tqdm(total=len(loader), position=0, leave=True)
        for batch_idx, (data, target) in enumerate(loader, start=1):

            data, target = data.to(device), target.to(device)

            #output = model(data).detach()
            output = model(data)
            
            # grabbing activations of the hooked layer
            #h = features[value].data.view(data.shape[0],-1).detach()

            #h = features[value].data.detach()
            h = features[value].data
            if len(h.shape) == 4:
              h = h[:, :, pixels_x, pixels_y]
              h = torch.flatten(h, start_dim = 1)
            else:
              h = h[:, pixels]

            # during calculation of class means, calculate loss
            if computation == 'Mean':
                if str(criterion_summed) == 'CrossEntropyLoss()':
                  loss += criterion_summed(output, target).item()
                elif str(criterion_summed) == 'MSELoss()':
                  loss += criterion_summed(output, F.one_hot(target, num_classes=num_classes).float()).item()

            for c in range(C):
                # indices of activations belonging to class c
                idxs = (target == c).nonzero(as_tuple=True)[0]

                if len(idxs) == 0: # If no class-c in this batch
                  continue

                h_c = h[idxs,:]

                if computation == 'Mean':
                    # update class means
                    mean[c] += torch.sum(h_c, dim=0) # CHW
                    N[c] += h_c.shape[0]

                elif computation == 'Cov':
                    # update within-class cov

                    z = h_c - mean[c].unsqueeze(0) # B CHW
                    cov = torch.matmul(z.unsqueeze(-1), # B CHW 1
                                       z.unsqueeze(1))  # B 1 CHW
                    Sw += torch.sum(cov, dim=0)

                    # during calculation of within-class covariance, calculate:
                    # 1) network's accuracy
                    net_pred = torch.argmax(output[idxs,:], dim=1)
                    net_correct += sum(net_pred==target[idxs]).item()

                    # 2) agreement between prediction and nearest class center
                    NCC_scores = torch.stack([torch.norm(h_c[i,:] - M.T,dim=1) \
                                              for i in range(h_c.shape[0])])
                    NCC_pred = torch.argmin(NCC_scores, dim=1)
                    NCC_match_net += sum(NCC_pred==net_pred).item()

            pbar.update(1)
            pbar.set_description(
                'Analysis {}\t'
                'Epoch: {} [{}/{} ({:.0f}%)]'.format(
                    computation,
                    epoch,
                    batch_idx,
                    len(loader),
                    100. * batch_idx/ len(loader)))

            if debug and batch_idx > 20:
                break
        pbar.close()

        if computation == 'Mean':
            for c in range(C):
                mean[c] /= N[c]
                M = torch.stack(mean).T
            loss /= sum(N)
        elif computation == 'Cov':
            Sw /= sum(N)

    graphs.loss.append(loss)
    graphs.accuracy.append(net_correct/sum(N))
    graphs.NCC_mismatch.append(1-NCC_match_net/sum(N))

    # global mean
    muG = torch.mean(M, dim=1, keepdim=True) # CHW 1

    # between-class covariance
    M_ = M - muG
    Sb = torch.matmul(M_, M_.T) / C

    # avg norm
    #W  = classifier.weight
    M_norms = torch.norm(M_,  dim=0)
    #W_norms = torch.norm(W.T, dim=0)

    graphs.norm_M_CoV.append((torch.std(M_norms)/torch.mean(M_norms)).item())
    #graphs.norm_W_CoV.append((torch.std(W_norms)/torch.mean(W_norms)).item())

    # tr{Sw Sb^-1}
    #Sw = Sw.numpy()
    #Sb = Sb.numpy()
    #eigvec, eigval, _ = svds(Sb, k=C-1)
    print('Starting svd')
    start = time.time()
    eigvec, eigval, _ = torch.linalg.svd(Sb, full_matrices=False)
    eigvec = eigvec[..., :C-1]
    eigval = eigval[:C-1]
    print(f'SVD complete. Took {time.time()-start} seconds')
    
    for i in range(len(eigval)):
      if eigval[i] == 0:
        eigval[i] += 1e-2
    inv_Sb = eigvec @ torch.diag(eigval**(-1)) @ eigvec.T
    graphs.Sw_invSb.append(torch.trace(Sw @ inv_Sb).item())

    def coherence(V):
        G = V.T @ V
        G += torch.ones((C,C),device=device) / (C-1)
        G -= torch.diag(torch.diag(G))
        return torch.norm(G,1).item() / (C*(C-1))

    graphs.cos_M.append(coherence(M_/M_norms))

# gets all children
def get_children(model: torch.nn.Module):
    children = list(model.children())
    flat_children = []
    if children == []:
        return model
    else:
       for child in children:
            try:
                flat_children.extend(get_children(child))
            except TypeError:
                flat_children.append(get_children(child))
    return flat_children


features = {}
l_hooked = []


if model_name == 'Resnet18':
  all_layers = [layer for layer in get_children(model) if (isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear))]
  layers = [layer for layer in all_layers if (isinstance(layer, nn.Linear) or (isinstance(layer, nn.Conv2d) and layer.kernel_size[0] != 1))]
  layer_names = [f'conv{i+1}' if isinstance(l, nn.Conv2d) else 'fc' for i, l in enumerate(layers)]
  l_names = layer_names[1:]
  layers = layers[1:]
  assert len(l_names) == len(layers)
  values = l_names

  for i, l_name in enumerate(l_names):
    def hook(self, input, output, l_name=l_name):
      features[l_name] = input[0].clone()

    layers[i].register_forward_hook(hook)
    l_hooked.append(l_name)

elif model_name == 'Densenet121':
    blocks = [block for block in list(model.children())[0] if isinstance(block, models.densenet._DenseBlock)]
    layers = []
    for b in blocks:
      layers += list(b.children())
    print(f'{len(layers)} dense layers total, selecting half')
    layers = layers[::2] # only take half of densenet dense blocks
    layers.append(model.classifier)
    layer_names = [f'dense{i+1}' if isinstance(l, models.densenet._DenseLayer) else 'fc' for i, l in enumerate(layers)]
    values = layer_names

    for i, name in enumerate(layer_names):
      def hook(self, input, output, name = name):
        features[name] = input[0][0].clone()

      layers[i].register_forward_hook(hook)
      l_hooked.append(name)

else:
  l_names = [l[0] for l in model.named_children()]
  l_names = l_names[1:] # ignoring first layer
  values = l_names

  # adding hooks to each layer
  for l_name, l in model.named_children():
      if l_name in l_names:
          def hook(self, input, output, l_name=l_name):
            #features[l_name] = torch.flatten(input[0].clone())
            features[l_name] = input[0].clone()

          l.register_forward_hook(hook)
          l_hooked.append(l_name)
      else:
          print(f'Not hooking layer {l_name}.')

print('Hooked layers ' + ', '.join(l_hooked) + '.')



class graphs:
  def __init__(self):
    self.accuracy     = []
    self.loss         = []
    self.reg_loss     = []
    self.epoch        = []

    # NC1
    self.Sw_invSb     = []

    # NC2
    self.norm_M_CoV   = []
    self.norm_W_CoV   = []
    self.cos_M        = []
    self.cos_W        = []

    # NC3
    self.W_M_dist     = []

    # NC4
    self.NCC_mismatch = []

    # Decomposition
    self.MSE_wd_features = []
    self.LNC1 = []
    self.LNC23 = []
    self.Lperp = []


graphs_list = [graphs() for _ in values]
cur_epochs = []
acc_attained = -1
glob_accuracy = 0
for epoch in range(1, epochs + 1):
    if epoch != 1:
      glob_accuracy = train(model, criterion, device, C, train_loader, optimizer, lr_scheduler, epoch)

    if train_only != True:
      acc_analysis = False
      if not acc_attained == len(acc_targets) - 1 and glob_accuracy >= acc_targets[acc_attained + 1]:  # if not all attained already and new target attained
          acc_attained += 1  # increment attained target
          acc_analysis = True

      if epoch in epoch_list or acc_analysis:
          cur_epochs.append(epoch)

          for i, value in enumerate(values):
              graphs_list[i].epoch.append(epoch)
              modded_analysis(graphs_list[i], model, criterion_summed, device, C, analysis_loader, features, value)

          plt.figure(1)
          plt.semilogy(cur_epochs, graphs_list[0].loss)
          plt.xlabel('Epoch')
          plt.ylabel('Value')
          plt.title('Training Loss over Epochs')

          plt.figure(2)
          plt.plot(cur_epochs, 100*(1 - np.array(graphs_list[0].accuracy)))
          plt.xlabel('Epoch')
          plt.ylabel('Training Error (%)')
          plt.title('Training Error over Epochs')

          plt.figure(3)
          for i, value in enumerate(values):
            plt.semilogy(cur_epochs, graphs_list[i].Sw_invSb, label=value)
          #for graph in graphs_list:
          #    plt.semilogy(cur_epochs, graph.Sw_invSb)
          plt.xlabel('Epoch')
          plt.ylabel('Tr{Sw Sb^-1}')
          plt.title('Activation Collapse')
          plt.legend()

          plt.figure(4)
          for i, value in enumerate(values):
              plt.plot(cur_epochs, graphs_list[i].norm_M_CoV, label = value)
          #plt.plot(cur_epochs, graphs.norm_W_CoV)
          #plt.legend(['Class Means','Classifiers'])
          plt.xlabel('Epoch')
          plt.ylabel('Std/Avg of Norms')
          plt.title('Convergence to Equinorm')
          plt.legend()

          plt.figure(5)
          for i, value in enumerate(values):
              plt.plot(cur_epochs, graphs_list[i].cos_M, label = value)
          #plt.plot(cur_epochs, graphs.cos_W)
          #plt.legend(['Class Means','Classifiers'])
          plt.xlabel('Epoch')
          plt.ylabel('Avg|Cos + 1/(C-1)|')
          plt.title('Emergence of Maximal Equiangularity')
          plt.legend()

          plt.figure(7)
          for i, value in enumerate(values):
              plt.plot(cur_epochs, graphs_list[i].NCC_mismatch, label = value)
          plt.xlabel('Epoch')
          plt.ylabel('Mismatch from NCC')
          plt.title('Convergence to NCC')

          plt.legend()
          plt.show()


# define dictionary
dict = {'graphs':graphs_list}

nc_filename = '-'.join([model_name, dataset, loss_name, activation]) + '-nc'
model_filename = '-'.join([model_name, dataset, loss_name, activation]) + '-model'

if not train_only:
    with open(path + nc_filename + '.pkl', "wb") as f:
        pickle.dump(dict, f)