import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from torchvision.transforms import Compose
from torch import nn, optim
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
import vcf_data_loader
import csv

#Individual Child Block Class
class autoencoder_model_1 (pl.LightningModule):
  def __init__(self, input_dim, n_hidden, lr = 1e-3, combine = False):
    super().__init__()
    self.encoder = nn.Sequential(nn.Linear(input_dim, n_hidden), nn.ReLU())
    self.decoder = nn.Sequential(nn.Linear(n_hidden, input_dim), nn.ReLU(), 
                                 nn.Linear(input_dim, 3*input_dim), 
                                 ReshapeLogSoftmax(n_snps = input_dim, combine = combine))
    self.double()
    self.save_hyperparameters()
    self.learning_rate = lr
    self.weights = None

  def forward (self, features):
    reconstruction = self.encoder(features)
    reconstruction = self.decoder(reconstruction)
    return reconstruction
  
  def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x = batch
        x = x[0].view(x[0].size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.nll_loss(x_hat, x.to(int), weight = self.weights)
        #loss = F.mse_loss(x_hat, x)
        # Logging to TensorBoard by default
        self.log("train_loss", loss, on_epoch = True)
        wandb.log({ "loss": loss})
        return loss
  
  def configure_optimizers(self):   
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

class ReshapeLogSoftmax(nn.Module):
    def __init__(self, n_snps, combine = False):
        super().__init__()
        self.combine = combine
        self.n_snps = n_snps
        
    def forward(self, x):
        #combining two overlapping blocks (ignore since not used anymore)
        if self.combine: 
          first_block = x[:,0:3000].view(-1, 3, 1000)
          last_block = x[:,5601000:].view(-1, 3, 1000)
          middle = x[:,3000:5601000]
          softmax = F.log_softmax(first, dim=1)
          for i in range(3000,5601000,6000):
            x1 = x[:,i:i+3000].view(-1,3,1000)
            x2 = x[:,i+3000:i+6000].view(-1,3,1000)
            x12 = torch.concat((x1,x2), dim = 1)
            combined_softmax = F.log_softmax(x12, dim = 1)
            combined_softmax.reshape(-1,2,3,1000).sum(axis=1)
            softmax.concat((softmax,combined_softmax), dim = 1)
          last = F.log_softmax(last, dim=1)
          softmax = torch.concat ((softmax,last), dim = 1)
          return softmax
        x = x.view(-1, 3, self.n_snps)
        return F.log_softmax(x, dim=1)

#Joint Parent Model Class
class ParentModel(pl.LightningModule):
    def __init__(self, modelA, modelB, modelC):
        super(ParentModel, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.double()
        self.weights = None
        
    def forward(self, x1, x2):
        x1 = self.modelA(x1)
        x2 = self.modelB(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.modelC(x)
        return x
    def training_step(self, batch, batch_idx):
      # training_step defined the train loop.
      # It is independent of forward
      x = batch
      x1 = x[0].view(x[0].size(0), -1)
      x2 = x[1].view(x[1].size(0),-1)
      z1 = self.modelA(x1)
      z2 = self.modelB(x2)
      z = torch.cat((z1, z2), dim=1)
      x_hat = self.modelC(z)
      loss = F.nll_loss(x_hat, torch.concat((x1,x2),dim=1).to(int), weight = self.weights)
      accuracy = (torch.concat((x1,x2),dim=1) == x_hat.argmax(dim=1)).to(float).mean(dim=1).mean()
      #loss = F.mse_loss(x_hat, x)
      # Logging to TensorBoard by default
      self.log("train_loss", loss, on_epoch = True)
      self.log("accuracy",accuracy, on_epoch = True)
      wandb.log({"loss": loss,"accuracy":accuracy})
      return loss
    def configure_optimizers(self):   
      return torch.optim.Adam(self.parameters(), lr=1e-3)

#Function to add the softmax of two overlapping blocks together
def concat_softmax (first_block, second_block):
  x12 = torch.concat((first_block,second_block), dim = 1)
  x12.reshape(-1,2,3,1000).sum(axis=1)
  return x12

#Load dataset and file to write to 
vcf = vcf_data_loader.FixedSizeVCFChunks("all_1kg_chr1_phased_GRCh38_snps_maf0.01.recode.vcf.gz")
accuracies = []
error = []
f = open('./model_predictions.csv', 'w')

writer = csv.writer(f)

for i in range(0,934):
  #Create two child models (a and b) and a parent model (c), pass them as input into a ParentModel network
  modela = autoencoder_model_1(1000, 16)
  modelb = autoencoder_model_1(1000, 16)
  modelc = autoencoder_model_1(2000, 32)
  parent = ParentModel(modela.encoder, modelb.encoder, modelc.decoder)
  #Load the weights from a pretrained savve parent model 
  parent.load_state_dict(torch.load(f"./parent-blocks/model-{i}-{i+1}.pth"))
  parent.eval()
  # Find the chunk # (dataset is seperated into chunks of 5000 SNPs) and get the tensor for that chunk
  chunks = i//5
  data = vcf.get_tensor_for_chunk_id(chunks)
  # Find the index # within the chunk of dataset (indices are multiples of 1000)
  ind = i%5
  #Create a tensor of a subset of 1000 SNPs within the chunk
  dataset = torch.tensor(data[:,(ind)*1000:(ind+1)*1000],dtype=torch.float64)
  #Edge case when the subset is the last 1000 SNPs in a chunk
  if ind == 4:
    data = vcf.get_tensor_for_chunk_id(chunks+1)
    ind = -1
  # Create a tensor of the next adjacent subset of 1000 SNPs in the chunk
  dataset2 = torch.tensor(data[:,(ind+1)*1000:(ind+2)*1000],dtype=torch.float64)
  # Pass tensors to parent model to get output 
  predictions = parent(dataset, dataset2)
  # Write the output to the file
  writer.writerow(predictions.detach().numpy().tolist())
  
print("LOAD MODELS COMPLETE")
f.close()



