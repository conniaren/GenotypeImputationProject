import scipy
from scipy import sparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from torchvision.transforms import Compose
from torch import nn, optim
import wandb
import pytorch_lightning as pl
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

#Dataset preprocessing/Baselines 

### Hardy-Weinberg Baseline ###
dataset = sparse.load_npz("output_matrix.npz").todense()
hw_dataset = np.mean(dataset,axis = 0)/2
pp = (np.square(hw_dataset)).T
q = [1-x for x in hw_dataset]
pq = 2*np.multiply(hw_dataset,q[0]).T
qq = (np.square(q[0])).T
hw_dataset = np.concatenate([qq,pq,pp], axis = 1)

hw_variants = np.argmax(hw_dataset, axis= 1)
baseline_hw_dataset = np.repeat(hw_variants.T, 503, axis = 0)


### Imperical Mean Baseline ###
imperical_mean = np.mean(dataset, axis = 0)
mean_dataset = np.repeat (imperical_mean, 503, axis = 0)

### Standardizing Dataset ###
def standardize (dataset):
  standard = StandardScaler()
  return standard.fit_transform(dataset)

standardized_dataset = standardize(dataset)


class autoencoder_model_1(pl.LightningModule):
    def __init__(self, input_dim, n_hidden=256, lr=1e-3):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, n_hidden), nn.ReLU(), nn.Linear(n_hidden,128),nn.ReLU(), nn.Linear(128,64), nn.ReLU(), nn.Linear(64,32))
        self.decoder = nn.Sequential(nn.Linear(32,64), nn.ReLU(),nn.Linear(64,128), nn.ReLU(), nn.Linear(128,n_hidden), nn.ReLU(), nn.Linear(n_hidden,input_dim))
        self.double()
        self.save_hyperparameters()

    def forward(self, features):
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
        loss = F.mse_loss(x_hat, x)
        # Logging to TensorBoard by default
        self.log("train_loss", loss, on_epoch=True)
        wandb.log({"loss": loss})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])

#Configurations
k_fold = 5
epochs = 15
results = {}

#Set seed number
torch.manual_seed(333)

#K-fold cross validator
kFold = KFold (n_splits=k_fold, shuffle = True)


print("-----------------------------------")
dataset = dataset[:,0:500]
#K-Fold loop
for fold, (train_i, valid_i) in enumerate(kFold.split(dataset)):
  print(f"FOLD{fold}")
  print("-----------------------------------")

  train_subsample = torch.utils.data.SubsetRandomSampler(train_i)
  valid_subsample = torch.utils.data.SubsetRandomSampler(valid_i)

  genotype_dataset = TensorDataset(torch.tensor(dataset, dtype=torch.float64))
  #genotype_dataset = genotype_dataset[:,0:500]
  train_loader = DataLoader(genotype_dataset, batch_size = 10, sampler=train_subsample)
  valid_loader = DataLoader(genotype_dataset, batch_size = 10, sampler=valid_subsample)

  wandb_logger = pl.loggers.WandbLogger(project="Imputation Autoencoder Project")
  model = autoencoder_model_1(500)
  trainer = pl.Trainer(
      logger=wandb_logger,    # W&B integration
      log_every_n_steps=1,    # set the logging frequency
      gpus=-1,                # use all GPUs
      max_epochs=epochs,      # number of epochs
      deterministic=True,     # keep it deterministic
  )
  args = {
      "learning_rate": 0.001,
      "architecture": "Autoencoder",
      "dataset": "1000 Genome Project",
      "layer_1": 256,
      "activation_1": "relu",
      "layer_2": 128,
      "activation_2": "relu",
      "optimizer": "Adam",
      "loss": "MSE",
      "metric": "CV Accuracy",
      "epoch": 30,
      "batch_size": 10,
      "n_hidden_layers":8}
  group_name = "bin500_group_256_128_64_32"
  name=group_name+'_seed_'+str(np.random.randint(100000000))
  run=wandb.init(project="Imputation Autoencoder Project",save_code=False,
                group=group_name,entity="connia",name=name,
                mode="online",id=name,config=args,allow_val_change=True)
  trainer.fit(model, train_loader)
  wandb.finish()
  print('Training process has finished. Saving trained model.')
  print('Starting testing')

  #Save model
 # path = f"./model-fold-{fold}.pth"
 # torch.save(model.state_dict(), path)

  #Evaluation per fold
  correct, total = 0,0
  model.eval()
  with torch.no_grad():
    for i, data in enumerate(valid_loader,0):
      inputs = data
      outputs = model(inputs[0])
      #predicted = outputs.data
      prediction = np.round(outputs)
      total += inputs[0].shape[0] * inputs[0].shape[1]
      prediction = prediction.numpy()
      inputs = inputs[0].numpy()
      correct += np.sum(prediction==inputs)
     # error = F.mse_loss(outputs, torch.tensor(mean_dataset[valid_i][i*10:i*10+10,:], dtype=torch.float64))

      # Print accuracy
      print('Accuracy for fold %d: %d %%' % (fold, 100.0 * (correct/total)))
      print('--------------------------------')
      results[fold] = 100.0 * (correct/total)
    # Print fold results
  print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_fold} FOLDS')
  print('--------------------------------')
  sum = 0.0
  for key, value in results.items():
    print(f'Fold {key}: {value} %')
    sum += value
  print(f'Average: {sum/len(results.items())} %')
