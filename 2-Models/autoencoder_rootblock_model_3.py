import csv
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from torchvision.transforms import Compose
from torch import nn, optim
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
import vcf_data_loader
import pandas as pd

#Function to add the softmax of two overlapping blocks together
def concat_softmax (first_block, second_block):
  x12 = torch.concat((first_block,second_block), dim = 1)
  x12.reshape(-1,2,3,1000).sum(axis=1)
  return x12

if __name__ == "__main__":

    #Load dataset and create a list of accuracy and errors
    vcf = vcf_data_loader.FixedSizeVCFChunks("all_1kg_chr1_phased_GRCh38_snps_maf0.01.recode.vcf.gz")
    accuracies = []
    error = []
    
    #predictions = torch.tensor(pd.read_csv('./model_predictions.csv')) This causes an out of memory kill

    #Open and read the csv file of model outputs
    f = open('./model_predictions.csv', 'r')
    reader = csv.reader(f)

    for i in range(0,934):
        #Get the chuunk number and index # within the chunk
        block = i//5 
        index = i%5

        #Get the tensor for that chunk & Create a tensor of a subset of 1000 SNPs within the chunk
        chunk = vcf.get_tensor_for_chunk_id(block)
        dataset = torch.tensor(chunk[:,(index)*1000:(index+1)*1000],dtype=torch.float64)

        #Get the next line of the csv file
        line = next(reader)
        print(len(line))

        # Attempting to parse the csv file (This is where the script crashes) into a list of floats
        for sample in line: 
            new_list = []
            sample = sample[2:len(sample)-2]
            for i in sample.split(","):
                new_list.append(float(i))
            print(len(new_list))
       # predictions = torch.tensor(float(z) for z in line2)

        # Calculating accuracy and error for each block's output 
        # Note there is a different process (no need for concat_softmax() call) f
        #or blocks that do not need to be combined (first and last block)
        if i == 0: 
            x1 = predictions[:,:,0:1000]
            previous_block = predictions[:,:,1000:2000]
            loss = F.nll_loss(x1, dataset.to(int))
            error.append(loss)
            accuracy = (dataset == x1.argmax(dim=1)).to(float).mean(dim=1).mean()
            accuracies.append(accuracy)
        else:
            x1 = predictions[:,:,0:1000]
            aggregated_predictions = concat_softmax(previous_block,x1)
            previous_block = predictions[:,:,1000:2000]
            loss = F.nll_loss(aggregated_predictions, dataset.to(int))
            error.append(loss)
            accuracy = (dataset == aggregated_predictions.argmax(dim=1)).to(float).mean(dim=1).mean()
            accuracies.append(accuracy)
        if i == 933:
            x2 = predictions[:,:,1000:2000]
            loss = F.nll_loss(x2, dataset2.to(int))
            error.append(loss)
            accuracy = (dataset2 == x2.argmax(dim=1)).to(float).mean(dim=1).mean()
            accuracies.append(accuracy)
        

    print(error)
    print (accuracy)
    

    print("AVERAGE ACCURACY:" + sum(accuracies)/len(accuracies))
    print("AVERAGE ERROR:"+ sum(error)/len(error))

    f.close()

