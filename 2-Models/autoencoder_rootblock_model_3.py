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
import sys

#Function to add the softmax of two overlapping blocks together
def concat_softmax (first_block, second_block):
  x12 = torch.concat((first_block,second_block), dim = 1)
  x12 = x12.view(-1,2,3,1000).sum(axis=1)
  return x12

if __name__ == "__main__":

    #Load dataset and create a list of accuracy and errors
    vcf = vcf_data_loader.FixedSizeVCFChunks("all_1kg_chr1_phased_GRCh38_snps_maf0.01.recode.vcf.gz")
    accuracies = []
    error = []
    
    #predictions = torch.tensor(pd.read_csv('./model_predictions.csv')) This causes an out of memory kill
    csv.field_size_limit(sys.maxsize)
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

        # Attempting to parse the csv file into a list of floats
        tensors = []
        for sample in line: 
            new_list = []
            sample = sample[2:-2] #remove list square brackets
            for index, output in enumerate(sample.split(",")):
                if index== 1999 or index == 3999 : 
                    output = output[:-1] # remove closing square bracket
                elif index == 2000 or index == 4000: 
                    output = output[2:] # remove whitespace and opening square bracket
                new_list.append(float(output))
            tensors.append(torch.tensor(new_list).reshape(1,6000))
            
        predictions = torch.cat(tensors, dim=0)

        # Calculating accuracy and error for each block's output 
        # Note there is a different process (no need for concat_softmax() call) 
        # for blocks that do not need to be combined (first and last block)
        losses = []
        if i == 0: 
            x1 = predictions[:,0:3000].view(-1, 3, 1000)
            previous_block = predictions[:,3000:6000].view(-1, 3, 1000)
            for position in range(1000):
                loss = F.nll_loss(x1[:,:,position], dataset[:,position].to(int))
                losses.append(loss)
            mean_loss =  sum(losses)/len(losses)
            error.append(mean_loss)
            accuracy = (dataset == x1.argmax(dim=1)).to(float).mean(dim=1).mean()
            accuracies.append(accuracy)
        else:
            x1 = predictions[:,0:3000].view(-1, 3, 1000)
            aggregated_predictions = concat_softmax(previous_block,x1)
            previous_block = predictions[:,3000:6000].view(-1, 3, 1000)
            for position in range(1000):
                loss = F.nll_loss(aggregated_predictions[:,:,position], dataset[:,position].to(int))
                losses.append(loss)
            mean_loss =  sum(losses)/len(losses)
            error.append(mean_loss)
            accuracy = (dataset == aggregated_predictions.argmax(dim=1)).to(float).mean(dim=1).mean()
            accuracies.append(accuracy)
        if i == 933:
            x2 = predictions[:,3000:6000].view(-1, 3, 1000)
            for position in range(1000):
                loss = F.nll_loss(x2[:,:,position], dataset[:,position].to(int))
                losses.append(loss)
            mean_loss =  sum(losses)/len(losses)
            error.append(mean_loss)
            accuracy = (dataset == x2.argmax(dim=1)).to(float).mean(dim=1).mean()
            accuracies.append(accuracy)
        print(f"Error at the {i} block: {mean_loss}")
        print (f"Accuracy at the {i} block: {accuracy}")
        
    print(f"AVERAGE ACCURACY: {sum(accuracies)/len(accuracies)}")
    print(f"AVERAGE ERROR: {sum(error)/len(error)}")

    f.close()

