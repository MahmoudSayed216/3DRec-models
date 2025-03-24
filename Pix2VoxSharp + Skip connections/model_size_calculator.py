# this method is a very accurate but not an exact calculation
import yaml

configs = None
with open("config.yaml", "r") as f:
    configs = yaml.safe_load(f)

from model import full_model

model = full_model.Pix2VoxSharp(configs)


sum_params = lambda M: sum(p.numel() for p in M.parameters()) 

# parameters = sum([sum_params(M) for M in [Encoder, Decoder, Merger, Refiner]])
parameters = sum_params(model)
print("params: ", parameters)

size_in_mb = parameters>>18 


print("Model size: ", size_in_mb)




# ## INFO TO BE ADDED TO THE PRESENTATION
# import torchinfo
# import torch
# from Model import ALittleBitOfThisAndALittleBitOfThatNet



# model = ALittleBitOfThisAndALittleBitOfThatNet("cpu", 0.2, False)

# torchinfo.summary(model, input_data=(torch.randn(1, 5, 3, 224, 224), torch.randn(1, 5, 3, 224, 224)))

