import torch

model = torch.load('/Volumes/蒋昊/推荐系统代码/NRMS_GCN_pytorch/model/model.pkl',map_location=torch.device('cpu'))
for name, parameters in model.named_parameters():
    print(name,':',parameters)
