"""Synthetic three-step native training body for startup-admission controls."""
import os
import torch

torch.manual_seed(20261023)
model=torch.nn.Sequential(torch.nn.Linear(16,64),torch.nn.SiLU(),torch.nn.Linear(64,2)).to('npu')
optimizer=torch.optim.SGD(model.parameters(),lr=.01)
x=torch.randn(32,16,device='npu'); y=torch.randn(32,2,device='npu')
losses=[]
for _ in range(3):
    optimizer.zero_grad(set_to_none=True)
    loss=(model(x)-y).square().mean(); loss.backward(); optimizer.step()
    losses.append(float(loss))
torch.npu.synchronize()
torch.save({'losses':losses,'state':{k:v.detach().cpu() for k,v in model.state_dict().items()}},os.environ['STARTUP_TEST_RESULT'])
