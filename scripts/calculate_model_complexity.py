import torch
from torchvision.models import resnet18
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
from fvcore.nn import ActivationCountAnalysis

model = resnet18()

inputs = torch.randn(1, 3, 128, 128)

act = ActivationCountAnalysis(model, inputs)

flops = FlopCountAnalysis(model, inputs)

model_complexity = flop_count_table(flops, activations=act)

print(model_complexity)