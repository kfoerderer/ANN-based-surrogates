import torch
import torch.nn as nn

class Swish(nn.Module):
    """
    Swish activation function.
    
    y = x * sigmoid (beta * x)
    """
    def __init__(self, beta):
        super().__init__()
        
        self.sigmoid = nn.Sigmoid()
        self.beta = nn.Parameter(torch.Tensor([beta]))
        
    def forward(self, x):
        return x * self.sigmoid(self.beta * x)


class MixedActivation(nn.Module):
    """
    Module for applying different activation functions to different neuron outputs
    """
    def __init__(self, activations: [(nn.Module,int)]=[], device: torch.device=torch.device('cpu')):
        super().__init__()
        self.activations = activations
        self.device = device
        
        for i, (activation, number) in enumerate(activations):
            self.add_module('%d_%s_x_%d'%(i, type(activation).__name__, number), activation)
        
    def forward(self, x):
        results = torch.Tensor([], device=self.device)
        
        activated_neuron_count = 0
        for activation, number in self.activations:
            results = torch.cat([results, activation(x[:,activated_neuron_count:activated_neuron_count+number])])
            activated_neuron_count += number
        
        if activated_neuron_count < x.size(-1):
            results = torch.cat([results, x[:,activated_neuron_count:]])
        
        return results