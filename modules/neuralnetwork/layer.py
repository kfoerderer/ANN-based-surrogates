import torch
import torch.nn as nn

class DebugLayer(nn.Module):

    def __init__(self, depth):
        super().__init__()

        self.depth = depth

    def forward(self, input):
        print('nn, DebugLayer %d, %s'%(self.depth, str(input.size())))
        return input

class SkipConnection(nn.Module):
    """
    Module for skipping other modules.
    
    The first time called, it passes through the input and saves the reference. Any subsequent time the stored reference is appended to the input.
    """
    def __init__(self):
        super().__init__()
        
        self.store = None
        
    def forward(self, x):        
        if self.store is None:
            self.store = x.clone()
            return x        

        result = torch.cat((x, self.store), dim=-1)
        self.store = None
        
        return result