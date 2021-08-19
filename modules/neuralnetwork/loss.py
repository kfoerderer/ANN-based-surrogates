import torch
import torch.nn as nn


class L1RegularizationLoss(nn.Module):
    """
    Regularization term: scale * L1(parameters)
    """
    def __init__(self, scale: float, device: torch.device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.scale = scale

    def __str__(self):
        return 'L1RegularizationLoss(device={!s}, scale={:E})'.format(self.device, self.scale)

    def __repr__(self):
        return str(self)

    def forward(self, neural_network: nn.Module) -> torch.Tensor:
        parameters = neural_network.parameters()
        loss = torch.zeros(1, device=self.device)
        for parameter in parameters:        
            loss += parameter.norm(1)
        return self.scale * loss

class SigmoidGatedL1RegularizationLoss(nn.Module):
    """
    Regularization term:
        
        output_scale * L1(parameters) * Sigmoid(input_scale * (L1(parameters) - input_shift))

    With input_scale=0 and output_scale=2 this is identical to the L1Regularization (without any scaling)
    """
    def __init__(self, input_shift: float, input_scale: float, output_scale: float, device: torch.device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.input_shift = input_shift
        self.input_scale = input_scale
        self.output_scale = output_scale

    def __str__(self):
        return 'SigmoidGatedL1RegularizationLoss(device=%s, input_shift=%f, input_scale=%f, output_scale=%f)'%(self.device, self.input_shift, self.input_scale, self.output_scale)

    def __repr__(self):
        return str(self)

    def forward(self, neural_network: nn.Module) -> torch.Tensor:
        parameters = neural_network.parameters()
        loss = torch.zeros(1, device=self.device)
        for parameter in parameters:        
            loss += parameter.norm(1)
        return self.output_scale * loss * torch.sigmoid((loss-self.input_shift) * self.input_scale)

class MixedLoss(nn.Module):
    """
    Combines multiple loss function for individual fragments of the model output

    #### Arguments
    - losses [(nn.Module, int)]: A (ordered) list of losses combined with the respective number of elements to take from the ANN output. The number of elements does not necessarily equal the number of parameters to estimate (for instance when using a softmax distribution).
    """

    def __init__(self, losses: [(nn.Module,int)]=[], weights: [float]=None, device: torch.device=torch.device('cpu')):
        super().__init__()

        if weights is None:
            self.weights = torch.ones(len(losses), device=device)
        else:
            self.weights = torch.Tensor(weights).to(device)
        self.losses = losses
        self.device = device
        for i, (loss, number) in enumerate(losses):
            self.add_module('%d_%s_x_%d'%(i, type(losses).__name__, number), loss)

    def __str__(self):
        return 'MixedLoss({},{},{})'.format(self.losses, self.weights, self.device)

    def __repr__(self):
        return str(self)

    def forward(self, x: torch.Tensor, target: torch.Tensor):
        result = torch.zeros(1, device=self.device)
        input_neuron_pos = 0
        target_neuron_pos = 0
        for (loss, number), weight in zip(self.losses, self.weights):
            if type(loss) is nn.CrossEntropyLoss: # or type(loss) is nn.BCEWithLogitsLoss: ?
                result += weight * loss(x[:,input_neuron_pos:input_neuron_pos+number], target[:,target_neuron_pos].long())
                target_neuron_pos += 1
            else:
                result += weight * loss(x[:,input_neuron_pos:input_neuron_pos+number], target[:,target_neuron_pos:target_neuron_pos+number])
                target_neuron_pos += number
            input_neuron_pos += number                
        return result

class MDNLoss(nn.Module):

    def __init__(self, gaussians_count: int):
        super().__init__()

        self.gaussians_count = gaussians_count

    def __str__(self):
        return 'MDNLoss(gaussians_count=%s)'%(self.gaussians_count)

    def __repr__(self):
        return str(self)

    def forward(self, x: torch.Tensor, target: torch.Tensor):
        # compute estimate(s)
        estimate = x.view(x.size(0), -1, self.gaussians_count, 3)    
        pi = nn.functional.softmax(estimate[:,:,:,0], -1)
        mu = estimate[:,:,:,1]
        sigma = torch.exp(estimate[:,:,:,2])

        # compute loss
        # probability of target for each gaussian (hence unsqueeze(2))
        gaussians = torch.distributions.Normal(loc=mu, scale=sigma)       
        log_prob = gaussians.log_prob(target.unsqueeze(2))
        
        # numerically better implementation of log(sum(exp(a_i)))
        # log(sum(exp(a_i-a*)))+a*
        # https://discuss.pytorch.org/t/dealing-with-nans-in-gradients/5529/7
        max_log_prob = torch.max(log_prob, dim=2, keepdim=True).values
        loss = torch.exp(log_prob-max_log_prob) 
        loss = torch.sum(pi * loss, dim=2) # weighted probability
        loss = torch.log(loss) + max_log_prob.squeeze(2) # log probability

        return torch.mean(-loss).unsqueeze(0)
