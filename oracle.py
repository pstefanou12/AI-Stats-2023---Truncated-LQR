import torch as ch
from abc import ABC


class oracle(ABC):
    """
    Oracle for data sets.
    """
    def __call__(self, x):
        """
        Membership oracle.
        Args: 
            x: samples to check membership
        """
        pass


class Identity(oracle): 
    """
    Identity membership oracle for DNNs. All logits are accepted within the truncation set.
    """
    def __call__(self, x): 
        return ch.ones(x.size()).prod(-1, keepdim=True)

    def __str__(self): 
        return 'identity'


class LogitBall(oracle): 
    """
    Truncation based off of norm of logits. Logt norm needs to be smaller than input bound.
    In other words, retain the input that the classifier is less certain on. Smaller 
    unnormalized log probabilities implies uncertainty in classification.
    """
    def __init__(self, bound): 
        self.bound = bound
        
    def __call__(self, x): 
        return (x.norm(dim=-1, keepdim=True) <= self.bound)

    def __str__(self): 
        return 'logit ball'


class LogitBallComplement(oracle): 
    
    """
    Truncation based off of complement norm of logits. Logit norm needs to be greater than input bound.
    In other words, retain the inputs that the classifier is more certain on. Larger 
    unnormalized log probabilities implies more certraining in classification.
    """
    def __init__(self, bound, temperature=ch.ones(1)): 
        self.bound = bound
        self.temperature = temperature
        
    def __call__(self, x): 
        x_ = x
        return (x_.norm(dim=-1, keepdim=True) >= self.bound)

    def __str__(self): 
        return 'logit ball complement'