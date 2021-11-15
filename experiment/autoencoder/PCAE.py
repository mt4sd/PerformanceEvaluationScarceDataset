import os
from torch import load
from torch import nn
from model import SDAE, SDAE_TYPE

from .AutoEncoderExperiment import AutoEncoderExperiment

class PCAE(AutoEncoderExperiment):
    def __init__(self):
        super(PCAE, self).__init__()

        self.model_name = 'PCAE'
        self.model = SDAE([1, 6, 8, 16], SDAE_TYPE.conv, activation_func=nn.Sigmoid(), dropout=False).to(self.device)
            
    def load_model(self, file: str) -> None:
        self.model.load_state_dict(load(file))