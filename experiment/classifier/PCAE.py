'''
    Experiment for PCAE
'''

import torch, os
# import config # Project config
from torch import nn

from model import SDAE, SDAE_TYPE, AE_CONV_UPSAMPLING, ExpClassifier

from IPDL import ClassificationInformationPlane

from .ClassifierExperiment import ClassifierExperiment


class PCAE(ClassifierExperiment):
    def __init__(self, ae_state_dict_path: str, n_classes = 2):
        super(PCAE, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.n_classes_ = n_classes

        self.model_name = 'PCAE'
        ae = SDAE([1, 6, 8, 16], SDAE_TYPE.conv, activation_func=nn.Sigmoid(), dropout=False,
                 skip_connection=False, upsample=AE_CONV_UPSAMPLING.up_layer)
        
        print(ae.load_state_dict(torch.load(ae_state_dict_path)))
        # ae.load_state_dict(torch.load(os.path.join(config.MODELS_DIR, "{}/AE_DFU.pth".format(self.model_name))))

        self.model = ExpClassifier(sdae=ae, input_size=(64,64), n_classes=n_classes, encoder_frozen = True).to(self.device)
             
        # Information plane estimator
        self.ip = ClassificationInformationPlane(self.model, use_softmax=True)

        #IP Estimation using Z-space as input
        self.z_input = True