import numpy as np
import torch
from torch import nn, Tensor
from .SDAE import SDAE
from IPDL import MatrixEstimator

class ExpClassifier(nn.Module):
    def __init__(self, sdae: SDAE, input_size = (128, 128), n_classes = 2, encoder_frozen = True) -> None:
        super(ExpClassifier, self).__init__()

        output_size = np.array(input_size)
        out_channels = 1
        n_encoder_path = 0
        if isinstance(sdae, SDAE):
            self.encoder = sdae.get_encode(matrix_estimator=not(encoder_frozen))
            for m in self.encoder.modules():
                if isinstance(m, nn.Conv2d):
                    output_size = self.__spatial_resolution_out(output_size, m.kernel_size, m.stride,
                                        m.padding)
                    out_channels = m.out_channels
                    n_encoder_path += 1

                if isinstance(m, nn.Linear):
                    output_size = m.out_features
        else:
            self.encoder = sdae

        if encoder_frozen:
            for param in self.encoder.parameters():
                param.requires_grad = False


        in_features = (output_size.prod() * out_channels).astype(np.uint) if isinstance(output_size, np.ndarray) else output_size

        self.classifier = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.Sigmoid(),
                MatrixEstimator(0.1),
                nn.Linear(512, 256),
                nn.Sigmoid(),
                MatrixEstimator(0.1),
                nn.Linear(256, 64),
                nn.Sigmoid(),
                MatrixEstimator(0.1),
                nn.Linear(64, n_classes),
                MatrixEstimator(0.1)
            )

        for m in self.classifier:
            self.__weight_init(m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return self.classifier(x.flatten(1))

    def __spatial_resolution_out(self, input_size, kernel_size, stride, padding=0):
        '''
            This function is used in order to obtain the spatial resolution
            output of a convolution            
        '''
        input_size = np.array(input_size)
        stride = np.array(stride)
        padding = np.array(padding)
        kernel_size = np.array(kernel_size)
        return ((input_size + 2*padding - kernel_size)/stride + 1).astype(np.uint)

    def __weight_init(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight.data, nonlinearity='sigmoid')