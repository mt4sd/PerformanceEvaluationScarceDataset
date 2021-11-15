import torch
from torch import nn
from torch import tensor, Tensor
from torch.nn.modules import activation
from .AutoEnconder import AutoEncoderLinear, AutoEncoderConv
from .utils import sliding_window_iter as slide

from IPDL import MatrixEstimator
 

from enum import Enum
class SDAE_TYPE(Enum):
    linear = 1,
    conv = 2


class SDAE(nn.Module):
    def __init__(self, dims, sdae_type: SDAE_TYPE, activation_func=nn.ReLU(inplace=True), dropout=True, 
                    skip_connection=False, **kwargs):
        r'''
            Stacked Autoencoder composed of a symmetric decoder and encoder components accessible via the encoder and decoder
            attributes. The dimensions input is the list of dimensions occurring in a single stack
            e.g. [100, 10, 10, 5] will make the embedding_dimension 100 and the hidden dimension 5, with the
            autoencoder shape [100, 10, 10, 5, 10, 10, 100].
            We use ReLUs in all encoder/decoder pairs, except for g2 of the first pair (it needs to reconstruct input 
            data that may have positive  and  negative  values, such  as  zero-mean  images) and g1 of the last pair
            (so the final data embedding retains full information).
            :param dims: list of dimensions occurring in a single stack

            Args:
                dims:

                sdae_type:

                dropout:
        '''
        if isinstance(activation_func, list) and len(activation_func)!=len(dims)-1:
            raise ValueError('A list of activation has to be equal to the lenght of "dims" grouped in pairs')

        super(SDAE, self).__init__()
        self.type = sdae_type
        
        self.ae = []
        self.skip_connection = nn.Parameter(torch.tensor(False), requires_grad=False)
        
        ae = AutoEncoderLinear if self.type == SDAE_TYPE.linear else AutoEncoderConv
        self.skip_connection.data = torch.tensor(False) if self.type == SDAE_TYPE.linear else torch.tensor(skip_connection)

        for idx, dim in enumerate(slide(dims, 2)):
            activation = activation_func[idx] if isinstance(activation_func, list) else activation_func
            is_skip_connection = False if idx == 0 else skip_connection
            self.ae.append( ae(*dim, activation_func=activation, skip_connection=is_skip_connection, **kwargs) )
          
        self.encode = nn.Sequential(*list(map(lambda ae: ae.get_encode(dropout), self.ae)))
        self.decode = nn.Sequential(*list(map(lambda ae: ae.get_decode(dropout), self.ae[::-1])))

    def get_encode(self, matrix_estimator=False) -> nn.Sequential:
        r'''
            Indicate if you want to return the MatrixEstimator layers
        '''
        encode = []
        for sequential in self.encode:
            for module in sequential:
                if matrix_estimator or (not isinstance(module, MatrixEstimator)):
                    encode.append(module)

        return nn.Sequential(*encode)


    def forward(self, x: Tensor) -> Tensor:
        if not self.skip_connection:
            x = self.encode(x)
            return self.decode(x)
        else:
            return self.__forward_skip_connection(x)
    
    def __forward_skip_connection(self, x: Tensor) -> Tensor:
        r'''
            Forward function applying the skip connections
        '''
        out = []
        for layer in self.encode:
            x = layer(x)
            out.append(x.clone())
        
        x = out.pop()
        for layer in self.decode[:-1]:
            upsample, *conv = layer
            conv = nn.Sequential(*conv)

            x = upsample(x)
            x = torch.cat([x, out.pop()], dim=1)
            x = conv(x)

        return self.decode[-1](x)