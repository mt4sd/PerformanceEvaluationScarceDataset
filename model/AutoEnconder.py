import torch
from torch import tensor, Tensor, nn
from torch.nn.modules import activation, linear
from enum import Enum
from IPDL import MatrixEstimator

# Just for convolutionals layers
class AE_CONV_UPSAMPLING(Enum):
    up_layer= 1,
    transp_conv = 2

''' This class contains a Layer-wise training '''
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encode = nn.Sequential()

        self.decode = nn.Sequential()

        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = None

    def get_encode(self, dropout=False):
        encode = []
        for module in self.encode:
            if dropout or isinstance(module, (nn.Linear, nn.Conv2d, nn.ReLU, nn.Sigmoid, 
                                nn.Tanh, nn.Identity, nn.MaxPool2d, nn.BatchNorm1d, nn.BatchNorm2d,
                                MatrixEstimator)):
                encode.append(module)

        return nn.Sequential(*encode)

    def get_decode(self, dropout=False):
        decode = []
        for module in self.decode:
            if dropout or isinstance(module, (nn.Linear, nn.Conv2d, nn.ReLU, nn.Sigmoid, 
                                nn.Tanh, nn.Identity, nn.MaxPool2d, nn.BatchNorm1d, 
                                nn.BatchNorm2d, MatrixEstimator, nn.Upsample, nn.ConvTranspose2d)):
                decode.append(module)

        return nn.Sequential(*decode)

    def is_valid_activation_fuction(self, activation_funct: nn.Module) -> bool:
        return isinstance(activation_funct, (activation.ReLU, activation.Sigmoid,
                                                activation.Tanh, linear.Identity))

    def encode_decode_activation(self, activation_func) -> list:
        r'''
            This function obtain the encoder and decoder activation function.
        '''
        if isinstance(activation_func, (list, tuple)):
            if len(activation_func) != 2:
                raise ValueError("activation_func as a list has to contain 2 activation function, the encoder and decoder activation function")
            
            return activation_func
        else:
            return activation_func, activation_func

    def forward(self, x: Tensor) -> Tensor:
        # Train each autoencoder individually
        x = x.detach()
        y = self.encode(x)

        if self.training and isinstance(self.optimizer, torch.optim.Optimizer):
            x_reconstruct = self.decode(y)
            loss = self.criterion(x_reconstruct, tensor(x.data, requires_grad=False))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return y.detach()

    def weight_init(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)

    def reconstruct(self, x: Tensor) -> Tensor:
        return self.decoder(x)


class AutoEncoderLinear(AutoEncoder):
    def __init__(self, n_in, n_out, activation_func = nn.ReLU(inplace=True), **kwargs):
        super(AutoEncoderLinear, self).__init__()

        encode_act_func, decode_act_func = self.encode_decode_activation(activation_func)

        self.encode = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(n_in, n_out),
                encode_act_func if self.is_valid_activation_fuction(encode_act_func) else nn.ReLU(inplace=True),
                MatrixEstimator(0.1)
            )

        self.decode = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(n_out, n_in),
                decode_act_func if self.is_valid_activation_fuction(decode_act_func) else nn.ReLU(inplace=True),
                MatrixEstimator(0.1)
            )

        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.5)

        for m in self.modules():
            self.weight_init(m)


class AutoEncoderConv(AutoEncoder):
    def __init__(self, n_in, n_out, activation_func = nn.ReLU(inplace=True), 
            upsample = AE_CONV_UPSAMPLING.up_layer, skip_connection=False, **kwargs):
        super(AutoEncoderConv, self).__init__()

        encode_act_func, decode_act_func = self.encode_decode_activation(activation_func)

        self.encode = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(n_in, n_out, 5, stride=2, padding=2),
            encode_act_func if self.is_valid_activation_fuction(encode_act_func) else nn.ReLU(inplace=True),
            MatrixEstimator(0.1)
        )

        n_out_decode = (n_out + n_in) if skip_connection else n_out
        self.decode = nn.Sequential(
            nn.Dropout2d(0.2), 
            *(  nn.Upsample(scale_factor=2, mode='nearest'), ) if (upsample == AE_CONV_UPSAMPLING.up_layer) 
                else ( nn.ConvTranspose2d(n_out, n_out, 3, stride=2, padding=1, output_padding=1), ),
            nn.Conv2d(n_out_decode, n_in, 3, padding=1),
            decode_act_func if self.is_valid_activation_fuction(decode_act_func) else nn.ReLU(inplace=True),
            MatrixEstimator(0.1)
        )

        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.5)

        for m in self.modules():
            self.weight_init(m)
