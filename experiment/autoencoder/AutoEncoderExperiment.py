import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from abc import ABC

from torch.optim.lr_scheduler import ExponentialLR

from model.utils import TBLog

from functools import reduce
from IPDL import AutoEncoderInformationPlane
from IPDL.optim import SilvermanOptimizer
from IPDL.functional import matrix_estimator


class AutoEncoderExperiment(ABC):
    def __init__(self) -> None:
        super(AutoEncoderExperiment, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = ''
        self.model = nn.Identity()
        self.gamma = 2 # Gamma value for sigma estimation

        # information plane estimator
        self.ip = AutoEncoderInformationPlane(self.model)

    def get_Ax(self, loader: DataLoader):
        '''
            First batch as input for Entropy estimators
        '''
        val_inputs, val_targets = next(iter(loader))
        val_inputs = val_inputs.to(self.device)

        n = val_inputs.size(0)
        d = val_inputs.size(1) if len(val_inputs.shape) == 2 else reduce(lambda x, y: x*y, val_inputs.shape[2:])
        
        sigma = self.gamma * n ** (-1 / (4+d)) 

        _, Ax = matrix_estimator(val_inputs.flatten(1), sigma=sigma)
        return Ax.to(self.device)

    def train(self, train_loader: DataLoader, test_loader: DataLoader, 
            tb_writer: SummaryWriter, n_epoch=250) -> None:
        tb_log = TBLog(self.model, tb_writer)
        val_inputs, _ = next(iter(test_loader))
        val_inputs = val_inputs.to(self.device)
        Ax = self.get_Ax(test_loader).to(self.device)

        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4)
        scheduler = ExponentialLR(optimizer, gamma=0.999)

        # IPDL 
        #       Optimizer
        matrix_optimizer = SilvermanOptimizer(self.model, gamma=self.gamma, normalize_dim=False)
        #       InformationPlane
        self.ip = AutoEncoderInformationPlane(self.model)

        # Loss function
        criterion = torch.nn.MSELoss()

        epoch_iterator = tqdm(
            range(n_epoch),
            leave=True,
            unit="epoch",
            postfix={"model": self.model_name,  "tls": "%.4f" % 1, "vls": "%.4f" % 1,},
        )

        for epoch in epoch_iterator:
            # IP, MI
            self.model.eval()
            self.model(val_inputs)
            if epoch == 0: # Just first time
                matrix_optimizer.step()
            
            Ixt, Ity = self.ip.computeMutualInformation(Ax)
            MI = { 'MutualInformation/I(X,T)': {},
                'MutualInformation/I(T,Y)': {}  }
            for idx in range(len(Ixt)):
                scalar_name = 'CL{}'.format(idx)
                MI['MutualInformation/I(X,T)'][scalar_name] = Ixt[idx]
                MI['MutualInformation/I(T,Y)'][scalar_name] = Ity[idx]

            tb_log.log(MI, epoch-1, include_conv=True)
    
             # Train
            self.model.train()
            loss_tr = []
            for step, (input, _) in enumerate(train_loader):
                input = input.to(self.device)
                output = self.model(input)

                batch_loss = criterion(output, input)
                loss_tr.append(batch_loss.detach().item())
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

            with torch.no_grad():
                # Validation
                self.model.eval()
                loss_ts = []
                for step, (input, _) in enumerate(test_loader):
                    input = input.to(self.device)
                    output = self.model(input)
                    batch_loss = criterion(output, input)
                    loss_ts.append(batch_loss.detach().cpu().numpy())

                scheduler.step()

                epoch_iterator.set_postfix(
                    model=self.model_name, tls="%.4f" % np.mean(loss_tr), vls="%.4f" % np.mean(loss_ts),
                )

            scalars = { 'Loss': {'Train' : np.mean(loss_tr), 'Test': np.mean(loss_ts)},
                    'Learning Rate': scheduler.get_last_lr()[0]}
                
            tb_log.log(scalars, epoch, include_conv=False, input=input[:4].reshape(4,1,64,64), output=output[:4].reshape(4,1,64,64))