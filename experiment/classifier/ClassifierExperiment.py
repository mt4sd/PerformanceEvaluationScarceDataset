import numpy as np
import torch
from torch import nn
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from abc import ABC, abstractmethod

from IPDL import ClassificationInformationPlane
from IPDL.optim import AligmentOptimizer
from IPDL.functional import matrix_estimator
from model.utils import TBLog


class ClassifierExperiment(ABC):
    def __init__(self) -> None:
        super(ClassifierExperiment, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = ''
        self.model = nn.Identity()

        #information plane estimator
        self.ip = ClassificationInformationPlane(self.model, use_softmax=True)

        #IP Estimation using Z-space as input
        self.z_input = True

        self.n_classes_ = 2

    def train(self, train_loader: DataLoader, test_loader: DataLoader,
            tb_writer: SummaryWriter, n_epoch=800) -> None:

        (_, Ax), (Ky, Ay) = self.get_matrices(test_loader, self.z_input)

        it_val_inputs, _ = next(iter(test_loader))

        # Tensorboard Log
        tb_log = TBLog(self.model, tb_writer)

        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

        # Loss function
        criterion = nn.CrossEntropyLoss()

        # IPDL 
        #       Optimizer
        matrix_optimizer = AligmentOptimizer(self.model, beta=0.9, n_sigmas=200)
        #       InformationPlane
        self.ip = ClassificationInformationPlane(self.model, use_softmax=True)

        epoch_iterator = tqdm(
            range(n_epoch),
            leave=True,
            unit="epoch",
            postfix={"model": self.model_name,  "tls": "%.4f" % 1, "vls": "%.4f" % 1,},
        )
        
        for epoch in epoch_iterator:
            # IP, MI
            with torch.no_grad():
                self.model.eval()
                self.model(it_val_inputs.to(self.device))
                matrix_optimizer.step(Ky.to(self.device))
            
            Ixt, Ity = self.ip.computeMutualInformation(Ax, Ay)
            MI = { 'MutualInformation/I(X,T)': {},
                'MutualInformation/I(T,Y)': {}  }
            for idx in range(len(Ixt)):
                scalar_name = 'L{}'.format(idx)
                MI['MutualInformation/I(X,T)'][scalar_name] = Ixt[idx]
                MI['MutualInformation/I(T,Y)'][scalar_name] = Ity[idx]

            tb_log.log(MI, epoch-1, include_conv=False)

            # Train
            self.model.train()
            loss_tr = []
            for step, (input, target) in enumerate(train_loader):
                input = input.to(self.device)
                target = target.flatten().to(self.device)
                output = self.model(input)
                batch_loss = criterion(output, target)
                loss_tr.append(batch_loss.detach().item())
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

            with torch.no_grad():
                # Validation
                self.model.eval()
                loss_ts = []
                for step, (input, target) in enumerate(test_loader):
                    input = input.to(self.device)
                    target = target.flatten().to(self.device)
                    output = self.model(input)
                    batch_loss = criterion(output, target)
                    loss_ts.append(batch_loss.detach().cpu().numpy())

                scheduler.step()

                epoch_iterator.set_postfix(
                    model=self.model_name, tls="%.4f" % np.mean(loss_tr), vls="%.4f" % np.mean(loss_ts),
                )

            scalars = { 'Loss': {'Train' : np.mean(loss_tr), 'Test': np.mean(loss_ts)},
                    'Learning Rate': scheduler.get_last_lr()[0]}
            
            tb_log.log(scalars, epoch, include_conv=False)

    def get_matrices(self, data_loader: DataLoader, z_space=True) -> tuple:
        '''
            Ax generates from the latent space

            Params:
            ----
            
            - z_space (bool): Indicate if the Kx matrix is generated from encoder space Z
        '''
        it_val_inputs, it_val_targets = next(iter(data_loader))
        if z_space:
            it_val_inputs = self.model.encoder(it_val_inputs.to(self.device)).flatten(1)
        else:
            it_val_inputs = it_val_inputs.flatten(1).to(self.device)
        it_val_targets = one_hot(it_val_targets.flatten(), num_classes=self.n_classes_).float().to(self.device) 

        Kx, Ax = matrix_estimator(it_val_inputs, sigma=4)
        Ky, Ay = matrix_estimator(it_val_targets, sigma=.1)

        return (Kx, Ax), (Ky, Ay)
