from collections import deque
from itertools import islice

def sliding_window_iter(iterable, size):
    '''
        Iterate through iterable using a sliding window of several elements.
        Important: It is a generator!.
        
        Creates an iterable where each element is a tuple of `size`
        consecutive elements from `iterable`, advancing by 1 element each
        time. For example:
        >>> list(sliding_window_iter([1, 2, 3, 4], 2))
        [(1, 2), (2, 3), (3, 4)]
        
        source: https://codereview.stackexchange.com/questions/239352/sliding-window-iteration-in-python
    '''
    iterable = iter(iterable)
    window = deque(islice(iterable, size), maxlen=size)
    for item in iterable:
        yield tuple(window)
        window.append(item)
    if window:  
        # needed because if iterable was already empty before the `for`,
        # then the window would be yielded twice.
        yield tuple(window)


from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from IPDL import MatrixEstimator

class TBLog():
    '''
        TensorBoard Log. This class contains the essential functions
        in order to obtain a standardized tensorboard log.
    '''
    def __init__(self, model: nn.Module, tb_writer: SummaryWriter):
        if not isinstance(tb_writer, SummaryWriter):
            raise TypeError('tb_writer has to be a Tensorboard SummaryWriter class')
        
        self.matrix_estimators = []
        for module in model.modules():
            if isinstance(module, (MatrixEstimator)):
                self.matrix_estimators.append(module)

        self.tb_writer = tb_writer
        
    
    def __conv_outputs__(self, epoch: int, transform=None):
        '''
            Using the first sample, generates a image which
            contains the output of the different filters or channels.
        '''
        for idx, matrix_estimator in enumerate(self.matrix_estimators[:-1]):
            if not self.__is_image_batch(matrix_estimator.x):
                continue
            
            out = matrix_estimator.x[0].unsqueeze(1)
            self.__images__('Filters/CL{}'.format(idx), out, epoch, transform)
    
    def __scalars__(self, scalars: dict, epoch: int):
        '''
            @param scalars: this parameters can contains a scalar value or another 'dict'.
        '''
        for key, values in scalars.items():
            add_scalar = self.tb_writer.add_scalars if isinstance(values, dict) else self.tb_writer.add_scalar
            add_scalar(key, values, epoch)

    def __images__(self, label: str, images: Tensor, epoch: int, transform=None):
        if transform:
            images = transform(images)
        img_grid = make_grid(images)
        self.tb_writer.add_image(label, img_grid, epoch)

    def __is_image_batch(self, x: Tensor):
        return len(x.size()) == 4

    def log(self, scalars, epoch, include_conv=False, transform=None, **kwargs):
        '''
            Logging data in a tensorboard.

            Params:
            ------
                scalars (float or dict): this parameter can contain a scalar value or another 'dict' whith
                    keys and values.
                
                epoch (int): training epoch 

                include_conv (bool): Set to 'True' if you want to log the output of convolutional layers which
                    are saved on MatrixEstimator class. 

                transform (nn.Module): Transformation applied to image data. 

            Optional params:
            ---------------
                input (Tensor): Tensor which contains a batch of images

                output (Tensor): Tensor which contains a batch of images

        '''
        self.__scalars__(scalars, epoch)
        if include_conv:
            self.__conv_outputs__(epoch, transform=transform)
       
        if 'input' in kwargs.keys():
            if self.__is_image_batch(kwargs['input']):
                self.__images__('Input', kwargs['input'], 0, transform=transform)
        if 'output' in kwargs.keys():
            if self.__is_image_batch(kwargs['output']):
                self.__images__('Output', kwargs['output'], epoch, transform=transform)

        self.tb_writer.flush()