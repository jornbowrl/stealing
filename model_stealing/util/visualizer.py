
import os 

import json 
import logging 
import numpy as np 

import time 
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision



def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.
    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """
    def flush(self):
        self.board_logger.flush()
        

    def __init__(self, log_name ):
        """Initialize the Visualizer class
        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        now = time.strftime("%c")
        logging.info('================ Training Loss (%s) ================\n' % now)
        logging.info(f'================save into {log_name} ================\n' )
#         self.log_name = os.path.join(opt.checkpoints_dir, opt.name)

        self.board_logger = SummaryWriter(log_name)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False


    def display_current_results(self, visuals, epoch, save_result=None,normalize=True):
        """Display current results on visdom; save current results to an HTML file.
        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        
        all_img = list(visuals.values())
        all_lbl =[str(x) for x in  list(visuals.keys()) ] 
        label   = "_".join(all_lbl) [:100]
        
        
        all_img = torch.cat(all_img) 
        
        image_numpy=torchvision.utils.make_grid(tensor=all_img,normalize=normalize,
                                     nrow=4, padding=2, )
#         image_numpy = tensor2im(image)
        self.board_logger.add_image(label, image_numpy,global_step=epoch)
        

#     def plot_current_losses(self, epoch, counter_ratio, losses):
#         """display the current losses on visdom display: dictionary of error labels and values
#         Parameters:
#             epoch (int)           -- current epoch
#             counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
#             losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
#         """
#         t=epoch + counter_ratio
#         for tag ,value in losses.items():
#             self.board_logger.add_scalar(str(tag), value, t)
    def plot_current_losses(self, total_iters, losses):
        """display the current losses on visdom display: dictionary of error labels and values
        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        t=total_iters
        for tag ,value in losses.items():
            self.board_logger.add_scalar(str(tag), value, t)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp=0.0, t_data=0.0):
        """print current losses on console; also save the losses to the disk
        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        logging.info(message)
#         with open(self.log_name, "a") as log_file:
#             log_file.write('%s\n' % message)  # save the message




if __name__=="__main__":
    import torch 
    
    #display_current_results(self, visuals, epoch, save_result=None):
    
    vis=  Visualizer("experiments/123")
    
    for i in range(10):
        visuals={k:torch.randn(4,3,10,10) for k in range(20) }
        vis.display_current_results(
            visuals=visuals, epoch=i, save_result=None)
        
        total_loss= {k:torch.tensor([0.0], requires_grad=True) for k in range(10) }
        vis.plot_current_losses( epoch=i, 
                                 counter_ratio=0.1, losses=total_loss)
        
        vis.print_current_losses(epoch=i, iters=10, losses=total_loss, t_comp=0.0, t_data=0.0)
    
    vis.board_logger.flush()