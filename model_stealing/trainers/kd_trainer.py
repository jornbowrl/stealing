import torch 
from . import util as networks
from .base  import BaseTrainer 

import torch.nn.functional as F 
import torch.nn as nn 



def loss_fn_kd(outputs, labels, teacher_outputs, params):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    '''
    copy from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py
    '''
    alpha = params.alpha
    T = params.temperature
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss


to_torch =  lambda x: torch.from_numpy(x) if not torch.is_tensor(x) else x 
to_np =  lambda x: x.cpu().numpy() if  torch.is_tensor(x) else x 

class KDTrainer(BaseTrainer):
 
    def __init__(self,victim_net,synthenic_net,opt,**kwargs):
        BaseTrainer.__init__(self,opt)

        self.criterionIdt = torch.nn.L1Loss()

        
        self.teacher_model = victim_net
        self.student_model = synthenic_net
        

        self.optimizer_G = torch.optim.Adam(self.student_model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers.append(self.optimizer_G)
        
        self.teacher_outputs = kwargs.get("teacher_outputs",[]) 

        self.loss_names=["kd"]
        self.visual_names=["A_input"]
 
 
    def set_input(self,**kwargs):

        self.A_input=kwargs.get("A_input",None) 
        self.A_input_lbl=kwargs.get("A_input_lbl",None)  
        self.A_input_ids=kwargs.get("A_input_ids",[])  
#         print (self.A_input_ids[:10],"idx--->") 
#         print (self.A_input_ids[:10],"idx--->") 
        
        self.A_output =None#kwargs.get("A_output",None)  
        
        self.B_output =None# kwargs.get("B_output",None) 

        if self.B_output  is None :
            assert self.A_input_ids is not None and  len(self.A_input_ids )==len(self.A_input),"expect the {type(self.A_input_ids)} not None"
            self.B_output = self.teacher_outputs[self.A_input_ids]
        pass 
    
    
    def optimize_parameters(self,**kwargs):
        self.student_model.train()
        
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        
        self.forward(**kwargs)      # compute fake images and reconstruction images.
        self.backward_student(**kwargs)             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights

    def backward_student(self,**kwargs):
        params = kwargs["params"]
        
#         self.output_teacher = None
        #loss_real_D = self.criterionGAN(pred_real_tuple[0], True)
        
        self.loss_kd = loss_fn_kd(outputs=self.A_output , 
               labels=self.A_input_lbl, 
               teacher_outputs=self.B_output, 
               params=params,)
        print ("===="*8)
        print (self.B_output.shape,self.A_output.shape,self.A_input_lbl.shape)
        
        pre_vim_lbl =    torch.argmax(self.B_output,dim=-1)
        gt_lbl      =    self.A_input_lbl 
        pre_mic_lbl =    torch.argmax(self.A_output,dim=-1)
        
#         print (pre_vim_lbl[:10])
#         print (gt_lbl[:10])
#         print (pre_mic_lbl[:10])
# #         print (pre1_lbl.shape,gt_lbl.shape,pre2_lbl.shape)
#         print (torch.sum(pre_vim_lbl==gt_lbl),
#                torch.sum(pre_mic_lbl==gt_lbl),
#             len(pre_mic_lbl)
#             ,"-----"*4)

    def forward(self,**kwargs):
        self.A_output = self.student_model .forward(self.A_input)
        
    
    