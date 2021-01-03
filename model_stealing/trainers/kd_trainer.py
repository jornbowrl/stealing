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

def loss_fn(outputs, labels, teacher_outputs, params):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    '''
    copy from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py
    '''
    loss  = F.cross_entropy(outputs, labels)
    return  loss
#     alpha = params.alpha
#     T = params.temperature
#     KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
#                              F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
#                * (1. - alpha)
# 
#     return KD_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
to_torch =  lambda x: torch.from_numpy(x) if not torch.is_tensor(x) else x 
to_np =  lambda x: x.cpu().numpy() if  torch.is_tensor(x) else x 

class KDTrainer(BaseTrainer):
 
    def __init__(self,victim_net,synthenic_net,opt,**kwargs):
        BaseTrainer.__init__(self,opt)

        self.criterionIdt = torch.nn.L1Loss()

        
#         self.teacher_model = victim_net#.to(deivce)
        self.net_student = synthenic_net
        
        self.net_student  = self.net_student .to(device)
#         self.set_requires_grad(self.teacher_model,requires_grad=False)
#         self.set_requires_grad(self.net_student,requires_grad=False)
        


#         self.optimizer_G = torch.optim.Adam(self.net_student.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net_student.parameters()), 
                                            lr=opt.lr, betas=(opt.beta1, 0.999))
        
        self.optimizers.append(self.optimizer_G)
        
        
        
        self.teacher_outputs = kwargs.get("teacher_outputs",[]) 

        self.loss_names=["kd"]
        self.visual_names=["A_input"]
        self.model_names=["student"]
 
        self.evalutor =kwargs.get("evalutor",None)
        self.val_dataset =kwargs.get("val_dataset",None)
 
    def set_input(self,**kwargs):

        self.A_input=kwargs.get("A_input",None) 
        self.A_input_lbl=kwargs.get("A_input_lbl",None)  
        self.A_input_ids=kwargs.get("A_input_ids",[])  
#         print (self.A_input_ids[:10],"idx--->") 
        
        self.A_output =None#kwargs.get("A_output",None)  
        
        self.B_output =None# kwargs.get("B_output",None) 

        if self.B_output  is None :
            assert self.A_input_ids is not None and  len(self.A_input_ids )==len(self.A_input),"expect the {type(self.A_input_ids)} not None"
            self.B_output = self.teacher_outputs[self.A_input_ids]
        
        
         
        if self.A_input is not None :
            self.A_input = self.A_input.to(device)
        if self.A_input_lbl is not None :
            self.A_input_lbl = self.A_input_lbl.to(device)
        if self.B_output is not None :
            self.B_output = self.B_output.to(device)
    
    
    def optimize_parameters(self,**kwargs):
        self.train()
        

        
        self.forward(**kwargs)      # compute fake images and reconstruction images.
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_student(**kwargs)             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights

    def backward_student(self,**kwargs):
        params = kwargs["params"]
        
        
#         self.loss_kd = loss_fn(outputs=self.A_output , 
#                labels=self.A_input_lbl, 
#                teacher_outputs=self.B_output, 
#                params=params,)
        self.loss_kd = loss_fn_kd(outputs=self.A_output , 
               labels=self.A_input_lbl, 
               teacher_outputs=self.B_output, 
               params=params,)
        self.loss_kd.backward()
        

    def forward(self,**kwargs):
        self.A_output = self.net_student .forward(self.A_input)

    def evaluate(self):
        self.eval()
        
        student_pred_score = self.evalutor.evaluate(
                                model=self.net_student, 
                                  dataloader=self.val_dataset,
                                   loss_fn=torch.nn.CrossEntropyLoss())
        print(student_pred_score["loss"] if "loss" in student_pred_score else {})
        return student_pred_score 