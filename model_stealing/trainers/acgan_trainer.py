import torch 
from . import util as networks
from .base  import BaseTrainer 



class AcganTrainer(BaseTrainer):
    
    
    def __init__(self,netG,netD, opt,netVictim=None):
        BaseTrainer.__init__(self,opt)

        self.netG=netG
        self.netD=netD
        self.netVictim=netVictim
        
        self.img_syn = None 
        self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
#         self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()
        
        # initialize optimizers; schedulers will be automatically created by function <BaseTrainer.setup>.
#         self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
#         self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)
    
    def set_input(self, input,fake_lbl,noise=None):
        img,lbl =  input
        
        self.real_A_img= img
        self.real_A_lbl= lbl
        
        self.fake_B_lbl= fake_lbl
        self.noise = noise 
#     def fetch_output (self,out,alias="gan"):
#         assert alias in ["gan","auxiliary",0,1]
#         if 
#         return out[0] 
    
    def forward(self,**kwargs):

        self.img_syn= self.netG(noise=self.noise,labels=self.real_A_lbl,**kwargs)
    
    def backward_D_basic(self,netD,real,fake ):
        # Real
        pred_real = netD(real)[0]
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())[0]
        loss_D_fake = self.criterionGAN(pred_fake, False)

        loss_D = (loss_D_real + loss_D_fake) * 0.5
#         loss_D.backward()

        return loss_D    
    
    def backward_D_auxiliary(self,netD,real,fake ):
        # Real
        pred_real_tuple = netD(real)
        loss_real_D = self.criterionGAN(pred_real_tuple[0], True)
        
        # Fake
        pred_fake_tuple = netD(fake.detach())
        loss_fake_D = self.criterionGAN(pred_fake_tuple[0], False)

        
        loss_real_aux = self.criterionIdt (pred_real_tuple[1],self.real_A_lbl)
        loss_fake_aux = self.criterionIdt (pred_fake_tuple[1],self.fake_B_lbl)
        

        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_aux = (loss_real_aux + loss_fake_aux) * 0.5
        
        loss= loss_D+loss_aux
#         loss_D.backward()

        return loss_D

    
    def backward_D(self,**kwargs):
        
        return self.backward_D_basic(self.netD,self.real_A_img,self.img_syn,**kwargs)

    def backward_G(self):

        self.loss_G = self.criterionGAN(self.netD(self.img_syn)[0], True)
        self.loss_G.backward()


    def optimize_parameters(self,**kwargs):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward(**kwargs)      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G(**kwargs)             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D(**kwargs)      # calculate gradients for D_A
#         self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
