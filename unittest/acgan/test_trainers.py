
import torch 
from model_stealing.trainers import acgan_trainer,kd_trainer

import torch .nn as nn 
import numpy as np 

class config:
    name="unittest"
    lr=0.01
    beta1=0.1
    gan_mode="lsgan"
    gpu_ids=[]
    isTrain=True 
    checkpoints_dir="./"
    preprocess="non_scale_width"
    alpha= 0.9
    temperature=20
class Mock_g(nn.Module):
    def __init__(self,img_size=32):
        super(Mock_g,self).__init__()
        self.img_size= img_size 
        self.fc=nn.Linear(10,3*self.img_size)
    def forward(self,noise,labels):
        z=noise
        l=labels
        img=torch.randn(z.shape[0],3,self.img_size)

        out=self.fc(z)
        out=out.view((z.shape[0],-1,3))
        out=torch.bmm(out,img)
        out.unsqueeze_(dim=1)
        out=torch.cat([out,out,out],dim=1)
        return out 
    
class Mock_d(nn.Module):
    def __init__(self):
        super(Mock_d,self).__init__()

        self.pool1= nn.MaxPool2d(3)
        self.pool2= nn.MaxPool2d(3)
        self.pool3= nn.MaxPool2d(3)
        
        self.fc_gan=nn.Linear(3,1)
        self.fc=nn.Linear(3,10)
    def forward(self,img):
        x=self.pool1(img)
        x=self.pool2(x)
        x=self.pool3(x)
        
        x=x.view((x.shape[0],-1))
        
        out=self.fc(x)
        out_gan=self.fc_gan(x)
        return out_gan,out 
class Mock_cls(nn.Module):
    def __init__(self):
        super(Mock_cls,self).__init__()

        self.pool1= nn.MaxPool2d(3)
        self.pool2= nn.MaxPool2d(3)
        self.pool3= nn.MaxPool2d(3)
        
        self.fc=nn.Linear(3,10)
    def forward(self,img):
        x=self.pool1(img)
        x=self.pool2(x)
        x=self.pool3(x)
        
        x=x.view((x.shape[0],-1))
        
        out=self.fc(x)
        return out 
    
       
# def test_fake():
#     ex=None
#     try :
#         model_g = Mock_g()
#         model_d = Mock_d()
#         
#         batch_size=4
#         z=torch.randn(batch_size,10)
#         l=torch.randint(0,10,(batch_size,))
#         img=model_g(z,l)
#         
#         ret_gan,ret_lbl = model_d(img=img)
#         
#         assert ret_gan.shape[1]==1,f"ret {ret_gan.shape}"
#         assert ret_lbl.shape[1]==10,f"ret {ret_gan.shape}"
#         
#     except Exception as ex2 :
#         ex=ex2
#         pass 
#     assert ex is None , f"no except should be occured, but get {ex}"
#     
# def test_acgan_trainer():
#     torch.random.manual_seed(1234)
#     opt = config()
#     model_g = Mock_g()
#     model_d = Mock_d()
#     t1= model_g.fc.weight.grad
#     assert t1 is None or t1.mean()==0
# 
#     tr= acgan_trainer.AcganTrainer(
#         netG=model_g,
#         netD=model_d,
#         opt= opt 
#         )
#     print ("start optim")
#     tr.set_input(
#         input=( torch.randn(4,3,32,32),
#                 torch.randint(0,10,(4,)) ),
#         
#         noise=torch.randn(4,10),
#         )
#     tr.optimize_parameters()
#     
#     t1= model_g.fc.weight.grad
#     assert t1.mean()!=0
#     t2= model_d.fc.weight.grad
#     
    
def test_kd_trainer():
    opt = config()
    torch.random.manual_seed(1234)
    model_teacher = Mock_cls()
    torch.random.manual_seed(12345)
    model_student = Mock_cls()
    t1= model_teacher.fc.weight.grad
    assert t1 is None or t1.mean()==0
    
    assert model_teacher.fc.weight.mean() !=model_student.fc.weight.mean()
    
    tr= kd_trainer.KDTrainer(
        victim_net=model_teacher,
        synthenic_net=model_student,
        opt= opt 
        )
    ###### generate mock 
    mock_img_list = torch.randn(4*10,3,32,32)
    mock_lbl_list = torch.randint(0,10,(4*10,))
    
    standard_dataset= (mock_img_list,mock_lbl_list)
    mock_teacher_list = tr.get_teacher_output(
        teacher_model=model_teacher,
        data_loader=standard_dataset,
        )
    mock_teacher_list= np.concatenate(mock_teacher_list)
    mock_teacher_list= torch.from_numpy(mock_teacher_list)
    assert len(mock_teacher_list)==len(mock_img_list),f"expect the teacher return {mock_teacher_list.shape}=1==1={mock_img_list.shape}"

    print ("start optim")
    mock_list = zip(
            torch.split(mock_img_list,10), 
            torch.split(mock_lbl_list,10),
            torch.split(torch.range(0,len(mock_img_list)).long(), 10) , 
            )
    for mock_one in mock_list:
        print (mock_one[2])
        
        print (type(mock_one[2]), )
        tr.set_input(
            A_input=mock_one[0],
            A_input_lbl=mock_one[1],
            
            B_output= mock_teacher_list[mock_one[2].tolist()],
            
            )
        tr.optimize_parameters(params=opt)



