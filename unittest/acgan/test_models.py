'''
Created on Dec 21, 2020

@author: a11
'''
import torch 
from model_stealing.models import arch  as acgan_model

def test_model():
    batch_size=4
    noise= torch.randn(batch_size,100)
    label= torch.randint(0,10,[batch_size,])
    
    g= acgan_model.Generator()
    img_syn=g.forward(
        noise=noise, 
        labels=label
              )
    assert img_syn.shape[1]==3, f"expect generate a new 3-channel image, but {img_syn.shape} "
    
    
    d= acgan_model.Discriminator()
     
    ret1,ret2 = d.forward(img=img_syn)
    
    assert ret1.shape[1]==1, f"expect discrimate return 1-0 , but {ret1.shape} "
    assert ret2.shape[1]==10, f"expect discrimate auxiliary return 10 channel , but {ret2.shape} "
