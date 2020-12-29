


# from .gan_base import Generator,Discriminator 
from torchvision.datasets import utils as dt_util 
import json 
import os 
import torch 
import traceback

from .kd_instill import Net as kd_net 

def get_model(opt_model,device=None):
    if opt_model.model_zoo is not None :
        return get_net_pytorchcv(
            model_name=opt_model.model_zoo,
            device=device,
            pretrained=opt_model.model_zoo_pretrained)
    
    args =opt_model.model_customised_args
    
    try :
        args = json.loads(args) if args is not None else {}
    except :
        traceback.print_exc()
        args ={}
        
    return get_net_customised(
        model_name=opt_model.model_customised,
        device=device,
        pretrained=opt_model.model_customised_pretrained,
        **args,
        )
    

def get_net_pytorchcv(model_name="resnet20_cifar10",device=None,pretrained=True):
    from pytorchcv.model_provider import get_model as ptcv_get_model
    net = ptcv_get_model(model_name, pretrained=pretrained)
    if device is not None :
        net = net.to(device)
    return net     
    
def get_net_customised(model_name="kd_net",
                       device=None,pretrained=None,
                       experiment_cach_dir="~/.cache",**model_args):
    
    if model_name=="kd_net":
        net= kd_net(**model_args)
    else :
        raise Exception(f"Unknow model name ={model_name}")
    if experiment_cach_dir is not None :
        experiment_cach_dir= os.path.expanduser(experiment_cach_dir)
    
    if device is not None :
        net = net.to(device)

    def default_load_func(x,net):
        ckp= torch.load(x,map_location=device)
        #net= net.load_state_dict(ckp)
        return net 
    load_func = model_args.get("load_func", default_load_func )

    if type(pretrained)==str:
        if "://" in pretrained:
            url = pretrained
            
            filename = os.path.basename(url)

            print (dir(dt_util),)
            is_tar = any([
                #dt_util._is_tarxz (url) ,
                dt_util._is_tar (url) ,
                dt_util._is_targz (url) ,
#                 dt_util._is_tgz (url) ,
                dt_util._is_gzip (url) ,
                dt_util._is_zip (url) ]  )
            
            dt_util. download_url(url=url, 
                                  root=experiment_cach_dir, 
                                  filename = None) 
            pretrained = os.path.join(experiment_cach_dir, filename)
            if is_tar:
                
                pretrained_from = os.path.join(experiment_cach_dir, filename)
                pretrained_to = os.path.join(experiment_cach_dir, filename+".pth")
                
                
                dt_util.extract_archive(
                    from_path=pretrained_from,
                    to_path=pretrained_to,
                    )
                
                pretrained= pretrained_to
            
        elif "/" not in pretrained :
            
            fileid=pretrained
            experiment_cach_dir = os.path.expanduser(experiment_cach_dir)
        

            dt_util.download_file_from_google_drive(pretrained, experiment_cach_dir, None )
            
            pretrained = os.path.join(experiment_cach_dir, fileid)

        elif "/" in pretrained:
            pretrained= os.path.normpath(pretrained)
            pretrained= os.path.expanduser(pretrained)
            assert   os.path.isfile(pretrained), f"expect the file {pretrained}"
        else :
            raise Exception("unknow path {pretrained}")

#     elif type(pretrained)==bool:
        print (f"will load from {pretrained}")
        net=load_func(pretrained,net )
        
        return net 
            
    return net 
    

if __name__=="__main__":
    file_list = [
        "/home/malei/.torch/models/alexnet-1824-8ada73bf.pth",
        "~/.torch/models/alexnet-1824-8ada73bf.pth",
        "https://github.com/osmr/imgclsmob/releases/download/v0.0.163/resnet20_cifar10-0597-9b0024ac.pth.zip",
        "1g8Oygp6U98-A-763g58hpknYlAE1huOeKQC1jP4Hdvs",
        ]
    from  torchvision. models import resnet  
    model = resnet.resnet18()
    
    for f in file_list:
        print (f)
        ret=  get_net(model_name="kd_net",net=model,device=torch.device("cuda"),pretrained=f,)
    
    