import torch 
from .kd_trainer import KDTrainer


def create_trainer(opt,**kwargs):
    if opt.name=="KDTrainer":
        return KDTrainer(opt=opt,**kwargs)
    
    raise Exception(f"unknow trainer name {opt.name}")





def get_teacher_output(teacher_model=None,data_loader=None,fetch_func=lambda x:(x[0],x[1]),
                        device= torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                       random_state=123,**kwargs ):
    def to_dataloader(dataset,batch_size=512,):
        
        if not hasattr(dataset,"batch_size"):
            if type(dataset) in [tuple,list]:
                dataset=[to_torch(x) for x in dataset]
            elif type(dataset) ==np.ndarray :
                dataset = [to_torch(x) ]
            dataset =torch.utils.data.TensorDataset(*dataset)
            
            return  torch.utils.data.DataLoader(dataset,batch_size=batch_size)
        return dataset 
    
    teacher_outputs_list=[]
    
    data_loader = to_dataloader(data_loader)
    with torch.no_grad():
        teacher_model= teacher_model.to(device)
        for data in data_loader:
            img = fetch_func(data)
            img  = img.to(device)
            
            logits =   teacher_model.forward(img)
            
            teacher_outputs_list.append(logits.cpu())
    teacher_outputs_list = torch.cat(teacher_outputs_list)
    
    return teacher_outputs_list 
    