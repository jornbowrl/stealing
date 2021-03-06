import torch 
from .kd_trainer import KDTrainer


def create_trainer(opt,**kwargs):
    if opt.class_name=="KDTrainer":
        return KDTrainer(opt=opt,**kwargs)
    
    raise Exception(f"unknow trainer name {opt.name}")





# def get_teacher_output(teacher_model=None,data_loader=None,fetch_func=lambda x:(x["A_input"],x["A_input_ids"],x["A_input_lbl"]),
#                         device= torch.device("cuda" if torch.cuda.is_available() else "cpu"),
#                        random_state=123,**kwargs ):
#     def to_dataloader(dataset,batch_size=512,):
#         
#         if not hasattr(dataset,"batch_size"):
#             if type(dataset) in [tuple,list]:
#                 dataset=[to_torch(x) for x in dataset]
#             elif type(dataset) ==np.ndarray :
#                 dataset = [to_torch(x) ]
#             dataset =torch.utils.data.TensorDataset(*dataset)
#             
#             return  torch.utils.data.DataLoader(dataset,batch_size=batch_size)
#         return dataset 
#     
#     teacher_outputs_list=[]
#     teacher_outputs_idx_list=[]
#     
#     data_loader = to_dataloader(data_loader)
#     loss_fn = torch.nn.CrossEntropyLoss()
#     
#     loss_list = []
#     with torch.no_grad():
#         teacher_model= teacher_model.to(device)
#         for data in data_loader:
#             img,idx,lbl = fetch_func(data)
#             img  = img.to(device)
#             lbl  = lbl.to(device) 
#             
#             logits =   teacher_model.forward(img)
#             
#             teacher_outputs_list.append(logits.cpu())
#             teacher_outputs_idx_list.append(idx.cpu())
#             
#             loss=loss_fn(logits,lbl)
#             loss_list.append(loss.item())
#             
#             
#     teacher_outputs_list = torch.cat(teacher_outputs_list)
#     teacher_outputs_idx_list = torch.cat(teacher_outputs_idx_list)
#     teacher_outputs_idx_list = teacher_outputs_idx_list.long()
#     
#     loss_mean = sum(loss_list)/float(len(loss_list))
#     
#     print ("loss-->",loss_mean)
#     assert torch.all(torch.eq(teacher_outputs_idx_list, torch.range(0,len(teacher_outputs_idx_list)-1).long())) ,f"{teacher_outputs_idx_list[:10]} == {torch.range(0,len(teacher_outputs_idx_list)-1).long()[:10]}"
#     assert len(teacher_outputs_idx_list)==len(teacher_outputs_list),"the index len same as len of output, but {len(teacher_outputs_idx_list)}!={len(teacher_outputs_list)}"
#     return {
#         "logits":teacher_outputs_list ,
#         "index":teacher_outputs_idx_list,
#         "loss":loss_mean
#         }
    