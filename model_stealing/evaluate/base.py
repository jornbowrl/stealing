from abc import ABC, abstractmethod
import tqdm 
import logging 

import numpy as np 
import torch 


def accuracyBase(outputs, labels,argmax_dim=-1):
    """
    Compute the accuracy, given the outputs and labels for all images.
    Args:
        outputs: (np.ndarray) output of the model
        labels: (np.ndarray) [0, 1, ..., num_classes-1]
    Returns: (float) accuracy in [0,1]
    """
    pred=torch.argmax(outputs,dim=argmax_dim)
#     outputs = np.argmax(outputs, axis=1)
    acc=torch.eq(pred,labels).float().mean()
    
    return float(acc) 
    return np.sum(outputs==labels)/float(labels.size)


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metricsBase = {
    'accuracy': accuracyBase,
    # could add more metrics such as accuracy for each token type
}
class EvaluateBase(ABC):

    def __init__(self,namespace="teacher",return_massive=True):
        '''
        namespace : prefix of k:v 
        return_massive : True/False 
            True -- > the logits and index to be returned 
            False --> otherwise 
        '''
        self.namespace=namespace
        self.return_massive= return_massive
        
    @staticmethod
    def to_dataloader(dataset,batch_size=512,):
            
        if not hasattr(dataset,"batch_size"):
            if type(dataset) in [tuple,list]:
                dataset=[to_torch(x) for x in dataset]
            elif type(dataset) ==np.ndarray :
                dataset = [to_torch(x) ]
            dataset =torch.utils.data.TensorDataset(*dataset)
            
            return  torch.utils.data.DataLoader(dataset,batch_size=batch_size)
        return dataset 
    

    
    def evaluate(self,model, dataloader, metrics=None, loss_fn=None ,
                 fetch_func=lambda x:(x["A_input"],x["A_input_lbl"],x["A_input_ids"])
                 ,verbose=True, params=None,
                 device= torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 **kwargs):
        """Evaluate the model on `num_steps` batches.
        Args:
            model: (torch.nn.Module) the neural network
            loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
            dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
            metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
            params: (Params) hyperparameters
            num_steps: (int) number of batches to train on, each of size params.batch_size
        """
        if metrics is None :
            metrics = metricsBase

        dataloader = self.to_dataloader(dataloader)

        
        # set model to evaluation mode
        model.to(device)
        model.eval()
    
        # summary for current eval loop
        summ = []
        progress=None
        if verbose :
            progress = tqdm.tqdm(total=len(dataloader) ,desc=f"{self.namespace} evaluate")

        teacher_outputs_list=[]
        teacher_outputs_idx_list=[]
    

        with torch.no_grad():
            # compute metrics over the dataset
            for data_one in dataloader:
                data_batch, labels_batch,idx  =fetch_func( data_one )
                # move to GPU if available
    #             if params.cuda:
                data_batch, labels_batch = data_batch.to(device), labels_batch.to(device)
                # fetch the next evaluation batch
    #             data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
                
                # compute model output
                output_batch = model(data_batch)
        
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu()#.numpy()
                labels_batch = labels_batch.data.cpu()#.numpy()
        
                # compute all metrics on this batch
                summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                
                if loss_fn is not None :
                    loss = loss_fn(output_batch, labels_batch)
                    summary_batch['loss'] = loss.item()
                
                summ.append(summary_batch)
        
                if verbose :
                    progress.update(1)

                if self.return_massive:
                    teacher_outputs_list.append(output_batch.cpu())
                    teacher_outputs_idx_list.append(idx.cpu())

        # compute mean of all metrics in summary
#         metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
        metrics_mean = {f"{self.namespace}_{metric}":np.mean([x[metric] for x in summ]) for metric in summ[0]} 
        
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
        logging.info(f"- Eval metrics : " + metrics_string)
        print ("------------------------------------")
        print(f"- Eval metrics : " + metrics_string)

        if self.return_massive:
            teacher_outputs_list = torch.cat(teacher_outputs_list)
            teacher_outputs_idx_list = torch.cat(teacher_outputs_idx_list)
            teacher_outputs_idx_list = teacher_outputs_idx_list.long()
    
            assert torch.all(torch.eq(teacher_outputs_idx_list, torch.arange(0,len(teacher_outputs_idx_list)).long())) ,f"{teacher_outputs_idx_list[:10]} == {torch.range(0,len(teacher_outputs_idx_list)-1).long()[:10]}"
            assert len(teacher_outputs_idx_list)==len(teacher_outputs_list),"the index len same as len of output, but {len(teacher_outputs_idx_list)}!={len(teacher_outputs_list)}"

#         return metrics_mean
        return {
            "logits":teacher_outputs_list ,
            "index":teacher_outputs_idx_list,
            "loss":metrics_mean,
            }
    
if __name__=="__main__":
    import sys 
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    from pytorchcv.model_provider import get_model as ptcv_get_model
    import torchvision 
    
    net = ptcv_get_model("resnet20_cifar10", pretrained=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    cifar_dl = torchvision.datasets.cifar.CIFAR10(root="~/.torch", train=False, 
                                                  transform=torchvision.transforms.ToTensor())
    cifar_dl = torch.utils.data.DataLoader(cifar_dl,batch_size=64)
    
    
    def accuracy(outputs, labels):
        outputs = np.argmax(outputs, axis=1)
        return np.sum(outputs==labels)/float(labels.size)


    val= Evaluate.evaluate(model=net, loss_fn=loss_fn, dataloader=cifar_dl,
                       metrics={"acc":accuracy}, verbose=True)

    print (type(val),val)
    

