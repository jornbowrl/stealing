from abc import ABC, abstractmethod
import tqdm 
import logging 

import numpy as np 
import torch 


class Evaluate(ABC):
    
    
    @staticmethod
    def evaluate(model, dataloader, metrics, loss_fn=None ,verbose=True, params=None,
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
    
        # set model to evaluation mode
        model.to(device)
        model.eval()
    
        # summary for current eval loop
        summ = []
        progress=None
        if verbose :
            progress = tqdm.tqdm(total=len(dataloader) ,desc="evaluate")
        with torch.no_grad():
            # compute metrics over the dataset
            for data_one in dataloader:
                data_batch, labels_batch = data_one 
                # move to GPU if available
    #             if params.cuda:
                data_batch, labels_batch = data_batch.to(device), labels_batch.to(device)
                # fetch the next evaluation batch
    #             data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
                
                # compute model output
                output_batch = model(data_batch)
        
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()
        
                if loss_fn is not None :
                    loss = loss_fn(output_batch, labels_batch)
                    summary_batch['loss'] = loss.item()
                # compute all metrics on this batch
                summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summ.append(summary_batch)
        
                if verbose :
                    progress.update(1)
        
        # compute mean of all metrics in summary
        metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
        logging.info(f"- Eval metrics : " + metrics_string)
        return metrics_mean
    
    
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
    

