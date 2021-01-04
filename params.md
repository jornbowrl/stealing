# The usage of hyperparameter

* Objective : 

  * reproduce the best score's procedge  
   * mitigate the duplicated stuff of train
   * note the fragil hyperparameter with evidence
  


* How to organise the structure
 * we keep three sections in our configuration 
   * the metainfo of experiment.
   * the evaluating and distilling section of victim.
   * the thief section reuse Model/Evaluator and new the trainer sub-section to run the train event. 
   
## Metainfo section 

 1. name

> you'd better use an unique name to distingush the best score   

## Victim section 
 1. model
	* model_zoo: 
	
	> can directly import a pretrained model from third-party liberary, eg.  [pytorchcv](https://github.com/osmr/imgclsmob/tree/master/pytorch)
	
	* "model_zoo_pretrained": (bool)
	
	>  whether load the pretrained into memory

	* "model_customised": (str)
	
	>  import the customised pytorch Module from code inline
	
	* "model_customised_pretrained": (str)
	
	>  the path of pretrained, no matter of http/unix-path
	
 2. evaluator 
  * class_name (str)
  
  > The python-inline className to be dynamically imported,   
  
  * metrics (dictionary)
  
  > The k-v measure of prediction and groundtruth ,eg {"accuracy": func, "in_consistency_rate":func}
  
  > more metrics
  

## Thieft section 
 1. model
> same as victim section 
 
 2. evaluator
> same as victim section
 
 3. trainer
 
	* class_name (str)
	
	* transform_name
	 preprocessing of the dataset 
>  alernative one of them or override by customised
> 
> (default) transform_cifar_non_normalize{ mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]}
>
> transform_imagenet { mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]}
> 
> transform_cifar { mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]}

	* dataset_name
>  support the ["cifar10","cifar100",], imagenet in future

	* checkpoints_dir 
>  models are saved here

	* beta1 [<span style="color:red">finetune<span>]
>  support the ["cifar10","cifar100",], imagenet in future

	* lr [<span style="color:red">finetune<span>]
>  initial learning rate for adam

	* lr_policy [<span style="color:red">finetune<span>]
>  learning rate policy. [linear | step | plateau | cosine]

	* lr_decay_iters [<span style="color:red">finetune<span>]
>  multiply by a gamma every lr_decay_iters iterations

	* alpha [<span style="color:red">finetune<span>]
>  the weight between soft_loss and hard_loss, more detail [link](http://cs230.stanford.edu/files_winter_2018/projects/6940224.pdf)

	* temperature [<span style="color:red">finetune<span>]
>  temperature value of KD-distill (see the paper)

       
