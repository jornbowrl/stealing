
#The Model Stealing by knowledge Distillation and Generation Perturbance  

## Install and preparation 
```
## recommend starting from py37
#conda create -n py37 python=3.7 


pip install -r req.txt
```
## HypeParams
* import the victim model from Zoo

```
model_zoo[str] and model_zoo_pretrained [bool]
#https://github.com/osmr/imgclsmob/tree/master/pytorch
```
* customised the structure by nn.Module 

```
model_customised[str], model_customised_args [json_str] and model_customised_pretrained [http_url]
#https://github.com/osmr/imgclsmob/tree/master/pytorch
```
https://intellabs.github.io/distiller/knowledge_distillation.html

## KD
*  A framework for exploring "shallow" and "deep" knowledge distillation (KD) experiments
![img](https://intellabs.github.io/distiller/imgs/knowledge_distillation.png)

* formular 

[<img src="./papers/kd/kd_loss_func.png" width="300" height="50" />]()
[<img src="./papers/kd/kd_loss_func_text.png" width="400" height="35%" />]()
[https://arxiv.org/pdf/1905.09747.pdf](reference)
### finetune

*  edit the hypeparamater
 
```

cat experiments/base_model/params.yaml  
#start the train epoches 
cd model_stealing
sh run.sh 

```

## Params

```

```

## Related Projects
** [ACGAN](https://arxiv.org/abs/1610.09585)

** [Knockoff Nets](https://openaccess.thecvf.com/content_CVPR_2019/papers/Orekondy_Knockoff_Nets_Stealing_Functionality_of_Black-Box_Models_CVPR_2019_paper.pdf) 

** [Twin Auxiliary Classifiers GAN](https://papers.nips.cc/paper/2019/file/4ea06fbc83cdd0a06020c35d50e1e89a-Paper.pdf)

** [MAZE](https://arxiv.org/pdf/2005.03161.pdf) 

** [ES Attack](https://arxiv.org/abs/2009.09560) 





## Acknowledgments


Our code is inspired by [pytorch-cyclegan](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
 

Our code is inspired by [pytorch-KD](https://github.com/peterliht/knowledge-distillation-pytorch).



