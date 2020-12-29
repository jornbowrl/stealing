#The Model Stealing by knowledge Distillation and Generation Perturbance  

## Install and preparation 
```
## recommend starting from py37
#conda create -n py37 python=3.7 


pip install -r req.txt
```

## KD
* train with hyperparameter  

* 1, edit the params
 
```

cat experiments/base_model/params.yaml  
# please 
```

* 2, start the train epoches 

```
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



