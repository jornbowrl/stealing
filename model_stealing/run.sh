


echo `pwd`

#ls `pwd`/model_stealing
cd `pwd`

CUDA_VISIBLE_DEVICES=0 nohup /home/malei/anaconda3/envs/tf13/bin/python   train.py  --model_dir ../experiments/res18_v1 2>&1 >./v1.log &
CUDA_VISIBLE_DEVICES=1  nohup /home/malei/anaconda3/envs/tf13/bin/python   train.py  --model_dir ../experiments/res18_v2 2>&1 >./v2.log &
CUDA_VISIBLE_DEVICES=2  nohup /home/malei/anaconda3/envs/tf13/bin/python   train.py  --model_dir ../experiments/res18_v3 2>&1 >./v3.log &
CUDA_VISIBLE_DEVICES=3  nohup /home/malei/anaconda3/envs/tf13/bin/python   train.py  --model_dir ../experiments/res18_v4 2>&1 >./v4.log &



