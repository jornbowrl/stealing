import sys 
import os 
sys.path.append("../")
# print (sys.path)
# print (os.path.abspath("./"))

import model_stealing
# print (dir(model_stealing))
# from options.train_options import TrainOptions
from model_stealing.dataset.data_loader import create_dataset
from model_stealing.models import get_model
from model_stealing.trainers import create_trainer#,get_victim_output
from model_stealing.util.visualizer import Visualizer

from model_stealing.evaluate import get_evaluator 

from model_stealing.util import util

import time
import math
import logging
import argparse 

import torch 

#set reproduce 
import random 
import numpy as np 
random.seed(1234)     # python random generator
np.random.seed(1234)  # numpy random generator
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True



parser = argparse.ArgumentParser()
# parser.add_argument('--data_dir', default='data/64x64_SIGNS', help="Directory for the dataset")
parser.add_argument('--model_dir', default='../experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir \
                    containing weights to reload before training")  # 'best' or 'train'

if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.yaml')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
#     opt = util.Params(json_path)
    opt = util.Params(json_path,is_json=False)

    setattr(opt.thief.trainer,"name",opt.name)
    trainer_opt=  opt.thief.trainer
    ############ dataset  ############
    train_dataset_shuffle_aug = create_dataset(trainer_opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(train_dataset_shuffle_aug)    # get the number of images in the dataset.
    logging.info('The number of training images = %d' % dataset_size)

    train_dataset_ordered = create_dataset(trainer_opt,is_shuffle=False)  # create a dataset given opt.dataset_mode and other options

    val_dataset_ordered = create_dataset(trainer_opt,is_train=False)  # create a dataset given opt.dataset_mode and other options
    dataset_size_val = len(val_dataset_ordered)    # get the number of images in the dataset.
    logging.info('The number of evaluating images = %d' % dataset_size_val)


    # reload weights from restore_file if specified
    ############ network  ############

    victime_net = get_model(opt.victim.model)
    stealing_net = get_model(opt.thief.model)

    # fetch victim outputs using victim_model under eval() mode
    loading_start = time.time()
    victime_net.eval()
    victim_outputs=[]
    assert len(train_dataset_ordered)==len(train_dataset_shuffle_aug),"expect victim dataset same as general one"
#     victim_pred = get_victim_output(victim_model=victime_net, 
# #                                         fetch_func=lambda x:x["A_input"],
#                                           data_loader=train_dataset_ordered,
#                                            params=trainer_opt)
    ############ evaluate victim on train_dataset ############
    victim_evalutor = get_evaluator(opt.victim.evaluator,namespace="victim_train",return_massive=True)
    victim_pred = victim_evalutor.evaluate(model=victime_net, 
                                          dataloader=train_dataset_ordered,
                                           params=trainer_opt,
                                           loss_fn=torch.nn.CrossEntropyLoss())
    victim_outputs,indx_list = victim_pred["logits"],victim_pred["index"]
    
    
    ############ evaluate victim on test_dataset ############
    test_evalutor = get_evaluator(opt.victim.evaluator,namespace="test",return_massive=False)
    test_evalutor.namespace="victim_test"
    victim_pred_score = test_evalutor.evaluate(model=victime_net, 
                                          dataloader=val_dataset_ordered,
                                           params=trainer_opt,
                                           loss_fn=torch.nn.CrossEntropyLoss())
    test_evalutor.namespace="thief_test"

    loss_victim = victim_pred_score["loss"]
    
#     print ("--->victim--->loss",loss_victim)
    elapsed_time = math.ceil(time.time() - loading_start)
    logging.info("- Finished computing victim outputs after {} secs..".format(elapsed_time))
    ############ evaluate thief  ############
    thief_evalutor = get_evaluator(opt.thief.evaluator)


    ############ trainer thief  ############
    kd_trainer = create_trainer(
        victim_net=victime_net,
        thief_net=stealing_net,
        victim_outputs= victim_outputs,
        opt=trainer_opt,
        evalutor=  test_evalutor,
        val_dataset=val_dataset_ordered,
        )      # create a model given opt.model and other options
    
    kd_trainer.setup(trainer_opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(name=opt.name,opt=trainer_opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    ############ train   ############
    for epoch in range(trainer_opt.epoch_count, trainer_opt.n_epochs + trainer_opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        ############ evalute  before epoch start   ############
        thief_pred_score= kd_trainer.evaluate()
#         print(thief_pred_score["loss"] if "loss" in thief_pred_score else {})

        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        for i, data in enumerate(train_dataset_shuffle_aug):  # inner loop within one epoch
            assert type(data)==dict,f"expect the data is dict {type(data)}"
            iter_start_time = time.time()  # timer for computation per iteration
#             if total_iters % trainer_opt.print_freq == 0:
#                 t_data = iter_start_time - iter_data_time

            total_iters += trainer_opt.batch_size
            epoch_iter += trainer_opt.batch_size
            kd_trainer.set_input(**data)         # unpack data from dataset and apply preprocessing
            kd_trainer.optimize_parameters(params=trainer_opt)   # calculate loss functions, get gradients, update network weights

            if total_iters % trainer_opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = True 
                if len(kd_trainer.visual_names)>0:
                    kd_trainer.compute_visuals()
                    visualizer.display_current_results(kd_trainer.get_current_visuals(), epoch, save_result)

            if total_iters % trainer_opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = kd_trainer.get_current_losses()
                t_comp = (time.time() - iter_start_time) / trainer_opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data=loss_victim)
#                 visualizer.plot_current_losses(float(total_iters) / dataset_size, losses)
#                 visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)


            if total_iters % trainer_opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if trainer_opt.save_by_iter else 'latest'
                kd_trainer.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % trainer_opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            kd_trainer.save_networks('latest')
            kd_trainer.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, trainer_opt.n_epochs + trainer_opt.n_epochs_decay, time.time() - epoch_start_time))
        kd_trainer.update_learning_rate()    # update learning rates in the beginning of every epoch.
        
#         visualizer.flush()
