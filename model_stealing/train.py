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
from model_stealing.trainers import create_trainer,get_teacher_output
from model_stealing.util.visualizer import Visualizer

from model_stealing.util import util

import time
import math
import logging
import argparse 


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

#     opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt.student.trainer)  # create a dataset given opt.dataset_mode and other options
    teacher_dataset = create_dataset(opt.student.trainer,is_shuffle=False)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    logging.info('The number of training images = %d' % dataset_size)

    dataset_val = create_dataset(opt.student.trainer,is_train=False)  # create a dataset given opt.dataset_mode and other options
    dataset_size_val = len(dataset_val)    # get the number of images in the dataset.
    logging.info('The number of evaluating images = %d' % dataset_size_val)


    trainer_opt=  opt.student.trainer
    # reload weights from restore_file if specified

    victime_net = get_model(opt.teacher.model)
    stealing_net = get_model(opt.student.model)

    # fetch teacher outputs using teacher_model under eval() mode
    loading_start = time.time()
    victime_net.eval()
    teacher_outputs=[]
    assert len(teacher_dataset)==len(dataset),"expect teacher dataset same as general one"
    teacher_outputs,indx_list = get_teacher_output(teacher_model=victime_net, 
#                                         fetch_func=lambda x:x["A_input"],
                                          data_loader=teacher_dataset,
                                           params=trainer_opt)
    
    print ("--->teacher--->",indx_list[:10])
    elapsed_time = math.ceil(time.time() - loading_start)
    logging.info("- Finished computing teacher outputs after {} secs..".format(elapsed_time))


    kd_trainer = create_trainer(
        victim_net=victime_net,
        synthenic_net=stealing_net,
        teacher_outputs= teacher_outputs,
        opt=opt.student.trainer)      # create a model given opt.model and other options
    
    kd_trainer.setup(opt.student.trainer)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(trainer_opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(trainer_opt.epoch_count, trainer_opt.n_epochs + trainer_opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        kd_trainer.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            assert type(data)==dict,f"expect the data is dict {type(data)}"
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % trainer_opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

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
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
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
        
#         visualizer.flush()
