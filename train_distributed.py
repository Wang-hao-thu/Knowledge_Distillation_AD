import os
from test import *
from utils.utils import *
from dataloader import *
from infer import infer_
from pathlib import Path
from torch.autograd import Variable
import pickle
from test_functions import detection_test
from loss_functions import *
from tools.vis_result import get_result
from mmcv.runner import init_dist, get_dist_info
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
import torch.distributed as dist
parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml', help="training configuration")
parser.add_argument('--launcher',default=None,choices=['None','slurm','pytorch'],help='job launcher')



def train(config,launcher=None):
    if launcher == None:
        distributed = False
    else:
        distributed = True
        init_dist(launcher,**config['dist_params'])
        rank,world_size = get_dist_info()
        print(rank)
        config['gpu_ids'] = range(world_size)


    direction_loss_only = config["direction_loss_only"]
    normal_class = config["normal_class"]
    learning_rate = float(config['learning_rate'])
    num_epochs = config["num_epochs"]
    lamda = config['lamda']
    continue_train = config['continue_train']
    last_checkpoint = config['last_checkpoint']

    checkpoint_path = "./outputs/{}/{}/checkpoints/".format(config['experiment_name'], config['save_name'])

    # create directory
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    train_dataloader, test_dataloader = load_data(config,launcher)
    
    print('load data success---',flush=True)
    if continue_train:
        vgg, model = get_networks(config, load_checkpoint=True)
    else:
        vgg, model = get_networks(config)
    if distributed:

         print('load model dist-',flush=True)
         model = MMDistributedDataParallel(
             model.cuda(),
             device_ids = [torch.cuda.current_device()],
             broadcast_buffers=False,
             find_unused_parameters=False
         )
    else:
        model = MMDataParallel(
             model.cuda(0,device_ids = range(1)))
    #device_ids = [0,1,2,3,4,5,6,7]
    #model = torch.nn.DataParallel(model,device_ids=device_ids)
    # Criteria And Optimizers
    print('load model success---',flush=True)
    if direction_loss_only:
        criterion = DirectionOnlyLoss()
    else:
        criterion = MseDirectionLoss(lamda)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if continue_train:
        optimizer.load_state_dict(
            torch.load('{}Opt_{}_epoch_{}.pth'.format(checkpoint_path, normal_class, last_checkpoint)))

    losses = []
    roc_aucs = []
    if continue_train:
        with open('{}Auc_{}_epoch_{}.pickle'.format(checkpoint_path, normal_class, last_checkpoint), 'rb') as f:
            roc_aucs = pickle.load(f)

    for epoch in range(num_epochs + 1):
        model.train()
        epoch_loss = 0
        for data in train_dataloader:
            X = data[0]
            if X.shape[1] == 1:
                X = X.repeat(1, 3, 1, 1)
            X = Variable(X).cuda()

            output_pred = model.forward(X)
            output_real = vgg(X)

            total_loss = criterion(output_pred, output_real)
            
            #total_loss = total_loss.data().clone()
            # Add loss to the list
            dist.all_reduce(total_loss.div_(world_size))
            epoch_loss += total_loss.item()
            losses.append(total_loss.item())

            # Clear the previous gradients
            optimizer.zero_grad()
            # Compute gradients
            total_loss.backward()
            # Adjust weights
            optimizer.step()

        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, epoch_loss),flush=True)
        if epoch % 5 == 1:
            save_file = 'tmp.txt'
            lst_file = os.path.join('Dataset',config['dataset_name'],'test.lst')
            infer_(model, vgg, config, lst_file, save_file)
            #get_result(save_file)
            #roc_auc = detection_test(model, vgg, test_dataloader, config)
            #roc_aucs.append(roc_auc)
            #print("RocAUC at epoch {}:".format(epoch), roc_auc)
        #os.system('rm -rf tmp.txt')
        if epoch % 5 == 1:
            torch.save(model.state_dict(),
                       '{}Cloner_{}_epoch_{}.pth'.format(checkpoint_path, normal_class, epoch))
            torch.save(optimizer.state_dict(),
                       '{}Opt_{}_epoch_{}.pth'.format(checkpoint_path, normal_class, epoch))
            with open('{}Auc_{}_epoch_{}.pickle'.format(checkpoint_path, normal_class, epoch),
                      'wb') as f:
                pickle.dump(roc_aucs, f)

    dist.barrier()
def main():
    args = parser.parse_args()
    config = get_config(args.config)
    config['dist_params']={'backend':'nccl'}
    launcher = args.launcher
    train(config,launcher)


if __name__ == '__main__':
    main()
