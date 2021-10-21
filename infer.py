import sys

import cv2
from tqdm import tqdm
from argparse import ArgumentParser
import torch.nn
#from torchvision.models import vgg16
from torch.autograd import Variable  

from models.network import get_networks
from utils.utils import get_config


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('save_file',type=str,default='None',help='save_file')
    parser.add_argument('--config', type=str, default='configs/jier_config.yaml', help='infer cfg')
    parser.add_argument('--imglist',type=str,default='None',help='imglist')
    parser.add_argument('--checkpoint',type=str,default='None',help='checkpoint')

    return parser.parse_args()

def infer_(model, vgg, config, lst_file, save_file):
    f2 = open(save_file,'w')
    lamda = config['lamda']
    similarity_loss = torch.nn.CosineSimilarity()
    direction_only = config['direction_loss_only']
    model.eval()
    with open(lst_file) as f:
        for line in tqdm(f.readlines()):
            img, label = line.strip().split(' ')
            try:
                X = cv2.imread(img,cv2.IMREAD_COLOR)
                X = cv2.resize(X, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
            except:
                print(img)
                continue
            X = torch.from_numpy(X)
            X = X.float()
            X = Variable(X).cuda()
            X = X.permute(2,0,1)
            X = X.unsqueeze(0)
            output_real = vgg(X)
            output_pred = model.forward(X)

            y_pred_1, y_pred_2, y_pred_3 = output_pred[6], output_pred[9], output_pred[12]
            y_1, y_2, y_3 = output_real[6], output_real[9], output_real[12]
            if direction_only:
                loss_1 = 1 - similarity_loss(y_pred_1.view(1, -1), y_1.view(1, -1))
                loss_2 = 1 - similarity_loss(y_pred_2.view(1, -1), y_2.view(1, -1))
                loss_3 = 1 - similarity_loss(y_pred_3.view(1, -1), y_3.view(1, -1))
                total_loss = loss_1 + loss_2 + loss_3
            else:
                abs_loss_1 = torch.mean((y_pred_1 - y_1) ** 2)
                loss_1 = 1 - similarity_loss(y_pred_1.view(1, -1), y_1.view(1, -1))
                abs_loss_2 = torch.mean((y_pred_2 - y_2) ** 2)
                loss_2 = 1 - similarity_loss(y_pred_2.view(1, -1), y_2.view(1, -1))
                abs_loss_3 = torch.mean((y_pred_3 - y_3) ** 2)
                loss_3 = 1 - similarity_loss(y_pred_3.view(1, -1), y_3.view(1, -1))
                total_loss = loss_1 + loss_2 + loss_3 + lamda * (abs_loss_1 + abs_loss_2 + abs_loss_3)
            total_loss = total_loss.cuda().data.cpu().numpy()[0]
            f2.write(img + ' ' + str(label) + ' ' + str(total_loss) + '\n')

def main():
    args = get_parser()
    config = get_config(args.config)
    checkpoint = args.checkpoint
    vgg, model = get_networks(config)
    model.load_state_dict(
        torch.load(checkpoint)
    )
    lst_file = args.imglist
    save_file = args.save_file
    infer_(model, vgg, config, lst_file, save_file)

if __name__=='__main__':
    main()


