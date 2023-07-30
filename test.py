
import torch
import torch.nn as nn
import torch.nn.init as init
from sklearn.metrics import roc_curve
from torch import optim
from torch.utils.data import DataLoader
from losses import LOSS_FUNCTIONS
from log import LOG
from Encoder.deal_data import MolGraphDataset, molgraph_collate_fn
# from Result_vis import vis_loss, vis_roc
from Model.MVA import MVA
import argparse
import numpy as np
import pandas as pd
from log import all_evaluate

common_args_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)

common_args_parser.add_argument('--train-set', type=str, default='./Data/v1.csv', help='Training dataset path')
common_args_parser.add_argument('--valid-set', type=str, default='./Data/v1.csv', help='Validation dataset path')
common_args_parser.add_argument('--test-set', type=str, default='./Data/v1.csv', help='Testing dataset path')

# common_args_parser.add_argument('--train-set', type=str, default='./Data/atc.csv', help='Training dataset path')
# common_args_parser.add_argument('--valid-set', type=str, default='./Data/atc.csv', help='Validation dataset path')
# common_args_parser.add_argument('--test-set', type=str, default='./Data/atc.csv', help='Testing dataset path')

# common_args_parser.add_argument('--train-set', type=str, default='./Data/MVADDI_train.csv', help='Training dataset path')
# common_args_parser.add_argument('--valid-set', type=str, default='./Data/MVADDI_valid.csv', help='Validation dataset path')
# common_args_parser.add_argument('--test-set', type=str, default='./Data/MVADDI_test.csv', help='Testing dataset path')

common_args_parser.add_argument('--loss', type=str, default='CrossEntropy', choices=[k for k, v in LOSS_FUNCTIONS.items()])
common_args_parser.add_argument('--score', type=str, default='All', help='roc-auc or MSE or All')
common_args_parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
common_args_parser.add_argument('--batch-size', type=int, default=256, help='Number of graphs in a mini-batch')
common_args_parser.add_argument('--learn-rate', type=float, default=0.0001) #0.001-inf
common_args_parser.add_argument('--savemodel', action='store_true', default=False, help='Saves model with highest validation score')
common_args_parser.add_argument('--logging', type=str, default='less')
common_args_parser.add_argument('--gcn_in_size', type=int, default=75, help='gcn input size')
common_args_parser.add_argument('--gcn_out_size', type=int, default=128, help='gcn output size')
common_args_parser.add_argument('--random_factor', type=bool, default=False, help='Whether to fix the random factor')


class EarlyStopping:
    def __init__(self, patience=10, delta=0.01):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop


def main():
    global args
    args = common_args_parser.parse_args()
    print(args)

    # train_dataset = MolGraphDataset(args.train_set)
    # train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
    #                               shuffle=True, collate_fn=molgraph_collate_fn)
    #
    # validation_dataset = MolGraphDataset(args.valid_set)
    # validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size,
    #                                    shuffle=False, collate_fn=molgraph_collate_fn)

    test_dataset = MolGraphDataset(args.test_set)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=True, collate_fn=molgraph_collate_fn)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.LayerNorm):
            init.normal_(m.weight, mean=1.0, std=0.02)
            init.constant_(m.bias, 0.0)

    net = torch.load('Out/MAV-DDI.pt', map_location=torch.device('cpu'))

    if args.random_factor:
        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        net.apply(init_weights)

    print('----------------')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    net.eval()
    for i_batch, batch in enumerate(test_dataloader):
        ft1, adj1, ft2, adj2, num_size1, target1, d11, d21, mask_1, mask_2 = batch
        # ft1 = ft1.cuda()
        # adj1 = adj1.cuda()
        # ft2 = ft2.cuda()
        # adj2 = adj2.cuda()
        # num_size1 = num_size1.cuda()
        # d11 = d11.cuda()
        # d21 = d21.cuda()
        output, _ = net(ft1, adj1, ft2, adj2, num_size1, d11, d21)
        scores = torch.sigmoid(output)
        # print(scores)
        # scores = scores * 10000
        # print(scores)
        # scores = torch.round(scores)
        # print(scores)
        # scores = scores / 10000
        print(scores)
        # F1, accuracy, recall, precision, auroc, aupr = all_evaluate(output, target1)
        # print("test answer: acc {}, auc {}".format(accuracy, auroc))


if __name__ == '__main__':
    main()
