
import torch
import torch.nn as nn
import torch.nn.init as init
from sklearn.metrics import roc_curve
from torch import optim
from torch.utils.data import DataLoader
from losses import LOSS_FUNCTIONS
from log import LOG
from Encoder.deal_data import MolGraphDataset, molgraph_collate_fn
from Result_vis import vis_loss, vis_roc
from Model.MVA import MVA
import argparse
import numpy as np
import pandas as pd

common_args_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)

# common_args_parser.add_argument('--train-set', type=str, default='./Data/v2.csv', help='Training dataset path')
# common_args_parser.add_argument('--valid-set', type=str, default='./Data/v2.csv', help='Validation dataset path')
# common_args_parser.add_argument('--test-set', type=str, default='./Data/v2.csv', help='Testing dataset path')

# common_args_parser.add_argument('--train-set', type=str, default='./Data/atc.csv', help='Training dataset path')
# common_args_parser.add_argument('--valid-set', type=str, default='./Data/atc.csv', help='Validation dataset path')
# common_args_parser.add_argument('--test-set', type=str, default='./Data/atc.csv', help='Testing dataset path')

common_args_parser.add_argument('--train-set', type=str, default='./Data/MVADDI_train.csv', help='Training dataset path')
common_args_parser.add_argument('--valid-set', type=str, default='./Data/MVADDI_valid.csv', help='Validation dataset path')
common_args_parser.add_argument('--test-set', type=str, default='./Data/MVADDI_test.csv', help='Testing dataset path')

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

    train_dataset = MolGraphDataset(args.train_set)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, collate_fn=molgraph_collate_fn)

    validation_dataset = MolGraphDataset(args.valid_set)
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size,
                                       shuffle=False, collate_fn=molgraph_collate_fn)

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

    net = MVA(gcn_in_features=args.gcn_in_size, gcn_out_features=args.gcn_out_size)

    # net = torch.load("atc5.pt")
    # net = torch.load('save.pt', map_location=torch.device('cpu'))

    if args.random_factor:
        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        net.apply(init_weights)

    optimizer = optim.Adam(net.parameters(), lr=args.learn_rate)
    loss_history = []
    early_stopping_obj = EarlyStopping(patience=10, delta=0.01)
    now_epoch = 0
    print(net.parameters())
    print('----------------')
    criterion = LOSS_FUNCTIONS[args.loss]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

# only test

    # net.eval()
    # for i_batch, batch in enumerate(test_dataloader):
    #     ft1, adj1, ft2, adj2, num_size1, target1, d11, d21, mask_1, mask_2 = batch
    #     ft1 = ft1.cuda()
    #     adj1 = adj1.cuda()
    #     ft2 = ft2.cuda()
    #     adj2 = adj2.cuda()
    #     num_size1 = num_size1.cuda()
    #     d11 = d11.cuda()
    #     d21 = d21.cuda()
    #     output, _ = net(ft1, adj1, ft2, adj2, num_size1, d11, d21)
        # print(torch.sigmoid(output))

# training
    for epoch in range(args.epochs):
        net.train()
        batch_losses = []

        for i_batch, batch in enumerate(train_dataloader):
            ft1, adj1, ft2, adj2, num_size1, target1, d11, d21, mask_1, mask_2 = batch

            ft1 = ft1.cuda()
            adj1 = adj1.cuda()
            ft2 = ft2.cuda()
            adj2 = adj2.cuda()
            num_size = num_size1.cuda()
            target = target1.cuda()
            d1 = d11.cuda()
            d2 = d21.cuda()
            optimizer.zero_grad()
            output, _ = net(ft1, adj1, ft2, adj2, num_size, d1, d2)
            # output = output.cpu()
            loss = criterion(output, target)
            batch_losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(), 5.0)
            optimizer.step()

        print("train losss : {}".format(np.mean(batch_losses)))
        loss_history.append(np.mean(batch_losses))
        print('epoch {}, train finish!'.format(epoch + 1))
        print('----------------')
        with torch.no_grad():
            net.eval()

            # early_stopping
            batch_outputs = []
            batch_losses = []
            batch_targets = []
            for i_batch, batch in enumerate(validation_dataloader):
                ft1, adj1, ft2, adj2, num_size1, target1, d11, d21, mask_1, mask_2 = batch
                ft1 = ft1.cuda()
                adj1 = adj1.cuda()
                ft2 = ft2.cuda()
                adj2 = adj2.cuda()
                num_size = num_size1.cuda()
                # target = target1.cuda()
                d1 = d11.cuda()
                d2 = d21.cuda()
                output, _ = net(ft1, adj1, ft2, adj2, num_size, d1, d2)
                output = output.cpu()
                loss = criterion(output, target1)
                batch_outputs.append(output)
                batch_losses.append(loss.item())
                batch_targets.append(target1)

            val_loss = np.mean(batch_losses) #average loss
            now_epoch = epoch + 1
            print("val_loss : {} ".format(val_loss))
            is_early_stop = early_stopping_obj(val_loss)
            if is_early_stop:
                vis_loss(epoch+1, loss_history)
                print("Early stopping in epoch:{}".format(epoch+1))
                output_history = []
                label_history = []
                batch_outputs = []
                batch_targets = []
                for i_batch, batch in enumerate(test_dataloader):
                    ft1, adj1, ft2, adj2, num_size1, target1, d11, d21, mask_1, mask_2 = batch
                    ft1 = ft1.cuda()
                    adj1 = adj1.cuda()
                    ft2 = ft2.cuda()
                    adj2 = adj2.cuda()
                    num_size = num_size1.cuda()
                    # target = target1.cuda()
                    d1 = d11.cuda()
                    d2 = d21.cuda()
                    output, output_feature = net(ft1, adj1, ft2, adj2, num_size, d1, d2)
                    output_feature = output_feature.cpu()

                    batch_outputs.append(output)
                    batch_targets.append(target1)

                    output_history.append(output_feature.numpy())
                    label_history.append(target1.numpy())

                # np.save("output_features50_att.npy", output_history)
                # np.save("output_labels50_att.npy", label_history)

                outputs = torch.cat(batch_outputs).cpu()
                targets = torch.cat(batch_targets).cpu()
                scores = torch.sigmoid(outputs)

                fpr, tpr, thresholds = roc_curve(targets, scores, pos_label=1)

                vis_roc(fpr, tpr)
                # torch.save(net, 'save.pt')
                break

            if epoch == args.epochs-1:
                output_history = []
                label_history = []
                batch_outputs = []
                batch_targets = []
                for i_batch, batch in enumerate(test_dataloader):
                    ft1, adj1, ft2, adj2, num_size1, target1, d11, d21, mask_1, mask_2 = batch
                    ft1 = ft1.cuda()
                    adj1 = adj1.cuda()
                    ft2 = ft2.cuda()
                    adj2 = adj2.cuda()
                    num_size = num_size1.cuda()
                    # target = target1.cuda()
                    d1 = d11.cuda()
                    d2 = d21.cuda()
                    output, output_feature = net(ft1, adj1, ft2, adj2, num_size, d1, d2)
                    output_feature = output_feature.cpu()

                    batch_outputs.append(output)
                    batch_targets.append(target1)

                    output_history.append(output_feature.numpy())
                    label_history.append(target1.numpy())

                # np.save("output_features50_att.npy", output_history)
                # np.save("output_labels50_att.npy", label_history)

                outputs = torch.cat(batch_outputs).cpu()
                targets = torch.cat(batch_targets).cpu()
                scores = torch.sigmoid(outputs)

                fpr, tpr, thresholds = roc_curve(targets, scores, pos_label=1)

                vis_roc(fpr, tpr)
                # torch.save(net, 'save.pt')
            LOG[args.logging](
                net, validation_dataloader, test_dataloader, criterion, epoch, args)
    torch.save(net, "MAV-DDI.pt")
    vis_loss(now_epoch, loss_history)


if __name__ == '__main__':
    main()
