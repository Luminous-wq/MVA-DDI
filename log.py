import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch

SAVEDMODELS_DIR = 'savedmodels/'
# container for all objects getting passed between log calls
class Globals:
    evaluate_called = False
g = Globals()
def all_evaluate(output, target):

    lengh = len(output)
    scores = torch.sigmoid(output)

    auroc = roc_auc_score(target, scores)
    aupr = average_precision_score(target, scores)

    scores = np.array(scores).astype(float)
    sum_scores = np.sum(scores)
    ave_scores = sum_scores / lengh
    target = np.array(target).astype(int)

    Confusion_M = np.zeros((2, 2), dtype=float)  # (TN FP),(FN,TP)
    for i in range(lengh):
        if (scores[i] < ave_scores):
            scores[i] = 0
        else:
            scores[i] = 1
    scores = np.array(scores).astype(int)

    for i in range(lengh):
        if(target[i] == scores[i]):
            if(target[i] == 1):
                Confusion_M[0][0] += 1 #TP
            else:
                Confusion_M[1][1] += 1 #TN
        else:
            if(target[i] == 1):
                Confusion_M[0][1] += 1 #FP
            else:
                Confusion_M[1][0]  +=1 #FN

    Confusion_M = np.array(Confusion_M, dtype=float)
    print('Confusion_M:', Confusion_M)
    accuracy = (Confusion_M[1][1] + Confusion_M[0][0]) / (
            Confusion_M[0][0] + Confusion_M[1][1] + Confusion_M[0][1] + Confusion_M[1][0])

    recall = Confusion_M[0][0] / (Confusion_M[0][0] + Confusion_M[0][1])
    precision = Confusion_M[0][0] / (Confusion_M[0][0] + Confusion_M[1][0])

    F1 = 2*precision*recall
    h = precision+recall
    F1 = F1/h
    sum = 0.0
    for i in range(lengh):
        sum = sum + (target[i] - scores[i]) * (target[i] - scores[i])

    return F1, accuracy, recall, precision, auroc, aupr


SCORE_FUNCTIONS = {
    'All':all_evaluate}


def feed_net(net, dataloader, criterion):

    batch_outputs = []
    # batch_losses = []
    batch_targets = []
    for i_batch, batch in enumerate(dataloader):
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
        # loss = criterion(output, target1)
        batch_outputs.append(output)
        # batch_losses.append(loss.item())
        batch_targets.append(target1)

    outputs = torch.cat(batch_outputs)

    # loss = np.mean(batch_losses)#average loss
    targets = torch.cat(batch_targets)

    return outputs, targets


def evaluate_net(net,validation_dataloader, test_dataloader, criterion, args):
    global g
    if not g.evaluate_called:
        g.evaluate_called = True
        g.best_mean_validation_score, g.best_mean_test_score = 0, 0

    validation_output, validation_target = feed_net(net,validation_dataloader, criterion)
    test_output, test_target = feed_net(net,test_dataloader, criterion)

    validation_scores = SCORE_FUNCTIONS[args.score](validation_output, validation_target)
    test_scores = SCORE_FUNCTIONS[args.score](test_output, test_target)

    new_best_model_found = test_scores[4] > g.best_mean_test_score


    if new_best_model_found:
        g.best_mean_validation_score = validation_scores[4]
        g.best_mean_test_score = test_scores[4]

        if args.savemodel:
            path = SAVEDMODELS_DIR + type(net).__name__
            torch.save(net, path)

    if(args.score=='All'):
     return{
         'F1 score':{'validation': validation_scores[0], 'test': test_scores[0]},
         'Accuracy':{'validation': validation_scores[1], 'test': test_scores[1]},
         'Recall':{'validation': validation_scores[2], 'test': test_scores[2]},
         'Precision':{'validation': validation_scores[3], 'test': test_scores[3]},
         'auroc':{'validation': validation_scores[4], 'test': test_scores[4]},
         'aupr':{'validation': validation_scores[4], 'test': test_scores[5]},
        'best mean':{'validation': g.best_mean_validation_score, 'test': g.best_mean_test_score}
        }


def get_run_info(net, args):
    return {
        'net': type(net).__name__,
        'args': ', '.join([str(k) + ': ' + str(v) for k, v in vars(args).items()]),
        'modules': {name: str(module) for name, module in  net._modules.items()}
    }


def less_log(net, validation_dataloader, test_dataloader, criterion, epoch, args):

    scalars = evaluate_net(net, validation_dataloader, test_dataloader, criterion, args)
    global g
    if not g.evaluate_called:
        run_info = get_run_info(net, args)
        print('net: ' + run_info['net'])
        print('args: {' + run_info['args'] + '}')
        print('****** MODULES: ******')
        for name, description in run_info['modules'].items():
            print(name + ': ' + description)
        print('**********************')

    if(args.score=='All'):
        print('epoch {}, F1 score :validation mean: {}, testing mean: {}'.format(
            epoch+1,
            scalars['F1 score']['validation'],
            scalars['F1 score']['test']))
        print('ACC:validation mean: {}, testing mean: {}'.format(
            scalars['Accuracy']['validation'],
            scalars['Accuracy']['test']))
        print('Precision:validation mean: {}, testing mean: {}'.format(
            scalars['Precision']['validation'],
            scalars['Precision']['test']))
        print('Recall:validation mean: {}, testing mean: {}'.format(
            scalars['Recall']['validation'],
            scalars['Recall']['test']))
        print('AUROC:validation mean: {}, testing mean: {}'.format(
            scalars['auroc']['validation'],
            scalars['auroc']['test']))
        print('AUPR:validation mean: {}, testing mean: {}'.format(
            scalars['aupr']['validation'],
            scalars['aupr']['test']))
        print('best auroc:validation mean: {}, testing mean: {}'.format(
            scalars['best mean']['validation'],
            scalars['best mean']['test']))

    else:
         mean_score_key = 'mean {}'.format(args.score)
         print('epoch {}, validation mean : {}, testing mean :{}'.format(
                epoch + 1,
                args.score, scalars[mean_score_key]['validation'],
                args.score, scalars[mean_score_key]['test']),
         )

LOG = {
    'less':less_log}