
import matplotlib.pyplot as plt
import time
import datetime
import os

def vis_loss(epochs, loss_vector):

    plt.figure("loss_yeah")
    x = list(range(epochs))
    x = [i + 1 for i in x]
    plt.plot(x, loss_vector, 'b*-')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("loss_line")
    timestamp = int(time.time())
    date_time = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H-%M-%S')
    loss_name = os.path.join("Out", "loss_" + date_time + '.png')
    plt.savefig(loss_name)

def vis_roc(fpr, tpr):

    plt.figure("roc_yeah")
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='MAV')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic of MAV')
    plt.legend(loc="lower right")
    timestamp = int(time.time())
    date_time = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H-%M-%S')
    loss_name = os.path.join("Out", "roc_" + date_time + '.png')
    plt.savefig(loss_name)