import os
import numpy as np
import matplotlib.pyplot as pyplot

# plot diagnostic learning curves
def plot_loss(train_dic, val_dic, index=0):
        # plot loss
        pyplot.subplot(211)
        pyplot.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        pyplot.title('Loss')
        pyplot.plot(train_dic['counter'], train_dic['losses'], color='cornflowerblue', label='train', linewidth=1.0)
        pyplot.plot(val_dic['counter'], val_dic['losses'], color='coral', label='val', linewidth=1.0)
        pyplot.legend(['Train Loss', 'Val loss'], loc='upper right')
        # plot accuracy
        pyplot.subplot(212)
        pyplot.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
        pyplot.title('Accuracy')
        pyplot.plot(train_dic['counter'], train_dic['acc'], color='cornflowerblue', label='train', linewidth=1.0)
        pyplot.plot(val_dic['counter'], val_dic['acc'], color='coral', label='val', linewidth=1.0)
        pyplot.legend(['Train accuracy', 'Val accuracy'], loc='upper right')
        # save plot to file
        pyplot.savefig('loss_info/loss' + str(index) + '.png')
        pyplot.close()

def write_info(epoch, train_acc, train_loss, val_acc, val_loss):
    if os.path.isdir("loss_info") == False:
            os.mkdir("loss_info")
    file = open('loss_info/loss_info.txt', 'a')
    file.write('epoch[ {} ]:     Training Accuracy = {:.8f}, Loss = {:.8f}\n'.format(epoch, train_acc, train_loss))
    file.write('                  Validation Accuracy = {:.8f}, Loss = {:.8f}\n'.format(val_acc, val_loss))
