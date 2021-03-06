import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import os

def document(epoch, epoch_time, train_loss, train_acc, val_loss, val_acc, doc):

    print("Epoch " + str(epoch + 1) + " took " + str(epoch_time) + "s")
    print('Training   Loss      : {:.4f}'.format(train_loss))
    print('Training   Accuracy  : {:.4f}'.format(train_acc))
    print('Validation Loss      : {:.4f}'.format(val_loss))
    print('Validation Accuracy  : {:.4f}'.format(val_acc))
    print("---")

    doc['train_loss'].append(train_loss)
    doc['train_acc'].append(train_acc)
    doc['val_loss'].append(val_loss)
    doc['val_acc'].append(val_acc)
    doc['epoch'].append(epoch)

def plot_doc(doc, save=True):
    # plot loss
    plt.subplot(211)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    plt.title('Loss')
    plt.plot(doc['epoch'], doc['train_loss'], color='cornflowerblue', label='train', linewidth=1.0)
    plt.plot(doc['epoch'], doc['val_loss'], color='coral', label='val', linewidth=1.0)
    plt.legend(['Train loss', 'Val loss'], loc='upper right')
    # plot accuracy
    plt.subplot(212)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    plt.title('Accuracy')
    plt.plot(doc['epoch'], doc['train_acc'], color='cornflowerblue', label='train', linewidth=1.0)
    plt.plot(doc['epoch'], doc['val_acc'], color='coral', label='val', linewidth=1.0)
    plt.legend(['Train acc', 'Val acc'], loc='upper right')
    # save plot to file
    if save:
        if os.path.isdir("info") == False:
            os.mkdir("info")
        plt.savefig('info/loss_acc_curve.png')
        plt.close()
    else :
        plt.show()

def write_info(best_epoch, best_acc):
    print("The best epoch : ", best_epoch," .")
    print("The best accuracy : ", best_acc, " .")
    print("--------------------")

    if os.path.isdir("info") == False:
            os.mkdir("info")
    file = open('info/best_epoch.txt', 'a')
    file.write('Best Epoch    = {:d}\n'.format(best_epoch))
    file.write('Best Accuracy = {:.8f}\n'.format(best_acc))
    file.write('-------')

def write_test(test_acc, test_time):
    print("test accuracy: {}".format(test_acc))
    print("--------------------")
    if os.path.isdir("info") == False:
            os.mkdir("info")
    file = open('info/test_acc.txt', 'a')
    file.write('Test acc = {:.8f}\n'.format(test_acc))
    file.write('Test time = {:.8f}\n'.format(test_time))
    file.write('-------')

def visualize(video):
    for frame in video:
        fig,ax = plt.subplots(1)
        ax.imshow(frame)
        plt.show()