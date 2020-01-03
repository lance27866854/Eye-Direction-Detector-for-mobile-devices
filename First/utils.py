import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import os

def document(epoch, epoch_time, train_loss, train_acc, val_loss, val_acc, doc):
    test_acc_avg = (train_acc[0]+train_acc[1]+train_acc[2])/3 
    val_acc_avg = (val_acc[0]+val_acc[1]+val_acc[2])/3
    
    print("Epoch " + str(epoch + 1) + " took " + str(epoch_time) + "s")
    print('Training   Loss      : {:.4f}, {:.4f}, {:.4f}'.format(train_loss[0], train_loss[1], train_loss[2]))
    print('Training   Accuracy  : {:.4f}, {:.4f}, {:.4f}'.format(train_acc[0], train_acc[1], train_acc[2]))
    print('Validation Loss      : {:.4f}, {:.4f}, {:.4f}'.format(val_loss[0], val_loss[1], val_loss[2]))
    print('Validation Accuracy  : {:.4f}, {:.4f}, {:.4f}'.format(val_acc[0], val_acc[1], val_acc[2]))
    print("---")

    for i in range(3):
        doc['train_loss_'+str(i)].append(train_loss[i])
        doc['train_acc_'+str(i)].append(train_acc[i])
        doc['val_loss_'+str(i)].append(val_loss[i])
        doc['val_acc_'+str(i)].append(val_acc[i])
    
    doc['epoch'].append(epoch)

def plot_doc(doc, save=True):
    # plot loss
    plt.subplot(211)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    plt.title('Loss')
    for i in range(3):
        plt.plot(doc['epoch'], doc['train_loss_'+str(i)], color='cornflowerblue', label='train', linewidth=1.0)
        plt.plot(doc['epoch'], doc['val_loss_'+str(i)], color='coral', label='val', linewidth=1.0)
    plt.legend(['Train loss S', 'Val loss S', 'Train loss M', 'Val loss M', 'Train loss L', 'Val loss L'], loc='upper right')
    # plot accuracy
    plt.subplot(212)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    plt.title('Accuracy')
    for i in range(3):
        plt.plot(doc['epoch'], doc['train_acc_'+str(i)], color='cornflowerblue', label='train', linewidth=1.0)
        plt.plot(doc['epoch'], doc['val_acc_'+str(i)], color='coral', label='val', linewidth=1.0)
    plt.legend(['Train acc S', 'Val acc S', 'Train acc M', 'Val acc M', 'Train acc L', 'Val acc L'], loc='upper right')
    # save plot to file
    if save:
        if os.path.isdir("info") == False:
            os.mkdir("info")
        plt.savefig('info/loss_acc_curve.png')
        plt.close()
    else :
        plt.show()

def write_info(best_epoch, best_acc):
    print("The best epoch : ", best_epoch, " .")
    print("The best accuracy : ", best_acc, " .")
    print("--------------------")

    if os.path.isdir("info") == False:
            os.mkdir("info")
    file = open('info/best_epoch.txt', 'a')
    file.write('Best Epoch    = {:d}\n'.format(best_epoch))
    file.write('Best Accuracy = {:.8f}\n'.format(best_acc))
    file.write('-------')

def write_test(test_acc, left_acc, right_acc):
    print("test accuracy: {}".format(test_acc))
    print("--------------------")
    if os.path.isdir("info") == False:
            os.mkdir("info")
    file = open('info/test_acc.txt', 'a')
    file.write('Test acc = {:.8f}\n'.format(test_acc))
    file.write('Left acc = {:.8f}\n'.format(left_acc))
    file.write('Right acc = {:.8f}\n'.format(right_acc))
    file.write('-------')
