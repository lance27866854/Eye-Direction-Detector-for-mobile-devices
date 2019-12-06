- FLAGS：
    is_train: True 进行 training，False 进行 testing。
    restore: True 会读取存储的 weights，此时须输入inference_version。
    batch: 这里指的是 file 要读到多少。 (抱歉，这跟一般的用语不同)
    slice: 就是指一般的 batch，一次更新要从多少 image 一起 (stochastic)。
    data_dir: 把 training data 放在哪个资料夹下。
    test_dir: 指 test data 的资料夹名称(预设放在 training directory 下)。
    val_num: 要拆多少作为 validation set (建议总量的　1/8 ~ 1/10　为合适)。
    train_dir: model 训练好的参数会放在这个资料夹下。

- 快速训练懒人包：
    先跑看看　python main.py
    再跑 python main.py --is_train=False --inference_version=xxx
    会看到多了 info 的资料夹，里面装了图跟精度的档案。

    如果要从过去已经训练好的模型继续训练，则可执行　python main.py --restore=True --inference_version=xxx
    但要注意，这里的图会从新的地方开始画，若要得到整个　loss、accuracy 的过程必须自己写代码实现。

- 架构调整说明：
    打开 model.py，基本上可以由 layer_1、layer_2、layer_3 三个函数修改模型架构。
    也可调整 init 中的 learning_rate、learning_rate_decay_factor，去得到较快的收敛。