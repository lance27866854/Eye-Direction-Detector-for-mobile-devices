# Real-Time Eye Detection with cascaded RNN

## 数据集采集懒人包
### :point_right: 搜集流程：
1. 首先将采集来的数据(视频)依"数字0,1,2,3,..."编号，图1。
2. 此时打开 cmd 由　github　上下载预先写好的 tools，打开 tool 资料夹，可以看到 encode.py、eye_region.py，档案放置情形如图2。
3. 编码步骤：开 encode.py 把视频编码成向量 -> 开 eye_region.py 将眼睛位置等资讯输入。
4. 当中需要修改encode.py 参数，如图3，包含FILE_NUM(影片数量)、BATCH_NUM(dataset的编号，在做之前须在群组先制定)、FILE_LOCATION(视频放置的位置，这必须是完整路径)、FILE_TYPE(.mp4或者任何cv2可以读的档案皆可)、ANSWER(此几支视频的动态的答案)。
5. 打开 cmd 编译档案，发现档案中多了2个档案(batchX.npy、batchX_info.txt)。
6. 将眼睛位置的资讯输入，编译 eye_region.py，发现档案中多了1个档案(batchX_region.npy)，如图4。

| 图 1 | 图 2 | 图 3 | 图 4 |
| -------- | -------- | -------- | -------- |
| ![](https://i.imgur.com/OkSKLNs.png) | ![](https://i.imgur.com/VEYq3Xr.png)| ![](https://i.imgur.com/wlQX4MR.png) | ![](https://i.imgur.com/eoS0KqM.png) |

### :point_right: 限制：
1. 视频 width、height 需一致。

### :point_right: 本周计画：
11/22 23:59 前出 5 个 batch -> 一个 batch 需有 20 支影片，这周先搜集基本款(NO occlussion、NO glasses)。

---

## 目前进度(~11/24)
已经写好CNN分类器，可以从 candidate regions 找出眼睛位置并分类，总结一下算法以及对应的function：
1. 首先是我们的档案架构：/First/放的是 cascaded 架构中的前半部分；/Second/ 是后半部分。
2. 前半部分当中，在启动 model 上分为 4 种模式：
```
# Mode 0: Training and also store the data points (usually for the first time).
(Mode 0 启动后，会先跑 Gaussian Kernel 卷积，用 sobel Kernel 找点(candidate_points)，并用进行训练。)
# Mode 1: Training but use stored data points (usually for the second time).
(Mode 1 启动后，会从档案中抓取预先存好的点(candidate_points)，并用进行训练。)
# Mode 2: Testing for checking the accuracy (usually for the first time).
(Mode 2 启动后，进行 Testing，此时要填入相对应的 inference version。)
# Mode 3: Testing for generate data for the cascaded network (usually for the second time).
(Mode 3 启动后，会先跑进行 Testing，并存储预测的眼睛位置。(这先不用管，应该是后期会需要用的。))
```
3. RNN 应该是会用左右眼的图像进行 Training，目前想到可以尝试的方向是双眼图像输入。

### :point_right: 算法瓶紧
我们的 model 遇到以下瓶紧：
1. Rubustness需要加强，有些 frame 会找不到眼睛位置。
2. 做不到 Realtime，看有没有更快的算法可以支持眼睛位置的框取。

### :point_right: 本周计画：
11/29 23:59 做出算法改良初步想法，继续搜集数据(another 200 videos)，预计再下周会进到实验阶段，到时候就是做hyperparameter search and structure search。
1. (1人) 专门搜集 dataset，务必确认 dataset 质量，也可以参考别人的论文如何搜集。
2. (1人) 看 paper，专门研究 region 算法(需要下周小小报告一下)，看能不能用上。
3. NOTE: 做搜集dataset 的人可以自由修改eye_region.py、encode.py，但请固定格式，必须提供接口(video 数量、长、宽、video 存储方式、眼睛位置存储)，我认为可以存的干净一点，目前有点乱。

- 有遮挡、不戴眼镜，尽量短(0.5s)

---

## 目前进度(~12/01)

搭建完毕两个 network 的雏形，可以 run 了。第一个 net 在第一个 batch 可以达到 85% 正确率。第二个 net 可以达到约 95% 正确率(overfit)。我还是希望我们 dataset 可以 300\~500 支影片，约 15\~25 batch。

下图总结我们的算法：

| Model Structure |
| --------------- |
|![](https://i.imgur.com/bryZNEc.png)|

### :point_right: 本周计画：

12/06 23:59 如果忙得话就搜集 data 吧，但我还需要一个人帮我做 structure search、fine tuning parameters(这部分我周一之前会整理出来)。<br>
请将 dataset 都转为 first.npy、second.npy。

## 进度日程表
<img src="https://i.imgur.com/ld2yiPG.png" width="330">

## What can we do next?
#### First, we must think of what is the requisite for a paper! <br>
New ideas, better performance(not necessary but important), objective evaluations, convincing description and images... <br>
Basically, three novel ideas are required.

#### Second, we must make our target clear and also ensure we can achieve it. <br>
This is gonna be a tough choice. <br>
We cannot do all things in one project. <br>
so, here are the experiments we can do... <br>

- (MUST) Compare the performance between different models.
```
>> Comparison between our algorithm and others' algorithm. (e.g. we can compare the top-1 accuracy and top-2 accuracy with our model and with others' model.)
>> Make sure we well-control the experiments with different conditions (e.g. 1. add the data augmentation 2. the test images are from unknown people or from known people...)
>> Document the results(e.g. tables, graphs...).
```
- Get more data. (so why we need to collect data by ourselves?)
- New (or old) training tricks. (e.g. dropout, batch normalization...)
- (MUST) ways to expand our dataset (e.g. data augmentation, random filp, ratation...).
