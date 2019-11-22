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
