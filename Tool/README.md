只需要调整 main.py 即可，使用方式如下：

1) Data_Editor: 

- 定义 folder name，直接修改 NAME 变数即可(建议取数字 0,1,2,...)。
- 定义该资料夹的大小(FILE_NUM)，规定为 20 个视频(每个视频必须是以0,1,2...19编号)。
- 定义动态方向(NO = 0, UP = 1, RIGHT = 2, DOWN = 3, LEFT=4)，写在 ANSWER 当中以 list 储存。
- 定义眼睛框中心点位置，写在 REGION_POINTS 当中以 list 储存。
- 眼睛框中心点位置，若该眼睛是被遮挡的状态，请存[-1, -1]。
- 设定路径(PATH)将"C:\\Users\\user\\Desktop\\Model\\Tool\\sources\\"改为你电脑中储存的路径。

- 先执行以下两行，不要改动 width, height：
    data_editor = Data_Editor(name=NAME, answer=ANSWER, file_num=FILE_NUM)
    data_editor.convert_videos_to_files()
  执行后会出现以 NAME 为命名的资料夹，中间有 ground_truth.npy(存ANSWER)、info.txt(存width、height等)、video.npy(存视频vector)。

- 执行下行：
    data_editor.random_show(REGION_POINTS[i], i)
  会随机由第i(i=0,1,2...)个视频中抽一张图，并显示眼睛框位置。在此请调整框的中间点位置，尽量对其眼睛中间。框很大没有关系，只要确认眼睛在内即可。

- 执行下行：
    data_editor.convert_eye_points_to_files(REGION_POINTS)
  会将你所填在 REGION_POINTS 当中的点存成 region_point.npy。

2) Data_Manager:

- 此工具将进一步把前述的资料转换为training data的形式。
- 设定相对路径，这时把path直接设为要转换的Folder名称(NAME)即可。
- 首先将data转换为预选点(points of region-proposal):
    data_manager.to_candidate_list()
  执行后会在以 NAME 为命名的资料夹下看到1个新文件，即candidate_list.npy。

- 对于第一部分的training data：
    data_manager.to_First_data()
  执行后会在以 NAME 为命名的资料夹下看到1个新文件，即first_data.npy。

- 对于第二部分的training data，只要执行下行即可：
    data_manager.to_Second_data()
  执行后会在以 NAME 为命名的资料夹下看到1个新文件，即 second_data.npy

- 确认是否撷取正确图像，可以呼叫to_Second_data时开启random_show：
    data_manager.to_Second_data(True)