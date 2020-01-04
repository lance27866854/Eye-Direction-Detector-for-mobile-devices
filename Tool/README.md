Download the dataset:


Just adjust main.py and use it as follows:

1) Data_Editor:

- Define the folder name and modify the NAME variable directly (recommended number 0,1,2, ...).
- Define the size of the folder (FILE_NUM) and specify 20 videos (each video must be numbered 0,1,2 ... 19).
- Define the dynamic direction (NO = 0, UP = 1, RIGHT = 2, DOWN = 3, LEFT = 4), write it in ANSWER and store it as a list.
- Define the position of the center of the eye frame, write it in REGION_POINTS and store it as a list.
- The position of the center of the eye frame. If the eye is blocked, please save [-1, -1].
- Set the path (PATH) and change "C: \\ Users \\ user \\ Desktop \\ Model \\ Tool \\ sources \\" to the path stored in your computer.

- Execute the following two lines without changing the width and height:
```
Data_editor = Data_Editor (name = NAME, answer = ANSWER, file_num = FILE_NUM)
Data_editor.convert_videos_to_files ()
```
After execution, a folder named NAME will appear, with ground_truth.npy (save ANSWER), info.txt (save width, height, etc.), and video.npy (save video vector) in the middle.

-Execute:
```
Data_editor.random_show (REGION_POINTS [i], i)
```
A random image will be drawn from the i-th (i = 0,1,2 ...) video and the position of the eye frame will be displayed. Please adjust the position of the middle point of the frame to the center of the eyes as precise as possible. It doesn't matter if the frame is large, just make sure your eyes are inside.

-Execute:
```
Data_editor.convert_eye_points_to_files (REGION_POINTS)
```
The points you filled in REGION_POINTS will be stored as region_point.npy.

## Data_Manager:

- This tool will further convert the aforementioned data into training data.
- Set the relative path. In this case, set the path directly to the folder name (NAME) to be converted.
- First convert data to pre-selected points (points of region-proposal):
```
Data_manager.to_candidate_list ()
```
After the execution, you will see a new file under the folder named NAME, namely candidate_list.npy.

- For the training data of the first part:
```
data_manager.to_First_data ()
```
After the execution, you will see a new file under the folder named NAME, that is, first_data.npy.
```
data_manager.to_First_data (zero_label = 5, data_augmentation = True)
```
You can adjust zero_label (the number of non-eye points to choose), and data_augmentation (whether to do data augmentation) to generate training data. The size of the dataset will be printed after the conversion, please be sure to record the report.

- For the training data of the second part, just execute the downlink:
```
Data_manager.to_Second_data ()
```
After execution, you will see a new file under the folder named NAME, namely second_data.npy

- Confirm whether to capture the correct image, you can open random_show when calling `to_Second_data`:
```
Data_manager.to_Second_data (True)
```
