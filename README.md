# ECTEL2023

This is the official repository for the paper:

Nghia Duong-Trung, Hitesh Kotte, Milos Kravcik (2023). Augmented Intelligence in Tutoring Systems: A Case Study in Real-time Pose Tracking to Enhance the Self-Learning of Fitness Exercises. In Proceedings of the Eighteenth European Conference on Technology Enhanced Learning (ECTEL2023). Springer LNCS.  September 6-8, 2023. Aveiro, Portugal.

%@% setup to run the code
1. create a virtual environment (Conda Env)
  * conda create -n yolov7_custom python=3.9
2. To activate environment 
  * conda activate yolov7_custom 
3. clone the repository
  * https://github.com/duongtrung/ECTEL2023.git
4. Go to cloned folder
  * cd ECTEL2023
5. Install packages using the following command
  * pip install -r requirements.txt
6. Download Yolov7 pose estimation weights from the official GitHub repository 
  * https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt
7. For pose estimation on video/webcam, use the pose-estimate.py file; to execute this file, use the following command.
-For CPU
  * python pose-estimate.py --source "your custom video.mp4" --device cpu
8. For Pushup_counting, use the pushup_counter.py file, and to execute this file, use the following command
-For CPU
  * python pushup_counter.py --source "pushup.mp4" --device 0 --curltracker=True
Use 8 for different exercises with their corresponding exercise names.

trainer.py is used for all variations of exercises.

Use plot_performance.py to plot the graph in ascending order of angles.
Use plot_performance1 to plot the graph in descending order of angles.
