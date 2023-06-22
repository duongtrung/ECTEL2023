import cv2
import csv
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt,strip_optimizer,xyxy2xywh
from utils.plots import output_to_keypoint, plot_skeleton_kpts,colors,plot_one_box_kpt


from trainer import findAngle
from PIL import ImageFont,ImageDraw,Image

@torch.no_grad()
def run(poseweights="yolov7-w6-pose.pt",source="./input/upper_chest_press_new.mp4",device='cpu',curltracker=False,drawskeleton=False):
    path = source
    ext = path.split('/')[-1].split('.')[-1].strip().lower()
    if ext in ["mp4","webm","avi"] or ext not in ["mp4","webm","avi"] and ext.isnumeric():
        
        input_path = int(path) if path.isnumeric() else path
    
        device = select_device(opt.device) 
        half = device.type != 'cpu'

        model = attempt_load('yolov7-w6-pose.pt', map_location=device)  
        _ = model.eval()
        
    
        
        cap = cv2.VideoCapture("./input/upper_chest_press_new.mp4")  
        webcam = False
        
        if (cap.isOpened() == False):
            print('Error while trying to read video. Please check path again')
        
        fw,fh = int(cap.get(3)),int(cap.get(4))  
        if ext.isnumeric():
            webcam =True
            fw,fh = 1280,768
        
        vid_write_image = letterbox(cap.read()[1], (fw), stride=64, auto=True)[0] 
        resize_height, resize_width = vid_write_image.shape[:2]
        out_video_name = "output" if path.isnumeric() else f"{input_path.split('/')[-1].split('.')[0]}"
        out = cv2.VideoWriter(f"{source}_keypoint.mp4",cv2.VideoWriter_fourcc(*'mp4v'), 30,(resize_width, resize_height))
        if webcam:
            out = cv2.VideoWriter(f"{out_video_name}_kpt.mp4",cv2.VideoWriter_fourcc(*'mp4v'),30,(fw,fh))

        frame_count, total_fps = 0,0
        
        shoulder_raise = 0
        direction = 0
        bar = 0
        angles = []
        percentages = []
        bars = []     
 
        Percentage = 0 
        
        fontpath = "futur.ttf"
        font = ImageFont.truetype(fontpath,32)
        
        font1 = ImageFont.truetype(fontpath,160)


        while(cap.isOpened):
            print("Frame {} Processing".format(frame_count+1))

            ret, frame = cap.read()  
            
            if ret: 
                orig_image = frame 
                
                image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB) 
                
                if webcam:
                    image = cv2.cvtColor(image,(fw,fh),interpolation = cv2.INTER_LINEAR)
                    
                
                image = letterbox(image, (fw), stride=64, auto=True)[0]
                image_ = image.copy()
                image = transforms.ToTensor()(image)
                image = torch.tensor(np.array([image.numpy()]))
            
                image = image.to(device)  
                image = image.float() 
                start_time = time.time() 
            
                with torch.no_grad():
                    output_data, _ = model(image)

                output_data = non_max_suppression_kpt(output_data,0.5,0.65,nc=model.yaml['nc'],nkpt=model.yaml['nkpt'],kpt_label=True)    
                                                                     
            
                output = output_to_keypoint(output_data)

                img = image[0].permute(1, 2, 0) * 255 
                img = img.cpu().numpy().astype(np.uint8)
                
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
                data = [["Angle", "Percentage", "Bar"]]
                
                if curltracker:
                    print(f"kpts shape: {output.shape}")
                    color = (255,0,0)#color = (254,118,136)
                    for idx in range(output.shape[0]):
                        kpts = output[idx,7:].T
                        
                        start_angle_R = 186
                        start_angle_L = 186
                        end_angle_R = 199
                        end_angle_L =199
                        

                        angleR = findAngle(img,kpts,5,7,9,draw = True)
                        angleL = findAngle(img,kpts,6,8,10,draw =True)
                        angles.append(angleR)

                        if idx == 0:
                            AngleR = start_angle_R
                            AngleL = start_angle_L
                        elif idx == output.shape[0] - 1:
                            AngleR = end_angle_R
                            AngleL = end_angle_L 
                        else:
                            AngleR = angleR
                            AngleL = angleL
                        
                        #PercentageR = np.interp(angleR, (start_angle_R, end_angle_R), (0, 100))
                        Percentage = np.interp(angleR, (start_angle_R, end_angle_R), (0, 100))

                        #Percentage = (PercentageR + PercentageL) / 2 
                        print("Percentage: {:.2f}%".format(Percentage))
                        percentages.append(Percentage)


                        #barR = np.interp(angleR, (start_angle_R, end_angle_R), (int(fh) - 100, 100))
                        bar = np.interp(angleR, (start_angle_R, end_angle_R), (int(fh) - 100, 100))
                        #bar = (barR + barL) / 2
                        #print("bar: {:.2f}%".format(bar))
                        #print("Angle R:", angleR)
                        #print("Angle L:", angleL)
                        bars.append(bar)
                        data.append([angleR, Percentage, bar])
                        
                        if direction == 0:
                            if Percentage < 35:
                                
                                direction = 1
                              
                        elif direction == 1:
                            if Percentage > 70:
                                shoulder_raise += 1
                                
                               
                                direction = 0
                        
                        cv2.line(img,(100,100),(100,int(fh)-100),(128,128,128),30)
                        cv2.line(img,(100,int(bar)),(100, int(fh)-100),color,30)
                        
                        if (int(Percentage) < 10):
                            cv2.line(img,(155,int(bar)),(190,int(bar)),color,40)
                        elif ((int(Percentage) >= 10) and (int(Percentage) < 100)):
                            cv2.line(img,(155,int(bar)),(200,int(bar)),color,40)
                        else:
                            cv2.line(img,(155,int(bar)),(210,int(bar)),color,40)
                      
                    im = Image.fromarray(img)
                    draw = ImageDraw.Draw(im)
                    draw.text((145,int(bar)-17),f"{int(Percentage)}%",font=font, fill= (255,255,255))
                    draw.text((fw-200,(fh//2)-250),f"{int(shoulder_raise)}",font=font1, fill= (128,0,0))    
                    img = np.array(im)

                with open('angles.csv', mode='w', newline='') as angles_file:
                    angles_writer = csv.writer(angles_file)
                    angles_writer.writerow(['Angle'])

                    with open('percentages.csv', mode='w', newline='') as percentages_file:
                        percentages_writer = csv.writer(percentages_file)
                        percentages_writer.writerow(['Percentage'])

                        with open('bars.csv', mode='w', newline='') as bars_file:
                            bars_writer = csv.writer(bars_file)
                            bars_writer.writerow(['Bar'])

                            # loop over the data and write to the CSV files
                            for angle, percentage, bar in zip(angles, percentages, bars):
                                angles_writer.writerow([angle])
                                percentages_writer.writerow([percentage])
                                bars_writer.writerow([bar])

                if webcam:
                    cv2.imshow('detection',img)
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break
                else:
                    img_ = img.copy()
                    img_= cv2.resize(img_,(960,540),interpolation = cv2.INTER_LINEAR)
                    cv2.imshow('detection',img_)
                    cv2.waitKey(1)
                  
                 
                end_time = time.time()
                fps = 1 / (end_time-start_time)
                total_fps += fps
                frame_count +=1
                out.write(img)
       
            else:
                break
                        
        cap.release()
        cv2.destroyAllWindows()
        avg_fps = total_fps / frame_count
        print(f"Average FPS: {avg_fps:.3f}")
    

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, help='video/0 for webcam') #video source
    parser.add_argument('--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)')   #device arugments
    parser.add_argument('--curltracker', type=bool, default='True', help='set as true to check count bicep curls')   #curltracker 
    parser.add_argument('--drawskeleton', type=bool, default='False', help='set as True to draw skeleton')   #curltracker
    opt = parser.parse_args()
    return opt

def plot_fps_time_comparision(time_list,fps_list):
    plt.figure()
    plt.xlabel('Time (s)')
    plt.ylabel('FPS')
    plt.title('FPS and Time Comparision Graph')
    plt.plot(time_list, fps_list,'b',label="FPS & Time")
    plt.savefig("FPS_and_Time_Comparision_pose_estimate.png")
    
def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    strip_optimizer(opt.device,opt.poseweights)
    main(opt)

                    
                        
                        
                        
                         
                             
                         
                        
                        
                        
                        
                                                     

                        
                                                     

