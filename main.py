import cv2
import numpy as np
import time
import argparse


import utills, plot

confid = 0.5
thresh = 0.5
mouse_pts = [(0,0),(640,0),(640,480),(0,480),(0,0),(384,0),(0,288)]



def calculate_social_distancing(vid_path, net, ln1):
    
    count = 0
    #vs = cv2.VideoCapture(0)    
    vs = cv2.VideoCapture("./data/test.mp4")
   
    height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(vs.get(cv2.CAP_PROP_FPS))
    
 
    scale_w, scale_h = utills.get_scale(width, height)
        
    points = []
    global image
    
    while True:

        (grabbed, frame) = vs.read()

        if not grabbed:
            print('here')
            break
            
        (H, W) = frame.shape[:2]
        
        
        points = mouse_pts      
                 
       
        src = np.float32(np.array(points[:4]))
        dst = np.float32([[0, H], [W, H], [W, 0], [0, 0]])
        prespective_transform = cv2.getPerspectiveTransform(src, dst)

       
        pts = np.float32(np.array([points[4:7]]))
        warped_pt = cv2.perspectiveTransform(pts, prespective_transform)[0]
        
        
        distance_w = np.sqrt((warped_pt[0][0] - warped_pt[1][0]) ** 2 + (warped_pt[0][1] - warped_pt[1][1]) ** 2)
        distance_h = np.sqrt((warped_pt[0][0] - warped_pt[2][0]) ** 2 + (warped_pt[0][1] - warped_pt[2][1]) ** 2)
        pnts = np.array(points[:4], np.int32)
        cv2.polylines(frame, [pnts], True, (70, 70, 70), thickness=2)
    
    
    
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln1)
        end = time.time()
        boxes = []
        confidences = []
        classIDs = []   
    
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if classID == 0:

                    if confidence > confid:
                        
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
                        
                    
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confid, thresh)
        font = cv2.FONT_HERSHEY_PLAIN
        boxes1 = []
        for i in range(len(boxes)):
            if i in idxs:
                boxes1.append(boxes[i])
                x,y,w,h = boxes[i]
                
        if len(boxes1) == 0:
            count = count + 1
            continue
            
        
        person_points = utills.get_transformed_points(boxes1, prespective_transform)
        
       
        distances_mat, bxs_mat = utills.get_distances(boxes1, person_points, distance_w, distance_h)
        risk_count = utills.get_count(distances_mat)
    
        frame1 = np.copy(frame)
        
       
        img = plot.social_distancing_view(frame1, bxs_mat, boxes1, risk_count)
       
        if count != 0:
            #output_movie.write(img)
            
            cv2.putText(img,"total Voilations : " + str(risk_count[0]), (30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255), 2,cv2.LINE_AA,False)
            cv2.putText(img,"Number of people : " + str(len(confidences)), (30,60),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 2,cv2.LINE_AA,False)
            cv2.imshow('real_time',img)
        cv2.namedWindow('real_time', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('real_time', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        #print(img.shape[:2])
        count = count + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
     
    vs.release()
    cv2.destroyAllWindows() 
        

if __name__== "__main__":

    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-v', '--video_path', action='store', dest='video_path', default='./data/test.mp4' ,
                    help='Path for input video')
    

    parser.add_argument('-m', '--model', action='store', dest='model', default='./models/',
                    help='Path for models directory')
                    
                    
    values = parser.parse_args()
    
    model_path = values.model
    if model_path[len(model_path) - 1] != '/':
        model_path = model_path + '/'
       
    
    weightsPath = model_path + "yolov4-tiny.weights"
    configPath = model_path + "yolov4-tiny.cfg"

    net_yl = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net_yl.getLayerNames()
    ln1 = [ln[i[0] - 1] for i in net_yl.getUnconnectedOutLayers()]

   
    calculate_social_distancing(values.video_path, net_yl, ln1)



