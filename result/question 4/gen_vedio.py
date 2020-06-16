import cv2

video_full_path = "../img/无人机拍庄园.mp4"
cap = cv2.VideoCapture(video_full_path)
print(cap.isOpened())

frame_count = 1920
cap.set(cv2.CAP_PROP_POS_FRAMES,frame_count)

success = True
while (success):
    success, frame = cap.read()
    print('Read a new frame: ', success)

    params = []
    # params.append(cv.CV_IMWRITE_PXM_BINARY)  , params
    params.append(1)
    #if (frame_count>19000) & (frame_count>20600) :
        #break
    cv2.imwrite("./data/" + "%d.png" % frame_count, frame)

    frame_count = frame_count + 1
    if frame_count> 1939:
        break

cap.release()