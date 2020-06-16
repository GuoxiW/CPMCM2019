import os
import numpy as np
import cv2


known_length = 720

A = (387, 511)
B = (325, 442)
C = (311, 443)
D = (362, 518)
E = (302, 420)
F = (283, 399)


def compute_distance(point1, point2):
    (x1, y1) = point1
    (x2, y2) = point2
    d = np.sqrt(np.square(x1-x2) + np.square(y1-y2))
    return d


def cross_ratio(c1, c2, c3, c4):
    cr = (compute_distance(c1, c3) * compute_distance(c2, c4)) / (compute_distance(c2, c3) * compute_distance(c1, c4))
    return cr


def compute_fun(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    k = (y2-y2) / (x2-x1)
    b = y1 - k*x1
    return k, b


def predict_length(P1, P2, A1, A2, V1, V2):
    l1 = (P1, A1)
    vl = (V1, V2)
    cross_point3 = get_crosspoint(l1, vl)
    l2 = (P2, cross_point3)
    la = (A1, A2)
    Ax = get_crosspoint(l2, la)
    cr = cross_ratio(A2, Ax, A1, V1)
    relative_length = known_length / cr

    return relative_length


def predict_distance1(P1, P2, A1, A2, V1, V2, rl):
    l1 = (P1, A1)
    vl = (V1, V2)
    cross_point3 = get_crosspoint(l1, vl)
    l2 = (P2, cross_point3)
    la = (A1, A2)
    Ax = get_crosspoint(l2, la)
    cr = cross_ratio(A2, Ax, A1, V1)
    require_distance = cr * rl

    return require_distance


def predict_distance2(P1, P2, A1, A2, V1, V2, rl):
    # P1=P2, P2=P3
    l1 = (P1, A1)
    vl = (V1, V2)
    cross_point3 = get_crosspoint(l1, vl)
    l2 = (P2, cross_point3)
    la = (A1, A2)
    Ax = get_crosspoint(l2, la)
    cr = cross_ratio(A2, Ax, A1, V2)
    require_distance = cr * rl
    return require_distance


def get_crosspoint(line1, line2):
    #求取相交点  line输入样例格式(x,y)
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) #Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return (x, y)

l1 = (A, B)
l2 = (C, D)
l3 = (A, D)
l4 = (B, C)
V1 = get_crosspoint(l1, l2)
V2 = get_crosspoint(l3, l4)
print(V1)

cr1 = cross_ratio(A, B, E, V1)
AE = cr1 / (cr1-1) * known_length
print(AE)

cr2 = cross_ratio(A, B, F, V1)
AF = cr2 / (cr2-1) * known_length
print(AF)

if __name__ == '__main__':

    img1 = cv2.imread('./car/*.jpg')

    src_pts = np.float32([[54, 128], [64, 123], [90, 159 ], [105, 152]]).reshape(-1, 1, 2)
    #src_pts = np.float32([[34, 100], [47, 96], [85, 175 ], [111, 166]]).reshape(-1, 1, 2)
    #src_pts = np.float32([[90, 63], [98, 61], [102, 77], [110, 74]]).reshape(-1, 1, 2)
    dst_pts = np.float32([[500, 4000], [515, 4000], [500, 4600 ], [515, 4600]]).reshape(-1, 1, 2)
    #dst_pts = np.float32([[200, 400], [215, 400], [200, 1000 ], [215, 1000]]).reshape(-1, 1, 2)
    # 透视变换：利用sifi等特征点计算透视变换矩阵（第一次）
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 7.5)
    perspectiveImg2 = cv2.warpPerspective(img1, M, (1200, 5000))
    perspectiveImg2 = cv2.resize(perspectiveImg2, (1920, 1080))


(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
def track(vid):

    # Set up tracker.
    # Instead of MIL, you can also use

    # MEDIANFLOW效果最佳
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW']
    tracker_type = tracker_types[-1]

    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()

    # Read video
    video = cv2.VideoCapture("vid_path")

    # Exit if video not opened.
    if not video.isOpened():
        print ("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print ('Cannot read video file')
        sys.exit()

    # Define an initial bounding box
    init_rect = cv2.selectROI(frame, False, False)
    x, y, w, h = init_rect
    # bbox = (287, 23, 86, 320)
    bbox = (x, y, w, h)

    # Uncomment the line below to select a different bounding box
    #bbox = cv2.selectROI(frame, False)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)

    count = 0
    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Draw bounding box
        if ok:
            count += 1
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            print(int(bbox[0]) + int(bbox[2]) / 2, int(bbox[1]) + int(bbox[3]) / 2)
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
        else :
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display tracker type on frame
        #cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display FPS on frame
        #cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display result
        cv2.imshow("Tracking", frame)
        cv2.imwrite(os.path.join(os.getcwd(), 'track result', str(count)+'.jpg'), frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break
        cv2.destroyAllWindows()
