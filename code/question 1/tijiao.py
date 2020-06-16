import numpy as np
import math
import cv2
from math import cos, pi, sin

def tuple_toint(point):
    #绘图前需要整数
    return (int(point[0]), int(point[1]))

def get_crosspoint(line1,line2):
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


def get_crossline(dian):
    #求灭线
    line1 = (dian[0],dian[1])
    line2 = (dian[2],dian[3])
    crosspoint1 = get_crosspoint(line1, line2)

    line3 = (dian[0],dian[3])
    line4 = (dian[1],dian[2])
    crosspoint2 = get_crosspoint(line3, line4)
    return crosspoint1,crosspoint2

def get_tuple_int(point):
    return ( int(point[0]), int(point[1]) )

def compute_distance(point1, point2):
    (x1, y1) = point1
    (x2, y2) = point2
    d = np.sqrt(np.square(x1-x2) + np.square(y1-y2))
    return d

def getAngle(point1, point2):
    #两点的x、y值
    (px1 ,py1) = point1
    (px2 ,py2) = point2
    x = px2-px1
    y = py2-py1
    hypotenuse = np.sqrt(np.square(x)+np.square(y))
    #斜边长度
    cos = x/hypotenuse
    radian = math.acos(cos)
    radian = math.acos(cos)
    if radian==0:
        return 0
    #求出弧度
    angle = 180/(np.pi/radian)
    #用弧度算出角度
    if (y<0) :
        angle = -angle
    elif (y == 0) and (x<0):
        angle = 180
    return angle


def get_point_inline(point1,point2,x):
    #计算point1,point2确定的直线上取x值时的点
    (x1, y1) = point1
    (x2, y2) = point2

    if x1==x2:
        y = y1
        print('y = y1:', y)
    else:
        y = (y1  / (x - x1) -y2 / (x - x2))/(1  / (x - x1) -1 / (x - x2))
    return (x,y)


def get_pointer_rad(img,center):
    '''获取角度  求切线使用 粗定位'''
    c_y, c_x, depth =int(center[0]), int(center[1]), 1 #    142 ,124, 1#, shape[2]
    x1=max(img.shape[0], img.shape[1])
    src = np.zeros([img.shape[0],img.shape[1],3])  #img.copy()
    src = img

    freq_list = []
    start_angle = 0
    debug = 0
    for i in range(start_angle,start_angle+359):
        x = x1  * cos(i * pi / 180) + c_x
        y = x1  * sin(i * pi / 180) + c_y
        temp = np.zeros(src.shape)#src.copy()
        cv2.line(temp, (c_x, c_y), (int(x), int(y)), (0, 0, 255), thickness=1)
        t1 = img.copy()
        if debug:
            cv2.imshow('t10', t1)
        t1[temp[:, :,2] == 255] = 255
        if debug:
            cv2.imshow('t1', t1)
        c = img[temp[:, :,2] ==255]

        points = c[c == 255]
        if len(points)>0:
            freq_list.append((len(points), i))
        if debug:
            cv2.imshow('d', temp)
            cv2.imshow('d1', t1)
            cv2.waitKey(1)
    print('len(freq_list):',len(freq_list))
    return freq_list[0][1],freq_list[len(freq_list)-1][1]


def get_pointer_rad2(img,center,start_angle, end_angle):
    '''获取角度  求切线使用  精定位'''
    c_y, c_x, depth =int(center[0]), int(center[1]), 1
    x1=max(img.shape[0], img.shape[1])
    src = img

    freq_list = []
    step = (end_angle-start_angle)/200
    debug = 0
    point_info = {}
    mask1 = cv2.inRange(src, np.array([125, 125, 125]), np.array([255, 255, 255]))
    for i in range(0,200):
        x = x1  * cos((start_angle +  step * i) * pi / 180) + c_x
        y = x1  * sin((start_angle + step * i) * pi / 180) + c_y
        temp = np.zeros(src.shape,dtype=np.uint8)

        temp = cv2.line(temp, (c_x, c_y), (int(x), int(y)), (0, 0, 255), thickness=1)
        img_masked = cv2.bitwise_and(temp, temp, mask=mask1)
        np_index = np.where(img_masked > 0)
        size = np_index[0].size
        if size>0:
            freq_list.append((size, start_angle + i*step,i))
            point_info[i] = np_index
        if debug:
            cv2.imshow('d', temp)
            cv2.waitKey(1)
    return freq_list[0][1],freq_list[len(freq_list)-1][1],point_info[freq_list[0][2]],point_info[freq_list[len(freq_list)-1][2]]
def add_tuple(point1, point2):
    return (point1[0] + point2[0], point1[1] + point2[1])


#直线的垂直向量
def  get_vertical_vector(point1,point2):
    vector = (point1[0] - point2[0] , point1[1] - point2[1])
    if vector[0] == 0:
        vertical_vector = (1,0)
    else:
        vertical_vector = (-vector[1]/vector[0], 1)
    return vertical_vector

#计算三角形垂心
def  get_perpendicular_center(point1,point2,point3):
    vertical_vector1_3 = get_vertical_vector(point1, point3)
    line_v1_3 = (point2, add_tuple(point2,vertical_vector1_3))

    vertical_vector2_3 = get_vertical_vector(point2, point3)
    line_v2_3 = (point1,add_tuple(point1,vertical_vector2_3))
    return get_crosspoint(line_v1_3, line_v2_3)

#得到垂足点
def get_pedal_point(line,point):
    vertical_vector = get_vertical_vector(line[0], line[1])
    line_v = (point, add_tuple(point, vertical_vector))
    return get_crosspoint(line,line_v)



def compute_distance(point1, point2):
    (x1, y1) = point1
    (x2, y2) = point2
    d = np.sqrt(np.square(x1-x2) + np.square(y1-y2))
    return d

#计算交并比
def cross_ratio(c1, c2, c3, c4):
    cr = (compute_distance(c1, c3) * compute_distance(c2, c4)) / (compute_distance(c2, c3) * compute_distance(c1, c4))
    return cr
#计算交并比
def cross_ratio_v(x1, x2, x3, v):
    cr = (compute_distance(x1, x2) * compute_distance(x3, v)) / (compute_distance(x1, x3) * compute_distance(x2, v))
    return cr

def cross_ratio_4point_v(x1, x2 , x3, x4 , v ,distance ):
    #已经知道x1,x2,求x3,x4
    cr1 = cross_ratio_v(x1, x2, x3, v)
    x1_x2 = distance
    x1_x3 = x1_x2 / cr1
    cr2 = cross_ratio_v(x3, x4, x1, v)
    x3_x4 = x1_x3 / cr2
    return cr1,cr2,x3_x4

def solve_height_demo():
    img1 = cv2.imread('../q4/data/170.png')

    point_1 = [[218, 434], [219, 464], [554, 516], [555, 481]]
    dian4_v1 = point_1

    src1 = np.float32(dian4_v1).reshape(-1, 1, 2)
    for i in range(4):
        cv2.putText(img1, 'v1_{}'.format(i), (src1[i][0][0], src1[i][0][1]), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 1)
        img1 = cv2.circle(img1, (src1[i][0][0], src1[i][0][1]), 3, (0, 0, 255), -1)

    line1 = (dian4_v1[0], dian4_v1[1])
    line2 = (dian4_v1[2], dian4_v1[3])
    crosspoint1_v1 = get_crosspoint(line1, line2)
    print('crosspoint1', crosspoint1_v1)
    crosspoint1_int_v1 = (int(crosspoint1_v1[0]), int(crosspoint1_v1[1]))
    img1 = cv2.circle(img1, (int(crosspoint1_v1[0]), int(crosspoint1_v1[1])), 20, (255, 0, 0), -1)

    line3 = (dian4_v1[0], dian4_v1[3])
    line4 = (dian4_v1[1], dian4_v1[2])
    crosspoint2_v1 = get_crosspoint(line3, line4)
    crosspoint2_int_v1 = (int(crosspoint2_v1[0]), int(crosspoint2_v1[1]))
    print('crosspoint2', crosspoint2_v1)
    img1 = cv2.circle(img1, (int(crosspoint2_v1[0]), int(crosspoint2_v1[1])), 20, (255, 0, 0), -1)

    dian4_h = [[82, 430], [18, 482], [732, 605], [728, 553]]
    src1 = np.float32(dian4_h).reshape(-1, 1, 2)
    for i in range(4):
        cv2.putText(img1, 'h_{}'.format(i), (src1[i][0][0], src1[i][0][1]), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0),
                    1)
        img1 = cv2.circle(img1, (src1[i][0][0], src1[i][0][1]), 20, (0, 0, 255), -1)

    crosspoint1_h, crosspoint2_h = get_crossline(dian4_h)
    crosspoint1_h_int = (int(crosspoint1_h[0]), int(crosspoint1_h[1]))
    crosspoint2_h_int = (int(crosspoint2_h[0]), int(crosspoint2_h[1]))
    img1 = cv2.circle(img1, crosspoint1_h_int, 20, (255, 255, 0), -1)
    img1 = cv2.circle(img1, crosspoint2_h_int, 20, (255, 255, 0), -1)
    print('crosspoint_v2:', crosspoint1_h, crosspoint2_h)

    line_cross_v1 = (crosspoint1_v1, crosspoint2_v1)

    point_mid_img = (int(img1.shape[0] / 2), int(img1.shape[1] / 2))
    img1 = cv2.circle(img1, (point_mid_img[0], point_mid_img[1]), 40, (255, 0, 255), -1)

    line_cross_h = (crosspoint1_h, crosspoint2_h)
    point_mid_img_cross = get_pedal_point(line_cross_h, point_mid_img)

    img1 = cv2.circle(img1, (int(point_mid_img_cross[0]), int(point_mid_img_cross[1])), 40, (255, 0, 255), -1)
    line_main = (point_mid_img_cross,point_mid_img)
    point_camera_projection = get_crosspoint(line_main, line_cross_v1)
    print('point_camera_projection:',point_camera_projection)


    line_hight = (dian4_v1[0], dian4_v1[1])
    height_crosspoint = get_crosspoint(line_cross_h, line_hight)
    vp_height = get_crosspoint(line_cross_v1, line_hight)

    cr = cross_ratio_v(x1=dian4_v1[0], x2=dian4_v1[1], x3=height_crosspoint, v=vp_height)
    print('height_crosspoint:', height_crosspoint)
    print('cr:',cr)
    dis = 3
    hight_aircraft =  dis / cr
    print('hight_aircraft:',hight_aircraft)


    cv2.namedWindow('img1', 0)
    cv2.imshow( "img1", img1 )  # 显示
    cv2.waitKey(0)  # 按下任意键退出
    cv2.destroyAllWindows()


def solve_figure_demo():

    img1 = cv2.imread('../img/img3.png')


    point_door = [[76, 518], [83, 583], [114, 584] ,[108, 519]]
    point_baluster = [[76, 518], [83, 583], [114, 584] ,[108, 519]]

    #垂直面点
    dian4_v1 = point_door
    src1 = np.float32(dian4_v1).reshape(-1, 1, 2)
    for i in range(4):
        cv2.putText(img1, 'v1_{}'.format(i), (src1[i][0][0], src1[i][0][1]), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0),
                    1)
        img1 = cv2.circle(img1, (src1[i][0][0], src1[i][0][1]), 20, (0, 0, 255), -1)

    line1 = (dian4_v1[0],dian4_v1[1])
    line2 = (dian4_v1[2],dian4_v1[3])
    crosspoint1_v1 = get_crosspoint(line1, line2)
    print('crosspoint1',crosspoint1_v1)
    crosspoint1_int_v1 = (int(crosspoint1_v1[0]),int(crosspoint1_v1[1]))
    img1 = cv2.circle(img1, (int(crosspoint1_v1[0]),int(crosspoint1_v1[1])), 20, (255, 0, 0), -1)

    line3 = (dian4_v1[0],dian4_v1[3])
    line4 = (dian4_v1[1],dian4_v1[2])
    crosspoint2_v1 = get_crosspoint(line3, line4)
    crosspoint2_int_v1 = (int(crosspoint2_v1[0]), int(crosspoint2_v1[1]))
    print('crosspoint2',crosspoint2_v1)
    img1 = cv2.circle(img1, (int(crosspoint2_v1[0]), int(crosspoint2_v1[1])), 20, (255, 0, 0), -1)

    #垂直面点2
    dian4_v2 = point_baluster
    src1 = np.float32(dian4_v2).reshape(-1, 1, 2)
    for i in range(4):
        cv2.putText(img1, 'h_{}'.format(i), (src1[i][0][0], src1[i][0][1]), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0),
                    1)
        img1 = cv2.circle(img1, (src1[i][0][0], src1[i][0][1]), 20, (0, 0, 255), -1)

    crosspoint1_v2,crosspoint2_v2 = get_crossline(dian4_v2)
    crosspoint1_v2_int = (int(crosspoint1_v2[0]), int(crosspoint1_v2[1]))
    crosspoint2_v2_int = (int(crosspoint2_v2[0]), int(crosspoint2_v2[1]))
    img1 = cv2.circle(img1, crosspoint1_v2_int, 20, (255, 255, 0), -1)
    img1 = cv2.circle(img1, crosspoint2_v2_int, 20, (255, 255, 0), -1)
    print('crosspoint_v2:',crosspoint1_v2,crosspoint2_v2)

    dian4_v2 = point_baluster
    src1 = np.float32(dian4_v2).reshape(-1, 1, 2)
    for i in range(4):
        img1 = cv2.circle(img1, (src1[i][0][0], src1[i][0][1]), 20, (0, 0, 255), -1)

    crosspoint1_v2,crosspoint2_v2 = get_crossline(dian4_v2)
    crosspoint1_v2_int = (int(crosspoint1_v2[0]), int(crosspoint1_v2[1]))
    crosspoint2_v2_int = (int(crosspoint2_v2[0]), int(crosspoint2_v2[1]))
    img1 = cv2.circle(img1, crosspoint1_v2_int, 20, (255, 255, 0), -1)
    img1 = cv2.circle(img1, crosspoint2_v2_int, 20, (255, 255, 0), -1)
    print('crosspoint_v2:',crosspoint1_v2,crosspoint2_v2)

    #水平面点
    dian4_h = [[660, 1226], [795, 1360], [1124, 1364], [944, 1231]]
    src1 = np.float32(dian4_h).reshape(-1, 1, 2)
    for i in range(4):
        img1 = cv2.circle(img1, (src1[i][0][0], src1[i][0][1]), 20, (0, 0, 255), -1)

    crosspoint1_h,crosspoint2_h = get_crossline(dian4_h)
    crosspoint1_h_int = (int(crosspoint1_h[0]), int(crosspoint1_h[1]))
    crosspoint2_h_int = (int(crosspoint2_h[0]), int(crosspoint2_h[1]))
    img1 = cv2.circle(img1, crosspoint1_h_int, 20, (255, 255, 0), -1)
    img1 = cv2.circle(img1, crosspoint2_h_int, 20, (255, 255, 0), -1)
    print('crosspoint_v2:',crosspoint1_h,crosspoint2_h)

    line_cross_v1 = (crosspoint1_v1, crosspoint2_v1)
    line_cross_v2 = (crosspoint1_v2,crosspoint2_v2)

    point_mid_img = (int(img1.shape[1] / 2), int(img1.shape[0] / 2))
    print('point_mid_img:', point_mid_img)
    img1 = cv2.circle(img1, (point_mid_img[0], point_mid_img[1]), 40, (255, 0, 255), -1)

    line_cross_h = (crosspoint1_h,crosspoint2_h)
    point_mid_img_cross = get_pedal_point(line_cross_h, point_mid_img)
    img1 = cv2.circle(img1, (int(point_mid_img_cross[0]), int(point_mid_img_cross[1])), 40, (255, 0, 255), -1)
    line_main = (point_mid_img_cross,point_mid_img)
    point_camera_projection = get_crosspoint(line_main, line_cross_v1)
    print('point_camera_projection:',point_camera_projection)
    point_camera_projection = get_crosspoint(line_main, line_cross_v2)
    print('point_camera_projection:',point_camera_projection)

    #已知矩形，求取任意2点距离
    point_b = point_camera_projection
    point_a = (413,709)
    line_ab = (point_a,point_b)

    point_rect1 = (1242, 1139)
    point_rect2 = (1159, 1072)
    point_rect3 = (1205, 1229)
    point_rect4 = (1116, 1144)

    img1 = cv2.circle(img1, point_rect1, 20, (0, 0, 255), -1)
    img1 = cv2.circle(img1, point_rect2, 20, (0, 0, 255), -1)
    img1 = cv2.circle(img1, point_rect3, 20, (0, 0, 255), -1)
    img1 = cv2.circle(img1, point_rect4, 20, (0, 0, 255), -1)

    cv2.putText(img1, 'p1', point_rect1 ,cv2.FONT_HERSHEY_COMPLEX, 3,  (255, 255, 0), 5)
    cv2.putText(img1, 'p2', point_rect2, cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 0), 5)
    cv2.putText(img1, 'p3', point_rect3, cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 0), 5)
    cv2.putText(img1, 'p4', point_rect4, cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 0), 5)

    line24_rect = (point_rect2, point_rect4)
    vp24 = get_crosspoint(line24_rect, line_cross_h)
    line21_rect = (point_rect2, point_rect1)
    vp21 = get_crosspoint(line21_rect, line_cross_h)


    point_b1 = get_crosspoint(line_ab, line24_rect)
    point_b2 = get_crosspoint(line_ab, line21_rect)
    print('point_b1 point_b2:',point_b1,point_b2)

    #在line24上线段point_b1 point_rect2 point_rect4
    cr = cross_ratio_v(x1=point_rect2, x2=point_b1, x3=point_rect4, v = vp24)
    dis = 0.9
    distance_b1_2 = dis * cr
    #在line34上线段point_b2 point_rect3 point_rect4
    cr = cross_ratio_v(x1=point_rect2, x2=point_b2, x3=point_rect1, v=vp21)
    cr2 = cross_ratio_v(x1=point_rect2, x2=point_rect1, x3=point_b2, v=vp21)
    print(cr,cr2,1/cr)
    dis = 0.9
    distance_b2_2 = dis * cr
    disrance_b1_b2 = np.sqrt(np.square(distance_b1_2)+np.square(distance_b2_2))

    dis = disrance_b1_b2
    print('disrance_b1_b2:',disrance_b1_b2)
    vp_ab = get_crosspoint(line_ab, line_cross_h)
    cr1,cr2,x3_x4 = cross_ratio_4point_v(x1 = point_b1, x2=point_b2, x3=point_a, x4=point_b, v = vp_ab ,distance =dis)
    print('cr1*cr2',cr1*cr2,x3_x4)
    disrance_ab = dis/cr1/cr2
    print('disrance_ab:',disrance_ab,cr1,cr2)



    cv2.namedWindow('img1', 0)
    cv2.imshow( "img1", img1 )
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    solve_figure_demo()  #求解图三

    #solve_height_demo()  #求解飞机高度



