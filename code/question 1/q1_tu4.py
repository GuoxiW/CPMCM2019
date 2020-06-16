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

def my_drawline(img,point1,point2,color,thickness):
    img = cv2.line(img,  (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])) , color, thickness=thickness)
    return img

def my_circle(img,point,radius,color,width):
    img = cv2.circle(img,(int(point[0]),int(point[1])),radius,color,width)
    return img

if __name__ == '__main__':

    img1 = cv2.imread('./img4.png')

    point_A = [223,1421]
    point_B = [1227,1384]
    point_C = [1025, 280]
    point_D = [355,286]

    point_door = [[76, 518], [83, 583], [114, 584] ,[108, 519]]
    point_baluster = [[76, 518], [83, 583], [114, 584] ,[108, 519]]

    #point_A2 = [550,1382]
    #point_B2 = [914,1370]
    #point_C2 = [870, 754]
    #point_D2 = [547,761]

    point_A1 = [553,1380]
    point_B1 = [911,1369]
    point_C1 = [869, 759]
    point_D1 = [549, 764]

    point_A1 = get_point_inline(point_A, point_B, point_A1[0])
    point_B1 = get_point_inline(point_A, point_B, point_B1[0])


    dian4_v1 = [point_A1,point_B1,point_C1,point_D1]

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



    dian4_v2 = [point_A1, point_B1, point_C1, point_D1]
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







    line1 = (point_A,point_B)
    line2 = (point_C,point_D)
    crossline = (crosspoint1_v2,crosspoint2_v2)
    crosspoint_ab = get_crosspoint(line1, crossline)
    print('crosspoint1',crosspoint_ab)
    crosspoint2 = get_crosspoint(line2, crossline)
    print('crosspoint2',crosspoint2)

    line_vp_A = (point_A,crosspoint2)
    line_AD = (point_A,point_D)
    point_A_new = get_crosspoint(line_AD, line_vp_A)
    print('point_A_new:',point_A_new)
    print('point_A:', point_A)

    point_A = point_A_new

    cr = cross_ratio_v(x1=point_B, x2=point_A, x3=point_A1, v = crosspoint_ab)
    distance_AB = 513.43
    distance_B_A1 = distance_AB / cr

    cr = cross_ratio_v(x1=point_B, x2=point_A, x3=point_B1, v = crosspoint_ab)
    dis = distance_AB
    distance_B_B1 = dis / cr
    distance_A1_B1 = distance_B_A1-distance_B_B1
    print('distance:',distance_A1_B1,distance_B_A1,distance_B_B1)
    radius = distance_A1_B1/2


    point_diameter_left = (548,426)
    line_B1C1 = (point_B1,point_C1)
    line_diameter_vp = (point_diameter_left,crosspoint_ab)
    point_diameter_right = get_crosspoint(line_B1C1, line_diameter_vp)

    img1 = my_circle(img1, point_diameter_left, 20, (255, 255, 0), -1)
    img1 = my_circle(img1, point_diameter_right, 20, (255, 255, 0), -1)

    #求圆心
    print(point_diameter_left,point_diameter_right)
    cr = 2
    x_start = point_diameter_left[0]
    x_end = point_diameter_right[0]
    x = x_start
    min_gap = 10
    min_point = x_start
    while x < x_end:
        x +=0.5
        round_center = get_point_inline(point_diameter_left, point_diameter_right, x)
        cr = cross_ratio_v(x1=point_diameter_left, x2=point_diameter_right, x3=round_center, v = crosspoint_ab)
        if abs(cr -2)<min_gap:
            min_gap = abs(cr -2)
            min_point = round_center
    round_center = min_point
    print('round_center',round_center)
    img1 = my_circle(img1, min_point, 5, (255, 255, 255), -1)



    line1 = (point_A1,point_D1)
    line2 = (point_B1,point_C1)
    crossline = (crosspoint1_v2,crosspoint2_v2)
    crosspoint_a1d1 = get_crosspoint(line1, crossline)

    img1 = my_drawline(img1,round_center,crosspoint_a1d1,(255,0,255),1)

    point_circle1 = (694,323)
    my_circle(img1, point_circle1, 5, (255, 255, 255), -1)

    line_AB = (point_A,point_B)
    line_CD = (point_C, point_D)
    line_perpendicular = (point_circle1,round_center)
    point_AB_perpendicular = get_crosspoint(line_AB, line_perpendicular)
    point_CD_perpendicular = get_crosspoint(line_CD, line_perpendicular)
    img1 = my_circle(img1, point_AB_perpendicular, 5, (0, 255, 255), -1)
    img1 = my_circle(img1, point_CD_perpendicular, 5, (0, 255, 255), -1)
    img1 = my_drawline(img1, point_AB_perpendicular, point_CD_perpendicular, (0, 0, 255), 1)

    crossline = (crosspoint1_v2, crosspoint2_v2)
    crosspoint_perpendicular = get_crosspoint(line_perpendicular, crossline)
    print('crosspoint_perpendicular:',crosspoint_a1d1,crosspoint_perpendicular)
    cr = cross_ratio_v(x1=round_center, x2=point_circle1, x3=point_AB_perpendicular, v = crosspoint_a1d1)
    print('cr ab',cr)
    distance_AB_round_center = radius / cr

    cr = cross_ratio_v(x1=round_center, x2=point_circle1, x3=point_CD_perpendicular, v = crosspoint_a1d1)
    print('cr cd', cr)
    distance_CD_round_center = radius / cr
    print('radius=',radius)
    print('distance_AB,CD=', distance_AB_round_center,distance_CD_round_center)
    print('distance_AB_CD=',distance_AB_round_center+distance_CD_round_center)

    dis_x3x4 = cross_ratio_4point_v(x1=round_center, x2=point_circle1,\
                    x3=point_AB_perpendicular, x4=point_CD_perpendicular,\
                                 v=crosspoint_a1d1, distance=radius)
    print('dis_x3x4=', dis_x3x4)
    _,_,dis_x3x4 = cross_ratio_4point_v(x1=point_circle1, x2=round_center,\
                    x3=point_AB_perpendicular, x4=point_CD_perpendicular,\
                                 v=crosspoint_a1d1, distance=radius)
    print('dis_x3x4=', dis_x3x4)
    distance_AB_CD = dis_x3x4


    line_A_vp = (point_A,crosspoint_a1d1)
    point_A_toCD = get_crosspoint(line_A_vp,(point_C,point_D))
    line_B_vp = (point_B,crosspoint_a1d1)
    point_B_toCD = get_crosspoint(line_B_vp,(point_C,point_D))
    img1 = my_circle(img1, point_A_toCD, 15, (0, 0, 255), -1)
    img1 = my_circle(img1, point_B_toCD , 15, (0, 0, 255), -1)
    print('point_A,B_toCD:',point_A_toCD,point_B_toCD)

    crossline = (crosspoint1_v2, crosspoint2_v2)
    crosspoint_CD = get_crosspoint((point_C, point_D), crossline)
    print('crosspoint_CD =',crosspoint_CD )
    crosspoint_CD = get_crosspoint((point_A,point_B), crossline)
    print('crosspoint_CD =', crosspoint_CD)
    dis_x3x4 = cross_ratio_4point_v(x1=point_A_toCD, x2=point_B_toCD,\
                    x3=point_C, x4=point_D,\
                                 v=crosspoint_CD, distance=radius)
    print('dis_CD=', dis_x3x4)

    cr = cross_ratio_v(x1=point_A_toCD, x2=point_B_toCD, x3=point_D, v=crosspoint_CD)
    print('cr cd', cr)
    distance_Acd_D = distance_AB / cr
    print('distance_Acd_D:',distance_Acd_D)

    cr = cross_ratio_v(x1=point_A_toCD, x2=point_B_toCD, x3=point_C, v=crosspoint_CD)
    print('cr cd', cr)
    distance_Acd_C = distance_AB / cr
    print('distance_Acd_C:',distance_Acd_C)
    distance_CD = distance_Acd_C - distance_Acd_D

    print('AB距离(cm)：', distance_AB )
    print('CD距离(cm)：', distance_CD )
    print('AB到CD距离(cm)：', distance_AB_CD)



    cv2.namedWindow('img1', 0)
    cv2.imshow( "img1", img1 )  # 显示
    cv2.imwrite("img4_result.png", img1)
    cv2.waitKey(0)  # 按下任意键退出
    cv2.destroyAllWindows()
