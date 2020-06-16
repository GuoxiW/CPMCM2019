import numpy as np
import math
import cv2
from math import cos, pi, sin



def tuple_toint(point):
    return (int(point[0]), int(point[1]))

d = 10  #p1p2长度已知

#https://cloud.tencent.com/developer/ask/151193
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
    line1 = (dian[0],dian[1])
    line2 = (dian[2],dian[3])
    try:
        crosspoint1 = get_crosspoint(line1, line2)
    except Exception  :
        print('1111111111')
        line3 = (dian[0], dian[3])
        line4 = (dian[1], dian[2])
        crosspoint2 = get_crosspoint(line1, line2)
        dian0 = np.array(dian[0], dtype=float)
        dian1 = np.array(dian[1], dtype=float)
        crosspoint1 = np.array(crosspoint2, dtype=float) + dian1 - dian0
        return crosspoint1, crosspoint2

    print('crosspoint1',crosspoint1)
    crosspoint1_int = (int(crosspoint1[0]),int(crosspoint1[1]))

    line3 = (dian[0],dian[3])
    line4 = (dian[1],dian[2])
    try:
        crosspoint2 = get_crosspoint(line3, line4)
    except Exception:
        print('1111111112')
        dian0 = np.array(dian[0], dtype=float)
        dian3 = np.array(dian[3], dtype=float)
        crosspoint2 = np.array(crosspoint1, dtype=float) + (dian3 - dian0)*100
        crosspoint2 = tuple(crosspoint2)

    crosspoint2_int = (int(crosspoint2[0]), int(crosspoint2[1]))
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
    (x1, y1) = point1
    (x2, y2) = point2

    if x1==x2:
        y = y1
        print('y = y1:', y)
    else:
        #y = y1+(y1-y2)/(x1-x2)
        y = (y1  / (x - x1) -y2 / (x - x2))/(1  / (x - x1) -1 / (x - x2))
        #print('(y1-y2)/(x1-x2):',(y1-y2)/(x1-x2))
        #print('(y-y2)/(x-x2):', (y - y2) / (x - x2))
    '''
    if y1==y2:
        x = x1
    else:
        x = y1+(x1-x2)/(y1-y2)
    '''
    return (x,y)


def get_pointer_rad(img,center):
    '''获取角度'''
    shape = img.shape
    c_y, c_x, depth =int(center[0]), int(center[1]), 1 #    142 ,124, 1#, shape[2]
    x1=max(img.shape[0], img.shape[1])
    src = np.zeros([img.shape[0],img.shape[1],3])  #img.copy()
    src = img
    '''
    src[:,:,0] = img
    src[:, :, 1] = img
    src[:, :, 2] = img
    '''

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
    '''获取角度'''
    shape = img.shape
    c_y, c_x, depth =int(center[0]), int(center[1]), 1 #    142 ,124, 1#, shape[2]
    x1=max(img.shape[0], img.shape[1])
    src = np.zeros([img.shape[0],img.shape[1],3])  #img.copy()
    src = img

    freq_list = []
    #start_angle = start_angle#43
    step = (end_angle-start_angle)/200
    debug = 0
    point_info = {}
    mask1 = cv2.inRange(src, np.array([125, 125, 125]), np.array([255, 255, 255]))
    for i in range(0,200):
        #i = 100
        x = x1  * cos((start_angle +  step * i) * pi / 180) + c_x
        y = x1  * sin((start_angle + step * i) * pi / 180) + c_y
        temp = np.zeros(src.shape,dtype=np.uint8)#src.copy()

        #print(mask1.shape)
        #np_index = np.where(mask1 > 0)
        #print(type(np_index))

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
        #break
    print('len(freq_list):',len(freq_list))
    #point_info[freq_list[0][2]]
    #point_info[freq_list[len(freq_list) - 1][2]]
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


if __name__ == '__main__':

    img1 = cv2.imread('./data/240.png')

    #屋檐
    eaves_point = [[208,403],[210,434],[507,518],[507,484]]

    dian4_v1 = eaves_point

    src1 = np.float32(dian4_v1).reshape(-1, 1, 2)
    for i in range(4):
        cv2.putText(img1, 'v1_{}'.format(i), (src1[i][0][0], src1[i][0][1]), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0),
                    1)
        img1 = cv2.circle(img1, (src1[i][0][0], src1[i][0][1]), 5, (0, 0, 255), -1)

    line1 = (dian4_v1[0],dian4_v1[1])
    line2 = (dian4_v1[2],dian4_v1[3])
    crosspoint1_v1 = get_crosspoint(line1, line2)
    print('crosspoint1',crosspoint1_v1)
    crosspoint1_int_v1 = (int(crosspoint1_v1[0]),int(crosspoint1_v1[1]))
    img1 = cv2.circle(img1, (int(crosspoint1_v1[0]),int(crosspoint1_v1[1])), 5, (255, 0, 0), -1)

    line3 = (dian4_v1[0],dian4_v1[3])
    line4 = (dian4_v1[1],dian4_v1[2])
    crosspoint2_v1 = get_crosspoint(line3, line4)
    crosspoint2_int_v1 = (int(crosspoint2_v1[0]), int(crosspoint2_v1[1]))
    print('crosspoint2',crosspoint2_v1)
    img1 = cv2.circle(img1, (int(crosspoint2_v1[0]), int(crosspoint2_v1[1])), 5, (255, 0, 0), -1)


    roof_point = [539,469]
    img1 = cv2.circle(img1, (int(roof_point[0]), int(roof_point[1])), 5, (255, 0, 0), -1)
    #求根据灭点求过roof_point点的平行线
    line_cross_v1 = (crosspoint1_v1, crosspoint2_v1)
    vp_v01 = get_crosspoint((get_tuple_int(dian4_v1[0]),dian4_v1[1]), line_cross_v1)
    line_vp_roof = (vp_v01,roof_point)
    roof_point_toground = get_crosspoint((get_tuple_int(dian4_v1[1]),dian4_v1[2]), line_vp_roof)
    img1 = cv2.circle(img1, (int(roof_point_toground[0]), int(roof_point_toground[1])), 5, (255, 0, 0), -1)

    roof_point_tomid = get_crosspoint((get_tuple_int(dian4_v1[0]), dian4_v1[3]), line_vp_roof)
    img1 = cv2.circle(img1, (int(roof_point_tomid[0]), int(roof_point_tomid[1])), 5, (255, 0, 0), -1)

    distance0 = 3
    distance1 = compute_distance(roof_point, roof_point_toground) / compute_distance(dian4_v1[2], dian4_v1[3]) * distance0
    print('distance1:',distance1)

    distance1 = compute_distance(roof_point, roof_point_toground) / compute_distance(roof_point_tomid, roof_point_toground) * distance0
    print('distance1:',distance1)

    vp0 = get_crosspoint(line_cross_v1, (roof_point, roof_point_toground))
    cr = cross_ratio_v(x1=roof_point_toground, x2=roof_point_tomid, x3=roof_point, v=vp0)
    #cr = cross_ratio_v(x1=roof_point_tomid, x2=roof_point_toground, x3=roof_point, v=vp0)
    dis = 3
    distance2 = dis / cr
    print('distance2:', distance2)
    


    '''
    point_b = tuple(roof_point)
    point_a = tuple(roof_point_toground)
    line_ab = (point_a, point_b)
    point_rect = eaves_point
    crosspoint1_rect, crosspoint2_rect = get_crossline(eaves_point)
    line_cross_rect = (crosspoint1_rect, crosspoint2_rect)

    point_rect1 = tuple(point_rect[0])
    point_rect2 = tuple(point_rect[1])
    point_rect3 = tuple(point_rect[2])
    point_rect4 = tuple(point_rect[3])
    
    img1 = cv2.circle(img1, point_rect1, 2, (0, 0, 255), -1)
    img1 = cv2.circle(img1, point_rect2, 2, (0, 0, 255), -1)
    img1 = cv2.circle(img1, point_rect3, 2, (0, 0, 255), -1)
    img1 = cv2.circle(img1, point_rect4, 2, (0, 0, 255), -1)

    cv2.putText(img1, 'p1', point_rect1 ,cv2.FONT_HERSHEY_COMPLEX, 1,  (255, 255, 0), 1)
    cv2.putText(img1, 'p2', point_rect2, cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 1)
    cv2.putText(img1, 'p3', point_rect3, cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 1)
    cv2.putText(img1, 'p4', point_rect4, cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 1)
    
    line41_rect = (point_rect4, point_rect1)
    vp41 = get_crosspoint(line41_rect, line_cross_rect)
    line43_rect = (point_rect4, point_rect3)
    vp43 = get_crosspoint(line43_rect, line_cross_rect)

    point_b1 = get_crosspoint(line_ab, line41_rect)
    point_b2 = get_crosspoint(line_ab, line43_rect)
    print('point_b1 point_b2:', point_b1, point_b2)

    # 在line24上线段point_b1 point_rect2 point_rect4
    cr = cross_ratio_v(x1=point_rect4, x2=point_b1, x3=point_rect1, v=vp41)
    dis = 3
    distance_b1_2 = dis / cr
    # 在line34上线段point_b2 point_rect3 point_rect4
    cr = cross_ratio_v(x1=point_rect4, x2=point_b2, x3=point_rect3, v=vp43)
    cr2 = cross_ratio_v(x1=point_rect4, x2=point_rect3, x3=point_b2, v=vp43)
    print(cr, cr2, 1 / cr)
    dis = 20
    distance_b2_2 = dis / cr
    disrance_b1_b2 = np.sqrt(np.square(distance_b1_2) + np.square(distance_b2_2))

    dis = disrance_b1_b2
    print('disrance_b1_b2:', disrance_b1_b2, distance_b1_2, distance_b2_2)
    vp_ab = get_crosspoint(line_ab, line_cross_rect)
    # dis = 1
    cr1, cr2, x3_x4 = cross_ratio_4point_v(x1=point_b1, x2=point_b2, x3=point_a, x4=point_b, v=vp_ab, distance=dis)
    print('cr1*cr2', cr1 * cr2, x3_x4)
    disrance_ab = dis / cr1 / cr2
    print('distance_ab:', disrance_ab, cr1, cr2)
    '''


    roof_point2 = [765,543]
    img1 = cv2.circle(img1, (int(roof_point2[0]), int(roof_point2[1])), 5, (255, 0, 0), -1)
    #求根据灭点求过roof_point点的平行线
    line_cross_v1 = (crosspoint1_v1, crosspoint2_v1)
    vp_v01 = get_crosspoint((get_tuple_int(dian4_v1[0]),dian4_v1[1]), line_cross_v1)
    line_vp_roof = (vp_v01,roof_point2)
    roof_point_toground = get_crosspoint((get_tuple_int(dian4_v1[1]),dian4_v1[2]), line_vp_roof)
    img1 = cv2.circle(img1, (int(roof_point_toground[0]), int(roof_point_toground[1])), 5, (255, 0, 0), -1)

    roof_point_tomid = get_crosspoint((get_tuple_int(dian4_v1[0]), dian4_v1[3]), line_vp_roof)
    img1 = cv2.circle(img1, (int(roof_point_tomid[0]), int(roof_point_tomid[1])), 5, (255, 0, 0), -1)


    vp0 = get_crosspoint(line_cross_v1, (roof_point2, roof_point_toground))
    cr = cross_ratio_v(x1=roof_point_toground, x2=roof_point_tomid, x3=roof_point2, v=vp0)
    print('cr=',cr)
    dis = 3
    distance2 = dis / cr
    print('distance2:', distance2)

    distance0 = 3
    distance1 = compute_distance(roof_point2, roof_point_toground) / compute_distance(dian4_v1[2], dian4_v1[3]) * distance0
    print('distance1:',distance1)






    cv2.namedWindow('img1', 0)
    print(img1.shape)
    cv2.imshow( "img1", img1 )  # 显示
    cv2.waitKey(0)  # 按下任意键退出
    cv2.destroyAllWindows()
