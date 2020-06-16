import numpy as np
import math
import cv2
from math import cos, pi, sin


def tuple_toint(point):
    return (int(point[0]), int(point[1]))


d = 10  # p1p2长度已知


# https://cloud.tencent.com/developer/ask/151193
def get_crosspoint(line1, line2):
    # 求取相交点  line输入样例格式(x,y)
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])  # Typo was here

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
    line1 = (dian[0], dian[1])
    line2 = (dian[2], dian[3])
    crosspoint1 = get_crosspoint(line1, line2)
    print('crosspoint1', crosspoint1)
    crosspoint1_int = (int(crosspoint1[0]), int(crosspoint1[1]))

    line3 = (dian[0], dian[3])
    line4 = (dian[1], dian[2])
    crosspoint2 = get_crosspoint(line3, line4)
    crosspoint2_int = (int(crosspoint2[0]), int(crosspoint2[1]))
    return crosspoint1, crosspoint2


def get_tuple_int(point):
    return (int(point[0]), int(point[1]))


def compute_distance(point1, point2):
    (x1, y1) = point1
    (x2, y2) = point2
    d = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))
    return d


def getAngle(point1, point2):
    # 两点的x、y值
    (px1, py1) = point1
    (px2, py2) = point2
    x = px2 - px1
    y = py2 - py1
    hypotenuse = np.sqrt(np.square(x) + np.square(y))
    # 斜边长度
    cos = x / hypotenuse
    radian = math.acos(cos)
    radian = math.acos(cos)
    if radian == 0:
        return 0
    # 求出弧度
    angle = 180 / (np.pi / radian)
    # 用弧度算出角度
    if (y < 0):
        angle = -angle
    elif (y == 0) and (x < 0):
        angle = 180
    return angle


def get_point_inline(point1, point2, x):
    (x1, y1) = point1
    (x2, y2) = point2

    if x1 == x2:
        y = y1
        print('y = y1:', y)
    else:

        y = (y1 / (x - x1) - y2 / (x - x2)) / (1 / (x - x1) - 1 / (x - x2))

    return (x, y)


def get_pointer_rad(img, center):
    '''获取角度'''
    shape = img.shape
    c_y, c_x, depth = int(center[0]), int(center[1]), 1  # 142 ,124, 1#, shape[2]
    x1 = max(img.shape[0], img.shape[1])
    src = np.zeros([img.shape[0], img.shape[1], 3])  # img.copy()
    src = img
    '''
    src[:,:,0] = img
    src[:, :, 1] = img
    src[:, :, 2] = img
    '''

    freq_list = []
    start_angle = 0
    debug = 0
    for i in range(start_angle, start_angle + 359):
        x = x1 * cos(i * pi / 180) + c_x
        y = x1 * sin(i * pi / 180) + c_y
        temp = np.zeros(src.shape)  # src.copy()
        cv2.line(temp, (c_x, c_y), (int(x), int(y)), (0, 0, 255), thickness=1)
        t1 = img.copy()
        if debug:
            cv2.imshow('t10', t1)
        t1[temp[:, :, 2] == 255] = 255
        if debug:
            cv2.imshow('t1', t1)
        c = img[temp[:, :, 2] == 255]

        points = c[c == 255]
        if len(points) > 0:
            freq_list.append((len(points), i))
        if debug:
            cv2.imshow('d', temp)
            cv2.imshow('d1', t1)
            cv2.waitKey(1)
    print('len(freq_list):', len(freq_list))
    return freq_list[0][1], freq_list[len(freq_list) - 1][1]


def get_pointer_rad2(img, center, start_angle, end_angle):
    '''获取角度'''
    shape = img.shape
    c_y, c_x, depth = int(center[0]), int(center[1]), 1  # 142 ,124, 1#, shape[2]
    x1 = max(img.shape[0], img.shape[1])
    src = np.zeros([img.shape[0], img.shape[1], 3])  # img.copy()
    src = img

    freq_list = []
    # start_angle = start_angle#43
    step = (end_angle - start_angle) / 200
    debug = 0
    point_info = {}
    mask1 = cv2.inRange(src, np.array([125, 125, 125]), np.array([255, 255, 255]))
    for i in range(0, 200):
        # i = 100
        x = x1 * cos((start_angle + step * i) * pi / 180) + c_x
        y = x1 * sin((start_angle + step * i) * pi / 180) + c_y
        temp = np.zeros(src.shape, dtype=np.uint8)  # src.copy()

        # print(mask1.shape)
        # np_index = np.where(mask1 > 0)
        # print(type(np_index))

        temp = cv2.line(temp, (c_x, c_y), (int(x), int(y)), (0, 0, 255), thickness=1)
        img_masked = cv2.bitwise_and(temp, temp, mask=mask1)
        np_index = np.where(img_masked > 0)
        size = np_index[0].size
        if size > 0:
            freq_list.append((size, start_angle + i * step, i))
            point_info[i] = np_index
        if debug:
            cv2.imshow('d', temp)
            cv2.waitKey(1)
        # break
    print('len(freq_list):', len(freq_list))
    # point_info[freq_list[0][2]]
    # point_info[freq_list[len(freq_list) - 1][2]]
    return freq_list[0][1], freq_list[len(freq_list) - 1][1], point_info[freq_list[0][2]], point_info[
        freq_list[len(freq_list) - 1][2]]


def add_tuple(point1, point2):
    return (point1[0] + point2[0], point1[1] + point2[1])


# 直线的垂直向量
def get_vertical_vector(point1, point2):
    vector = (point1[0] - point2[0], point1[1] - point2[1])
    if vector[0] == 0:
        vertical_vector = (1, 0)
    else:
        vertical_vector = (-vector[1] / vector[0], 1)
    return vertical_vector


# 计算三角形垂心
def get_perpendicular_center(point1, point2, point3):
    vertical_vector1_3 = get_vertical_vector(point1, point3)
    line_v1_3 = (point2, add_tuple(point2, vertical_vector1_3))

    vertical_vector2_3 = get_vertical_vector(point2, point3)
    line_v2_3 = (point1, add_tuple(point1, vertical_vector2_3))
    return get_crosspoint(line_v1_3, line_v2_3)


# 得到垂足点
def get_pedal_point(line, point):
    vertical_vector = get_vertical_vector(line[0], line[1])
    line_v = (point, add_tuple(point, vertical_vector))
    return get_crosspoint(line, line_v)


def compute_distance(point1, point2):
    (x1, y1) = point1
    (x2, y2) = point2
    d = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))
    return d


def cross_ratio(c1, c2, c3, c4):
    cr = (compute_distance(c1, c3) * compute_distance(c2, c4)) / (compute_distance(c2, c3) * compute_distance(c1, c4))
    return cr


def cross_ratio_v(x1, x2, x3, v):
    cr = (compute_distance(x1, x2) * compute_distance(x3, v)) / (compute_distance(x1, x3) * compute_distance(x2, v))
    return cr


def cross_ratio_4point_v(x1, x2, x3, x4, v, distance):
    # 已经知道x1,x2,求x3,x4
    cr1 = cross_ratio_v(x1, x2, x3, v)
    x1_x2 = distance
    x1_x3 = x1_x2 / cr1
    cr2 = cross_ratio_v(x3, x4, x1, v)
    x3_x4 = x1_x3 / cr2
    return cr1, cr2, x3_x4


def cross_ratio_3point(x, midpoint, endpoint):
    cr = compute_distance(midpoint, x) / compute_distance(midpoint, endpoint)
    return cr


def compute_fun(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    k = (y2 - y2) / (x2 - x1)
    b = y1 - k * x1
    return k, b


def get_line_tuple(line0):
    x1, y1 = line0[0]
    x2, y2 = line0[1]
    return (x1, y1), (x2, y2)

def my_drawline(img,point1,point2,color,thickness):
    img = cv2.line(img,  (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])) , color, thickness=thickness)
    return img

def my_circle(img,point,radius,color,width):
    img = cv2.circle(img,(int(point[0]),int(point[1])),radius,color,width)
    return img

def my_putText(img,label,point,FONT,s1,color,s2):
    img = cv2.putText(img, label, (int(point[0]),int(point[1])), \
                      FONT, s1, color, s2)
    return img

if __name__ == '__main__':
    img1 = cv2.imread('./img2.png')

    point_building = [[808, 63], [797, 172], [865, 139], [871, 16]]


    dian4_h = [[2457-1637, 1853-778], [2359-1637, 1833-778],\
               [2293-1637, 1856-778], [2394-1637, 1879-778]]
    #dian4_h = [[2452, 1853], [2362, 1835], [2293, 1856], [2396, 1880]]


    cross_line_h  = ((79,772),(1724,771))
    #cross_line_h = ((79, 752), (1724, 684))
    img1 = my_drawline(img1, cross_line_h[0], cross_line_h[1], (0, 255, 255), thickness=5)
    cross_point14 = get_crosspoint(cross_line_h,(dian4_h[0],dian4_h[3]))
    dian4_h[1] = get_point_inline(cross_point14,(dian4_h[2][0],dian4_h[2][1]),dian4_h[1][0])
    cross_point12 = get_crosspoint(cross_line_h, (dian4_h[0], dian4_h[1]))
    dian4_h[2] = get_point_inline(cross_point12,(dian4_h[3][0],dian4_h[3][1]),dian4_h[2][0])
        #get_point_inline((cross_point12, dian4_h[1]), (dian4_h[2], dian4_h[1]), dian4_h[2][0])

    src1 = np.float32(dian4_h).reshape(-1, 1, 2)# - np.array([1637,778])
    for i in range(4):
        img1 = my_circle(img1, (src1[i][0][0], src1[i][0][1]), 12, (0, 0, 255), -1)

    #crosspoint1_h,crosspoint2_h = get_crossline(dian4_h)
    crosspoint1_h = cross_point14
    crosspoint2_h = cross_point12
    crosspoint1_h_int = (int(crosspoint1_h[0]), int(crosspoint1_h[1]))
    crosspoint2_h_int = (int(crosspoint2_h[0]), int(crosspoint2_h[1]))
    img1 = cv2.circle(img1, crosspoint1_h_int, 20, (255, 255, 0), -1)
    img1 = cv2.circle(img1, crosspoint2_h_int, 20, (255, 255, 0), -1)
    print('crosspoint_h:',crosspoint1_h,crosspoint2_h)

    img1 = my_circle(img1, (int(crosspoint2_h[0]), int(crosspoint2_h[1])), 2, (255, 0, 0), -1)

    line_cross_h = (crosspoint1_h, crosspoint2_h)
    img1 = my_drawline(img1, crosspoint1_h, crosspoint2_h, (0, 255, 255), thickness=5)



    point_a = (1407, 1035)
    point_c = (649, 872)
    line_ac = (point_a,point_c)

    point_rect1 = get_tuple_int(dian4_h[0])
    point_rect2 = get_tuple_int(dian4_h[1])
    point_rect3 = get_tuple_int(dian4_h[2])
    point_rect4 = get_tuple_int(dian4_h[3])



    my_putText(img1, 'p1', point_rect1 ,cv2.FONT_HERSHEY_COMPLEX, 2,  (255, 255, 0), 2)
    my_putText(img1, 'p2', point_rect2, cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 0), 2)
    my_putText(img1, 'p3', point_rect3, cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 0), 2)
    my_putText(img1, 'p4', point_rect4, cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 0), 2)


    line_cross_h = (crosspoint1_h,crosspoint2_h)
    vp1 = get_crosspoint((point_rect1, point_rect2),(point_rect3, point_rect4))
    vp1_1 = get_crosspoint((point_rect1, point_rect2), line_cross_h)
    vp1_2 = get_crosspoint((point_rect3, point_rect4), line_cross_h)
    point_rect3 = get_crosspoint((vp1_1,point_rect4), (point_rect3, point_rect2))
    img1 = my_drawline(img1, vp1_1,point_rect1, (255, 255, 255), 3)
    img1 = my_drawline(img1, vp1_1, point_rect4, (255, 255, 255), 3)
    img1 = my_circle(img1, vp1_1, 7, (0, 0, 255), -1)

    line_cross_h = (crosspoint1_h,crosspoint2_h)
    vp2 = get_crosspoint((point_rect4, point_rect1),(point_rect3, point_rect2))
    vp2_1 = get_crosspoint((point_rect4, point_rect1), line_cross_h)
    point_rect1 = get_crosspoint((vp2_1, point_rect4), (point_rect2, point_rect1))
    vp2_2 = get_crosspoint((point_rect3, point_rect2), line_cross_h)
    img1 = my_drawline(img1, vp2_1,point_rect2, (255, 255, 255), 3)
    img1 = my_drawline(img1, vp2_1, point_rect1, (255, 255, 255), 3)
    img1 = my_circle(img1, vp2_1, 7, (0, 0, 255), -1)


    img1 = my_circle(img1, point_rect1, 5, (0, 0, 255), -1)
    img1 = my_circle(img1, point_rect2, 5, (0, 0, 255), -1)
    img1 = my_circle(img1, point_rect3, 5, (0, 0, 255), -1)
    img1 = my_circle(img1, point_rect4, 5, (0, 0, 255), -1)



    line23_rect = (point_rect2, point_rect4)
    vp23 = get_crosspoint(line23_rect, line_cross_h)
    line21_rect = (point_rect2, point_rect1)
    vp21 = get_crosspoint(line21_rect, line_cross_h)


    point_b1 = get_crosspoint(line_ac, line23_rect)
    point_b2 = get_crosspoint(line_ac, line21_rect)

    img1 = my_circle(img1, point_b1, 5, (255, 100, 255), -1)
    img1 = my_circle(img1, point_b1, 5, (255, 100, 255), -1)
    print('point_b1 point_b2:',point_b1,point_b2)

    #在line21上线段point_b1 point_rect2 point_rect4
    #cr = cross_ratio(A2, Ax, A1, V1)
    cr = cross_ratio_v(x1=point_rect2, x2=point_rect1, x3=point_b2, v = vp21)
    #cr = cross_ratio_3point(x=point_b1, midpoint=point_rect4, endpoint=point_rect2)
    dis = 0.9
    distance_b2_2 = dis / cr
    print('distance_b2_2=',distance_b2_2)
    #在line23上线段point_b2 point_rect3 point_rect4
    #cr = cross_ratio_3point(x=point_b2, midpoint=point_rect4, endpoint=point_rect3)
    cr = cross_ratio_v(x1=point_rect2, x2=point_rect3, x3=point_b1, v=vp23)
    cr2 = cross_ratio_v(x1=point_rect2, x2=point_rect3, x3=point_b2, v=vp23)
    print(cr,cr2,1/cr)
    dis = 0.9
    distance_b1_2 = dis / cr
    disrance_b1_b2 = np.sqrt(np.square(distance_b1_2)+np.square(distance_b2_2))

    dis = disrance_b1_b2
    print('disrance_b1_b2:',disrance_b1_b2)
    vp_ac = get_crosspoint(line_ac, line_cross_h)
    cr1,cr2,x3_x4 = cross_ratio_4point_v(x1 = point_b1, x2=point_b2, x3=point_a, x4=point_c, v = vp_ac ,distance =dis)
    print('cr1*cr2',cr1*cr2,x3_x4)
    disrance_ac = dis/cr1/cr2
    print('disrance_ac:',disrance_ac,cr1,cr2)




    dian4_v1 = point_building

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

    point_mid_img = (int(img1.shape[1] / 2), int(img1.shape[0] / 2))
    print('point_mid_img:', point_mid_img)
    img1 = cv2.circle(img1, (point_mid_img[0], point_mid_img[1]), 40, (255, 0, 255), -1)

    line_cross_h = (crosspoint1_h,crosspoint2_h)
    img1 = my_drawline(img1, crosspoint1_h,crosspoint2_h, (0, 0, 255), 1)
    point_mid_img_cross = get_pedal_point(line_cross_h, point_mid_img)
    img1 = cv2.circle(img1, (int(point_mid_img_cross[0]), int(point_mid_img_cross[1])), 40, (255, 0, 255), -1)
    line_main = (point_mid_img_cross,point_mid_img)
    line_cross_v1 = (crosspoint1_v1, crosspoint2_v1)
    point_camera_projection = get_crosspoint(line_main, line_cross_v1)
    #point_camera_projection = (1084.7444437105746, 2553.049772611333)
    print('point_camera_projection:',point_camera_projection)

    point_b = point_camera_projection

    line_ab = (point_a,point_b)

    my_putText(img1, 'p1', point_rect1 ,cv2.FONT_HERSHEY_COMPLEX, 1,  (255, 255, 0), 1)
    my_putText(img1, 'p2', point_rect2, cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 1)
    my_putText(img1, 'p3', point_rect3, cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 1)
    my_putText(img1, 'p4', point_rect4, cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 1)


    line_cross_h = (crosspoint1_h,crosspoint2_h)
    vp1 = get_crosspoint((point_rect1, point_rect2),(point_rect3, point_rect4))
    vp1_1 = get_crosspoint((point_rect1, point_rect2), line_cross_h)
    vp1_2 = get_crosspoint((point_rect3, point_rect4), line_cross_h)
    point_rect3 = get_crosspoint((vp1_1,point_rect4), (point_rect3, point_rect2))
    img1 = my_drawline(img1, vp1_1,point_rect1, (255, 255, 255), 3)
    img1 = my_drawline(img1, vp1_1, point_rect4, (255, 255, 255), 3)
    img1 = my_circle(img1, vp1_1, 7, (0, 0, 255), -1)

    line_cross_h = (crosspoint1_h,crosspoint2_h)
    vp2 = get_crosspoint((point_rect4, point_rect1),(point_rect3, point_rect2))
    vp2_1 = get_crosspoint((point_rect4, point_rect1), line_cross_h)
    point_rect1 = get_crosspoint((vp2_1, point_rect4), (point_rect2, point_rect1))
    vp2_2 = get_crosspoint((point_rect3, point_rect2), line_cross_h)
    img1 = my_drawline(img1, vp2_1,point_rect3, (255, 255, 255), 3)
    img1 = my_drawline(img1, vp2_1, point_rect4, (255, 255, 255), 3)
    img1 = my_circle(img1, vp2_1, 7, (0, 0, 255), -1)


    img1 = my_circle(img1, point_rect1, 5, (0, 0, 255), -1)
    img1 = my_circle(img1, point_rect2, 5, (0, 0, 255), -1)
    img1 = my_circle(img1, point_rect3, 5, (0, 0, 255), -1)
    img1 = my_circle(img1, point_rect4, 5, (0, 0, 255), -1)



    line23_rect = (point_rect2, point_rect4)
    vp23 = get_crosspoint(line23_rect, line_cross_h)
    line21_rect = (point_rect2, point_rect1)
    vp21 = get_crosspoint(line21_rect, line_cross_h)


    point_b1 = get_crosspoint(line_ab, line23_rect)
    point_b2 = get_crosspoint(line_ab, line21_rect)

    img1 = my_circle(img1, point_b1, 5, (255, 100, 255), -1)
    img1 = my_circle(img1, point_b1, 5, (255, 100, 255), -1)
    print('point_b1 point_b2:',point_b1,point_b2)

    #在line21上线段point_b1 point_rect2 point_rect4

    cr = cross_ratio_v(x1=point_rect2, x2=point_rect1, x3=point_b2, v = vp21)
    dis = 0.9
    distance_b2_2 = dis / cr
    print('distance_b2_2=',distance_b2_2)
    cr = cross_ratio_v(x1=point_rect2, x2=point_rect3, x3=point_b1, v=vp23)
    cr2 = cross_ratio_v(x1=point_rect2, x2=point_rect3, x3=point_b2, v=vp23)
    print(cr,cr2,1/cr)
    dis = 0.9
    distance_b1_2 = dis / cr
    disrance_b1_b2 = np.sqrt(np.square(distance_b1_2)+np.square(distance_b2_2))

    dis = disrance_b1_b2
    print('disrance_b1_b2:',disrance_b1_b2)
    vp_ab = get_crosspoint(line_ab, line_cross_h)
    cr1,cr2,x3_x4 = cross_ratio_4point_v(x1 = point_b1, x2=point_b2, x3=point_a, x4=point_b, v = vp_ab ,distance =dis)
    print('cr1*cr2',cr1*cr2,x3_x4)
    disrance_ab = dis/cr1/cr2
    print('disrance_ab:',disrance_ab,cr1,cr2)

    print('A到C的距离 (cm):', disrance_ac*100 )
    print('拍照者距B的距离 (cm):',disrance_ab*100)




    cv2.namedWindow('img1', 0)
    cv2.imshow("img1", img1)  # 显示
    cv2.imwrite("img2_result.png", img1)
    cv2.waitKey(0)  # 按下任意键退出
    cv2.destroyAllWindows()
