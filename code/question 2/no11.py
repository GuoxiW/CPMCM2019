import os
import numpy as np
import cv2


known_length = 270

P1 = (1454, 866)
P2 = (1395, 918)
P3 = (1701, 942)
P4 = (1720, 884)

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

    # la = (A1, A2)
    # lt = (P2, P1)
    # crosspoint_t = get_crosspoint(la, lt)
    # print(crosspoint_t, V1)

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

    # # la = (A1, A2)
    # # lt = (P1, P2)
    # # crosspoint_t = get_crosspoint(la, lt)
    # # print(crosspoint_t, V2)

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

V1 = get_crosspoint((P2, P1), (P3, P4))
V2 = get_crosspoint((P2, P3), (P1, P4))
print('v1: {}, v2: {}'.format(V1, V2))

relative_length = predict_length(P1=P1, P2=P2, A1=(720, 668), A2=(585, 684), V1=V1, V2=V2)
print(relative_length)

distance2 = predict_distance1(P1=P1, P2=P2, A1=(1251, 599.5), A2=(585, 684), V1=V1, V2=V2, rl=relative_length)
print('distance of two cars: {} cm'.format(distance2))

distance4 = predict_distance2(P1=P1, P2=P4, A1=(245, 828), A2=(1353, 915), V1=V1, V2=V2, rl=relative_length)
print('distance of half road: {} cm'.format(distance4))

distance5 = predict_distance2(P1=P1, P2=P4, A1=(1352, 915), A2=(1818, 952), V1=V1, V2=V2, rl=relative_length)
print('distance of sidewalk: {} cm'.format(distance5))
