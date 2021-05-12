import cv2
import dlib
import numpy as np
from warping import *

# training model from https://github.com/davisking/dlib-models/

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def cross_dissolve(src_img, tgt_img, alpha):
    res_img = np.zeros((max(src_img.shape[0], tgt_img.shape[0]), max(src_img.shape[1], tgt_img.shape[1]), 3), np.uint8)
    for i in range(0, res_img.shape[0]):
        for j in range(0, res_img.shape[1]):
            for c in range(0, 3):
                res_img[i, j, c] = (1 - alpha) * src_img[i, j, c] + alpha * tgt_img[i, j, c]

    return res_img


def get_facial_landmarks(img):
    faces = detector(img)
    img_display = img.copy()
    facial_landmarks = predictor(img, faces[0])
    landmarks = []
    for i in range(68):
        x, y = facial_landmarks.part(i).x, facial_landmarks.part(i).y
        cv2.circle(img_display, (x, y), 2, (0, 0, 255), cv2.FILLED)
        landmarks.append((x, y))

    return landmarks


def get_triangulation(img, landmarks):
    img_rect = (0, 0, img.shape[1], img.shape[0])
    subdiv = cv2.Subdiv2D(img_rect)

    # insert corner points
    corner_points = [
        (0, 0),
        (img.shape[1] // 2, 0),
        (img.shape[1] - 1, 0),
        (img.shape[1] - 1, img.shape[0] // 2),
        (img.shape[1] - 1, img.shape[0] - 1),
        (img.shape[1] // 2, img.shape[0] - 1),
        (0, img.shape[0] - 1),
        (0, img.shape[0] // 2),
    ]

    landmarks.extend(corner_points)

    for point in landmarks:
        subdiv.insert(point)

    triangles_list = subdiv.getTriangleList()
    triangles = []
    index_triangles = []
    for t_pt in triangles_list:
        tri_1 = (int(t_pt[0]), int(t_pt[1]))
        tri_2 = (int(t_pt[2]), int(t_pt[3]))
        tri_3 = (int(t_pt[4]), int(t_pt[5]))
        triangle_set = (tri_1, tri_2, tri_3)
        index_triangles.append((landmarks.index(tri_1),
                                landmarks.index(tri_2),
                                landmarks.index(tri_3)))
        triangles.append(triangle_set)

    img_display = img.copy()

    # draw lines to show triangulation
    for tri in triangles:
        cv2.line(img_display, tri[0], tri[1], (0, 255, 0), 1)
        cv2.line(img_display, tri[1], tri[2], (0, 255, 0), 1)
        cv2.line(img_display, tri[2], tri[0], (0, 255, 0), 1)

    # cv2.imshow("Triangled source", img_display)
    # cv2.waitKey()

    return triangles, index_triangles


def map_triangulation(img, index_triangles, landmarks):
    # insert corner points
    corner_points = [
        (0, 0),
        (img.shape[1] // 2, 0),
        (img.shape[1] - 1, 0),
        (img.shape[1] - 1, img.shape[0] // 2),
        (img.shape[1] - 1, img.shape[0] - 1),
        (img.shape[1] // 2, img.shape[0] - 1),
        (0, img.shape[0] - 1),
        (0, img.shape[0] // 2),
    ]

    landmarks.extend(corner_points)

    triangles = []
    for triangle in index_triangles:
        triangles.append((landmarks[triangle[0]], landmarks[triangle[1]], landmarks[triangle[2]]))

    img_display = img.copy()

    # draw lines to show triangulation
    for tri in triangles:
        cv2.line(img_display, tri[0], tri[1], (0, 255, 0), 1)
        cv2.line(img_display, tri[1], tri[2], (0, 255, 0), 1)
        cv2.line(img_display, tri[2], tri[0], (0, 255, 0), 1)

    # cv2.imshow("Mapped triangles", img_display)
    # cv2.waitKey()

    return triangles


# calculate intermediate triangles
def inter_tri(tri_src, tri_tgt, alpha):
    n = len(tri_src)
    tri_inter = []
    for i in range(n):
        tri_tuple = [[0,0] for _ in range(3)]
        for vertice in range(3):
            tri_tuple[vertice][0] = (1 - alpha) * tri_src[i][vertice][0] + alpha * tri_tgt[i][vertice][0]
            tri_tuple[vertice][1] = (1 - alpha) * tri_src[i][vertice][1] + alpha * tri_tgt[i][vertice][1]
        tri_inter.append(tuple(tri_tuple))

    return tri_inter


def morph(src_img, tgt_img, alpha, manual_landmark_src=None, manual_landmark_tgt=None):

    # get manual landmark markings
    if manual_landmark_tgt:
        landmark_tgt = manual_landmark_tgt
    else:
        landmark_tgt = get_facial_landmarks(tgt_img)

    if manual_landmark_src:
        landmark_src = manual_landmark_src
    else:
        landmark_src = get_facial_landmarks(src_img)

    # get triangulation of the source image using delaunay
    tri_src, tri_src_index = get_triangulation(src_img, landmark_src)

    # map the triangulation of the source image to the target image
    tri_tgt = map_triangulation(tgt_img, tri_src_index, landmark_tgt)

    # get intermediate triangle
    tri_inter = inter_tri(tri_src, tri_tgt, alpha)

    # prepare the output image
    inter_img_src = np.zeros((max(src_img.shape[0], tgt_img.shape[0]),max(src_img.shape[1], tgt_img.shape[1]),3), dtype=src_img.dtype)
    inter_img_tgt = inter_img_src.copy()

    # convert all data to array
    tri_src = np.array(tri_src, np.float32)
    tri_tgt = np.array(tri_tgt, np.float32)
    tri_inter = np.array(tri_inter, np.float32)

    # warp the triangles
    tri_len = len(tri_src)
    for i in range(tri_len):
        warp_affine_tri(src_img, tri_src[i], tri_inter[i], inter_img_src)
        warp_affine_tri(tgt_img, tri_tgt[i], tri_inter[i], inter_img_tgt)



    # cv2.imshow("Warped Image Source", inter_img_src)
    # cv2.waitKey()
    #
    # cv2.imshow("Warped Image Target", inter_img_tgt)
    # cv2.waitKey()

    res_img = cross_dissolve(inter_img_src, inter_img_tgt, alpha)

    # cv2.imshow("Morphed", res_img)
    # cv2.waitKey()

    return res_img