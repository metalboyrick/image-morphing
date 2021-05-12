import cv2
import numpy as np
import scipy.linalg as linalg


# find the affine transform between two triangles
def get_affine_transform(src_tri, dst_tri):
    # build b
    b = np.array([dst_tri[0][0], dst_tri[0][1],
                  dst_tri[1][0], dst_tri[1][1],
                  dst_tri[2][0], dst_tri[2][1]])

    # build A
    list_A = []
    for i in range(3):
        list_A.append([src_tri[i][0], src_tri[i][1], 1, 0, 0, 0])
        list_A.append([0, 0, 0, src_tri[i][0], src_tri[i][1], 1])

    A = np.array(list_A)

    x = linalg.solve(A, b)

    return np.array([[x[0], x[1], x[2]], [x[3], x[4], x[5]]])


# warps a triangular section into the destination
def warp_affine_tri(src_img, src_tri, dst_tri, dst):
    bound_box_src = cv2.boundingRect(src_tri)
    bound_box_dst = cv2.boundingRect(dst_tri)

    cropped_tri_src = []
    cropped_tri_dst = []
    for i in range(3):
        cropped_tri_src.append((src_tri[i][0] - bound_box_src[0], src_tri[i][1] - bound_box_src[1]))
        cropped_tri_dst.append((dst_tri[i][0] - bound_box_dst[0], dst_tri[i][1] - bound_box_dst[1]))

    cropped_src = src_img[bound_box_src[1]:bound_box_src[1] + bound_box_src[3],
                  bound_box_src[0]:bound_box_src[0] + bound_box_src[2]]

    affine_mat = get_affine_transform(cropped_tri_src, cropped_tri_dst)
    cropped_dst = cv2.warpAffine(cropped_src, affine_mat, (bound_box_dst[2], bound_box_dst[3]), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)

    # Get mask by filling triangle
    mask = np.zeros((bound_box_dst[3], bound_box_dst[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(cropped_tri_dst), (1.0, 1.0, 1.0), 16, 0)

    cropped_dst = cropped_dst * mask

    dst[bound_box_dst[1]:bound_box_dst[1]+bound_box_dst[3], bound_box_dst[0]:bound_box_dst[0]+bound_box_dst[2]] = dst[bound_box_dst[1]:bound_box_dst[1]+bound_box_dst[3], bound_box_dst[0]:bound_box_dst[0]+bound_box_dst[2]] * ( (1.0, 1.0, 1.0) - mask )
    dst[bound_box_dst[1]:bound_box_dst[1]+bound_box_dst[3], bound_box_dst[0]:bound_box_dst[0]+bound_box_dst[2]] = dst[bound_box_dst[1]:bound_box_dst[1]+bound_box_dst[3], bound_box_dst[0]:bound_box_dst[0]+bound_box_dst[2]] + cropped_dst



