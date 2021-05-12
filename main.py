from morphing import *

LION_LANDMARK = [(15, 164), (17, 201), (31, 240), (43, 273), (64, 310), (78, 343), (95, 387), (124, 414), (167, 417),
                 (217, 400), (238, 366), (239, 323), (248, 279), (271, 230), (276, 184), (278, 150), (255, 116),
                 (33, 117), (52, 95), (73, 88), (101, 90), (124, 108), (156, 104), (175, 82), (205, 84), (229, 89),
                 (246, 101), (140, 128), (143, 158), (147, 201), (151, 247), (98, 267), (122, 293), (154, 311),
                 (185, 293), (205, 260), (41, 158), (51, 143), (73, 139), (87, 155), (84, 171), (64, 171), (196, 153),
                 (193, 134), (213, 127), (239, 137), (229, 149), (210, 156), (116, 371), (127, 354), (143, 339),
                 (162, 337), (181, 340), (203, 352), (216, 366), (209, 383), (185, 377), (162, 378), (149, 381),
                 (132, 381), (131, 366), (147, 354), (162, 353), (182, 354), (195, 363), (185, 367), (163, 369),
                 (147, 368)]


def main():
    src_img_1 = cv2.imread("img/source1.png", cv2.IMREAD_COLOR)
    tgt_img_1 = cv2.imread("img/target1.png", cv2.IMREAD_COLOR)
    src_img_2 = cv2.imread("img/source2.png", cv2.IMREAD_COLOR)
    tgt_img_2 = cv2.imread("img/target2.png", cv2.IMREAD_COLOR)

    for i in range(0, 11):
        out_img_1 = morph(src_img_1, tgt_img_1, i / 10)
        out_img_2 = morph(src_img_2, tgt_img_2, i / 10, manual_landmark_tgt=LION_LANDMARK)
        cv2.imwrite(f"out1/{i}.jpg", out_img_1)
        cv2.imwrite(f"out2/{i}.jpg", out_img_2)
        print(f"processed {i} frames")




if __name__ == "__main__":
    main()
