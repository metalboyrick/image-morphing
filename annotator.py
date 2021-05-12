import cv2

# program to annotate manually

pos_list = []


# Creating mouse callback function
def draw_circle(event, x, y, flags, param):
    if (event == cv2.EVENT_LBUTTONDBLCLK):
        cv2.circle(img, (x, y), 2, (255, 255, 0), -1)
        pos_list.append((x, y))
        print(f"{len(pos_list)}/68")


# Creating a black image, a window and bind the function to window
img = cv2.imread("img/target2.png", cv2.IMREAD_COLOR)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

while (1):
    cv2.imshow('image', img)
    if (cv2.waitKey(20) & 0xFF == 27) or len(pos_list) == 68:
        with open("annnotated_points.txt", "w+") as f:
            f.write(str(pos_list))
        break

cv2.destroyAllWindows()
