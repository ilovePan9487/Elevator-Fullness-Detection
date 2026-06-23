import cv2
import numpy as np


img = cv2.imread("IMG_6327.jpg")

if img is None:
    print("讀不到圖片，請確認檔名是否正確")
    exit()


scale = 0.45
show_img = cv2.resize(img, None, fx=scale, fy=scale)


roi_small = cv2.selectROI("Select box inside area", show_img, False)
cv2.destroyAllWindows()


x = int(roi_small[0] / scale)
y = int(roi_small[1] / scale)
w = int(roi_small[2] / scale)
h = int(roi_small[3] / scale)


roi = img[y:y+h, x:x+w]


blur = cv2.GaussianBlur(roi, (5, 5), 0)


bg_color = np.median(blur.reshape(-1, 3), axis=0)


diff = np.linalg.norm(blur.astype(np.float32) - bg_color.astype(np.float32), axis=2)
diff = np.clip(diff, 0, 255).astype(np.uint8)


ret, binary = cv2.threshold(diff, 55, 255, cv2.THRESH_BINARY)


kernel = np.ones((7, 7), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


contours, hierarchy = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)


object_area = 0
valid_contours = []

for contour in contours:
    area = cv2.contourArea(contour)

   
    if area > 1000:
        object_area += area
        valid_contours.append(contour)


box_area = w * h


occupancy_rate = object_area / box_area * 100


if occupancy_rate >= 80:
    status = "FULL"
else:
    status = "NOT FULL"


result = img.copy()


cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 3)


global_contours = []
for contour in valid_contours:
    contour_global = contour + np.array([[[x, y]]])
    global_contours.append(contour_global)


cv2.drawContours(result, global_contours, -1, (0, 255, 0), 3)


cv2.putText(result, f"Occupancy: {occupancy_rate:.2f}%", (50, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 4)

cv2.putText(result, f"Status: {status}", (50, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 4)


print(f"物品佔用率: {occupancy_rate:.2f}%")
print(f"判斷結果: {status}")


cv2.imshow("binary", binary)
cv2.imshow("result", result)


cv2.imwrite("result_contour.jpg", result)
cv2.imwrite("binary_result.jpg", binary)

cv2.waitKey(0)
cv2.destroyAllWindows()
