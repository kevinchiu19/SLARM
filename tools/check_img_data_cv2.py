import cv2
from glob import glob

data_root_path = 'xxx/SLARM_data/datasets/waymo/training'

scene_path = glob(data_root_path + '/*')
scene_num = len(scene_path)
scene_id = 0

img_path = glob(scene_path[scene_id] + '/images/*_0.jpg')
img_num = len(img_path)
img_id = 0


# Read image
def read_img(img_name):
    img = cv2.imread(img_name)
    if img is None:
        print("Unable to read image, please check if the path is correct")
        exit()
    return img


# Create window and display initial image
img = read_img(img_path[img_id])
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)  # Resizable window
cv2.imshow("Image", img)


while True:
    # Wait for key press (returns ASCII code, press ESC to exit)
    key = cv2.waitKey(1) & 0xFF  # 0xFF ensures correct key value on 64-bit systems
    # Key operation logic
    if key == 27:  # ESC key ASCII code is 27
        print("Program exited")
        break
    elif key == ord('c') or key == ord('C'):  # Press 'c' or 'C' to record image name
        print(img_path[img_id])
    elif key == ord('w') or key == ord('W'):  # Press 'w' or 'W' to switch image (forward)
        img_id += 1
        if img_id == img_num:
            img_id = 0
    elif key == ord('s') or key == ord('S'):  # Press 's' or 'S' to switch image (backward)
        if img_id == 0:
            img_id = img_num
        img_id -= 1
    elif key == ord('d') or key == ord('D'):  # Press 'd' or 'D' to switch scene (forward)
        scene_id += 1
        if scene_id == scene_num:
            scene_id = 0
        img_path = glob(scene_path[scene_id] + '/images/*_0.jpg')
        img_num = len(img_path)
        img_id = 0
    elif key == ord('a') or key == ord('A'):  # Press 'a' or 'A' to switch scene (backward)
        if scene_id == 0:
            scene_id = scene_num
        scene_id -= 1
        img_path = glob(scene_path[scene_id] + '/images/*_0.jpg')
        img_num = len(img_path)
        img_id = 0
    img = read_img(img_path[img_id])
    cv2.imshow("Image", img)


# Cleanup resources
cv2.destroyAllWindows()
