import numpy as np
import cv2
from matplotlib import pyplot as plt

if __name__ == '__main__':
    
    # !!!!!! Set Video Path before run this code !!!!!!
    img1 = cv2.imread("./data/hakyeon1.PNG", cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread("./data/hakyeon2.png", cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x:x.distance)

    result = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches[:10], None, flags=2)
    plt.imshow(result),plt.show()
