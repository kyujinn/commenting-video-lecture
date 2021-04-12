import numpy as np
import cv2
import time
import datetime
import os

path = "./data/" #캡쳐 파일 및 텍스트 파일 저장 경로

def createDir(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print("Error: Creating Directory -", dir)

def load_image(src):
    image = cv2.imread(path+ src, cv2.IMREAD_UNCHANGED)
    cv2.imshow("Image Display", image)
    cv2.waitKey(0)
    cv2.destoryAllWindows()

def load_video(src, dir):
    capture = cv2.VideoCapture(path + src)

    capture_count = 0 #캡쳐 횟수를 저장
    txt = open(dir + src[:-4] + ".txt", "w") #캡쳐 타임라인 기록용 txt 파일

    start = time.time()  # 영상 재생 시간 측정 시작

    while capture.isOpened():
        run, frame = capture.read()

        if not run:
            print("[프레임 수신 불가] - 종료합니다.")
            break

        img = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
        cv2.imshow('video', frame)

        key = cv2.waitKey(30) & 0xFF #키보드 입력을 무한 대기, 8비트 마스크 처리

        if key == ord('q'): #q 입력시 재생 종료
            break
        elif key == ord('c'): #c 입력시 캡쳐하여 입력받은 디렉토리에 png로 저장

            #텍스트 파일에 캡쳐 파일에 따른 기록 시간 저장
            sec = time.time() - start
            times = str(datetime.timedelta(seconds=sec)).split(".")
            times = times[0]
            txt.write(src + "_" + str(capture_count) + ".png - " + times + "\n")

            print("스크린 캡처 - " + times)
            cv2.imwrite(dir + src + "_" + str(capture_count) + ".png", frame)
            capture_count += 1

    capture.release()
    cv2.destroyAllWindows()
    
    txt.close() #캡쳐 타임라인 기록 종료

if __name__ == '__main__':
    #load_image("./data/hakyeon1.PNG")

    new_captured_dir = path + "Test/"
    createDir(new_captured_dir)
    load_video("video_datamining.mp4", new_captured_dir)
