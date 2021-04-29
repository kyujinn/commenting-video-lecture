import cv2
import os
import sys
import time
import datetime
import pandas as pd
import numpy as np
import re
from pytesseract import *

from pdf2image import convert_from_path  # pdf2img
from gtts import gTTS

pdf2image_module_path = "data/Release-21.03.0/poppler-21.03.0/Library/bin/"

# 강의 동영상 내 전환시점 파악을 위한 라이브러리 호출
from scenedetect import VideoManager, SceneManager, StatsManager
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import save_images, write_scene_list_html

# 경로 설정
# !!!!! 경로 내에 한글 디렉토리 및 한글 파일이 있으면 제대로 동작하지 않음 유의 !!!!!
default_path = "Datamining/"
pdf_path = default_path + "lecture_doc.pdf"
video_path = default_path + "lecture_video.mp4"
capture_path = default_path + "capture/"
slide_path = default_path + "slide/"
txt_path = default_path + "txt/"
tts_path = default_path + "tts/"

# 최종 출력 파일
df = pd.DataFrame()
save_path = default_path + "transform_timeline_result.csv"

def txt2TTS(): #텍스트 파일을 기반으로 TTS 음성파일을 생성하는 함수

    print("\n[TTS 시작] TTS 변환을 시작합니다")
    txt_list = os.listdir(txt_path)
    txt_list = [txt_file for txt_file in txt_list if txt_file.endswith(".txt")] #jpg로 끝나는 것만 가져오기
    txt_list.sort()
    print(">>> 텍스트 파일 목록:", txt_list)

    # 디렉토리 유무 검사 및 디렉토리 생성
    try:
        if not os.path.exists(tts_path):  # 디렉토리 없을 시 생성
            os.makedirs(tts_path)
    except OSError:
        print('Error: Creating directory. ' + tts_path)  # 디렉토리 생성 오류

    for idx, txt_file in enumerate(txt_list):
        infile = txt_path + txt_file
        f = open(infile, 'r', encoding='UTF-8')
        sText = f.read()
        f.close()

        tts = gTTS(text=sText, lang='ko', slow=False)
        tts.save(tts_path + "tts_" + set_Filenum_of_Name(idx+1) + ".mp3")

        print(set_Filenum_of_Name(idx+1)+" MP3 file saved!")

    print("[TTS 종료] TTS 변환을 종료합니다\n")

# 의도한바와 같이 정렬될 수 있도록 파일번호 수정하여 반환하는 함수
def set_Filenum_of_Name(filenum):
    fileName = ""

    if (filenum < 10):  # 파일번호가 한자리일때
        fileName = "00" + str(filenum)
    elif (filenum >= 10 and filenum < 100):  # 파일번호가 두자리일때
        fileName = "0" + str(filenum)
    elif (filenum >= 100 and filenum < 1000):  # 파일번호가 세자리일때
        fileName = str(filenum)
    else:
        sys.exit(">>> PDF 파일이 너무 큽니다 - 999장 이상")

    return fileName


def pdf2jpg():  # pdf 파일을 기반으로 이미지(jpg)를 생성하는 함수

    print("\n[PDF2IMG 시작] PDF2JPG 이미지 변환을 시작합니다")

    pages = convert_from_path(pdf_path, poppler_path=pdf2image_module_path)
    print(">>> 인식된 강의자료 페이지 수:", len(pages))

    # 디렉토리 유무 검사 및 디렉토리 생성
    try:
        if not os.path.exists(slide_path):  # 디렉토리 없을 시 생성
            os.makedirs(slide_path)
    except OSError:
        print('Error: Creating directory. ' + txt_path)  # 디렉토리 생성 오류

    for i, page in enumerate(pages):
        page.save(slide_path + "slide_" + set_Filenum_of_Name(i + 1) + ".jpg", "JPEG")

    print("[PDF2IMG 종료] PDF2JPG 이미지 변환을 종료합니다\n")


def calOrb_CapNSlide():
    print("\n[ORB 계산 시작] 캡쳐 화면과 슬라이드 이미지의 유사도 계산을 시작합니다")

    # 장면전환 캡쳐 이미지 목록 로드
    capture_list = os.listdir(capture_path)  # .jpg 형식
    capture_list = [capture for capture in capture_list if capture.endswith(".jpg")]
    capture_list.sort()

    # PDF 슬라이드 변환 이미지 목록 로드
    slide_list = os.listdir(slide_path)  # .jpg 형식
    slide_list = [slide for slide in slide_list if slide.endswith(".jpg")]
    slide_list.sort()

    # ORB 계산
    tf_timeline_idx = []  # 장면전환 된 타임라인을 인덱싱하기 위한 이미지 인덱스 번호 저장
    for slide in slide_list:
        print(">>>", slide_path + slide)
        slide_img = cv2.imread(slide_path + slide, None)

        max_match_point = -1
        max_match_point_idx = -1
        for idx, capture in enumerate(capture_list):

            capture_img = cv2.imread(capture_path + capture, None)

            orb = cv2.ORB_create()

            kp1, des1 = orb.detectAndCompute(slide_img, None)
            kp2, des2 = orb.detectAndCompute(capture_img, None)

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            count = len(matches)
            print(">>> >>>", capture_path + capture, ":", count, "vs", max_match_point)
            if count > max_match_point:
                max_match_point = count
                max_match_point_idx = idx

        tf_timeline_idx.append(max_match_point_idx)

    # print(tf_timeline_idx)
    print("[ORB 계산 종료] 캡쳐 화면과 슬라이드 이미지의 유사도 계산을 종료합니다\n")

    return tf_timeline_idx


def ocr_img2txt():
    print("\n[OCR 변환 시작] 슬라이드 이미지를 텍스트로 변환을 시작합니다")

    slide_list = os.listdir(slide_path)
    slide_list = [slide for slide in slide_list if slide.endswith(".jpg")]  # jpg로 끝나는 것만 가져오기
    slide_list.sort()
    print(">>> 슬라이드 이미지 목록:", slide_list)

    df['slide'] = slide_list

    # 디렉토리 유무 검사 및 디렉토리 생성
    try:
        if not os.path.exists(txt_path):  # 디렉토리 없을 시 생성
            os.makedirs(txt_path)
    except OSError:
        print('Error: Creating directory. ' + txt_path)  # 디렉토리 생성 오류

    language = "kor+eng"  # ocr 언어 설정
    for idx, slide in enumerate(slide_list):
        img_file = slide_path + slide  # 번역할 슬라이드 이미지
        print(">>> >>>", slide, "슬라이드 OCR 변환")
        txtFile = open(txt_path + img_file[-7:-4] + ".txt", "w", -1, "utf-8")  # 번역한 내용을 저장할 텍스트 파일

        txtFile.write(img_file[-5:-4] + "번 슬라이드 해설 시작" + "\n" + "\n")
        result = pytesseract.image_to_string(img_file, lang=language)  # 언어 옵션(kor+eng) 지정 및 psm 옵션 지정 안함
        txtFile.write(result + "\n")
        txtFile.write(img_file[-5:-4] + "번 슬라이드 해설 끝")

        txtFile.close()

        #텍스트 변환 필터링
        txtfilter_open = open(txt_path + img_file[-7:-4] + ".txt", "r", -1, "utf-8")
        pp = re.compile("[ㄱ-ㅣ가-힣A-Za-z0-9\s]")
        txtfilter1 = txtfilter_open.read()
        txtfilter1 = pp.findall(txtfilter1)
        txtfilter1 = ''.join(txtfilter1)
        txtfilter2 = re.sub('\\n+','\n',txtfilter1)
        textfilter = re.sub('','',txtfilter2)
        txtfilter_out = open(txt_path + img_file[-7:-4] + ".txt", "w", -1, "utf-8")
        txtfilter_out.write(textfilter)

    print("[OCR 변환 종료] 슬라이드 이미지를 텍스트로 변환을 종료합니다\n")


def capture_video():
    print("\n[전환장면 캡처 시작] 영상 내 전환 시점을 기준으로 이미지 추출을 시작합니다")

    video_manager = VideoManager([video_path])
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)

    # 가장 예민하게 잡아내도록 1~100 중 1로 설정
    scene_manager.add_detector(ContentDetector(threshold=1))

    video_manager.set_downscale_factor()

    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    scene_list = scene_manager.get_scene_list()
    print(">>>", f'{len(scene_list)} scenes detected!')  # 전환 인식이 된 장면의 수

    save_images(
        scene_list,
        video_manager,
        num_images=1,
        image_name_template='$SCENE_NUMBER',
        output_dir=capture_path)

    write_scene_list_html(default_path + 'SceneDetectResult.html', scene_list)

    captured_timeline_list = []  # 전환된 시점을 저장할 리스트 함수
    for scene in scene_list:
        start, end = scene

        # 전환 시점 저장
        # print(f'{start.get_seconds()} - {end.get_seconds()}')
        captured_timeline_list.append(start.get_seconds())

    print("[전환장면 캡처 종료] 영상 내 전환 시점을 기준으로 이미지 추출을 종료합니다\n")
    return captured_timeline_list


if __name__ == '__main__':
    time_list = []  # pdf 2 image, 장면 추출 시간, OCR 시간, ORB 유사도 시간, 총 시간
    total_start = time.time()

    # PDF to Image
    tmp_start = time.time()
    pdf2jpg()
    tmp_sec = time.time() - tmp_start
    tmp_times = str(datetime.timedelta(seconds=tmp_sec)).split(".")
    time_list.append(tmp_times[0])

    # 장면전환 추출
    tmp_start = time.time()
    captured_timeline_list = capture_video()
    tmp_sec = time.time() - tmp_start
    tmp_times = str(datetime.timedelta(seconds=tmp_sec)).split(".")
    time_list.append(tmp_times[0])

    # OCR 인식 및 저장
    tmp_start = time.time()
    ocr_img2txt()
    tmp_sec = time.time() - tmp_start
    tmp_times = str(datetime.timedelta(seconds=tmp_sec)).split(".")
    time_list.append(tmp_times[0])

    # ORB 유사도 계산
    tmp_start = time.time()
    tf_timeline_idx = calOrb_CapNSlide()
    tmp_sec = time.time() - tmp_start
    tmp_times = str(datetime.timedelta(seconds=tmp_sec)).split(".")
    time_list.append(tmp_times[0])

    tf_timeline_list = []
    for i, idx_val in enumerate(tf_timeline_idx):
        tf_timeline_list.append(captured_timeline_list[idx_val])
        print("[" + str(i + 1) + "번째 슬라이드 등장시간]", int(captured_timeline_list[idx_val] / 60), "분",
              round(captured_timeline_list[idx_val] % 60), "초")

    df['time'] = tf_timeline_list

    txt2TTS()
    time_list.append(tmp_times[0])

    total_sec = time.time() - total_start
    total_times = str(datetime.timedelta(seconds=total_sec)).split(".")
    time_list.append(total_times[0])

    # 장면 추출 시간, OCR 시간, ORB 유사도 시간, 총 시간
    print("\n■ PDF2JPG 추출 시간:", time_list[0])
    print("■ 장면 추출 시간:", time_list[1])
    print("■ OCR 시간:", time_list[2])
    print("■ ORB 시간:", time_list[3])
    print("■ TTS 시간:", time_list[4])
    print("■□■ 총 소요 시간:", time_list[5])

    df.to_csv(save_path, mode='w')
