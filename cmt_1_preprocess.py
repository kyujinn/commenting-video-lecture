"""
* 코드 파일 이름: cmt_1_preprocess.py
* 코드 작성자: 박동연, 강소정, 김유진
* 코드 설명: 동영상 강의 해설 파일을 생성하기 위한 전처리 파일 생성
* 코드 최종 수정일: 2021/06/27 (박동연)
* 문의 메일: yeon0729@sookmyung.ac.kr
"""

# 패키지 및 라이브러리 호출
import imagehash
import jellyfish
from PIL import Image
import sys
import time
import datetime
import pandas as pd
import re
import pdfplumber

import shutil
import moviepy.editor as mp

from pdf2image import convert_from_path  # pdf2img
from gtts import gTTS
from pydub import AudioSegment

# 이미지 추출 import
import fitz  # PyMuPDF
import io
import os

# 이미지 캡션 import
import requests

# 번역 import
from googletrans import Translator

pdf2image_module_path = "data/Release-21.03.0/poppler-21.03.0/Library/bin/"

# 강의 동영상 내 전환시점 파악을 위한 라이브러리 호출
from scenedetect import VideoManager, SceneManager, StatsManager
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import save_images, write_scene_list_html

# pdf to text를 위한 라이브러리 호출
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

# 경로 설정 (경로 내에 한글 디렉토리 및 한글 파일이 있으면 제대로 동작하지 않음 유의 !!!!!)
default_path = "UIUX/"
pdf_path = default_path + "lecture_doc.pdf"
video_path = default_path + "lecture_video.mp4"
audio_path = default_path + "lecture_audio.mp3"
capture_path = default_path + "capture/"
slide_path = default_path + "slide/"
txt_path = default_path + "txt/"
tts_path = default_path + "tts/"

mix_path = default_path + "mix/"
lec_path = default_path + "lec/"
img_path = default_path + "img/"

# 최종 출력 파일
df = pd.DataFrame()
save_path = default_path + "transform_timeline_result.csv"

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
        sys.exit(">>> 파일이 너무 큽니다 - 999장 이상")

    return fileName

#pdf 파일을 jpg 파일로 변환하는 함수
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

# 동영상 내 화면 전환이 발생하는 시점을 기준으로 동영상 캡처(추출)을 하는 함수
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

# pdf 파일에 있는 텍스트를 슬라이드 별로 뽑아내는 함수
def pdf2txt():
    print("\n[PDF to TXT 변환 시작] 슬라이드 이미지를 텍스트로 변환을 시작합니다")

    # 디렉토리 유무 검사 및 디렉토리 생성
    try:
        if not os.path.exists(txt_path):  # 디렉토리 없을 시 생성
            os.makedirs(txt_path)
    except OSError:
        print('Error: Creating directory. ' + txt_path)  # 디렉토리 생성 오류

    Pdf = pdfplumber.open(pdf_path)

    for page_idx, page in enumerate(Pdf.pages):
        txtFile = open(txt_path + set_Filenum_of_Name(page_idx + 1) + ".txt", "w", -1, "utf-8")  # 번역한 내용을 저장할 텍스트 파일

        txtFile.write(str(page_idx + 1) + "번 슬라이드 해설 시작" + "\n" + "\n")
        result = page.extract_text()

        imgcaption = imgExtract(page_idx)

        if (imgcaption == "이미지 없음"):
            print("이미지 없음")
            txtFile.write(result + "\n")
        else:
            imgcaption = " ".join(imgcaption)
            txtFile.write(result + imgcaption + "\n")

        txtFile.write(str(page_idx + 1) + "번 슬라이드 해설 끝")

        txtFile.close()

        # 텍스트 변환 필터링
        txtfilter_open = open(txt_path + set_Filenum_of_Name(page_idx + 1) + ".txt", "r", -1, "utf-8")
        pp = re.compile("[ㄱ-ㅣ가-힣A-Za-z0-9-+=.()~\s]")
        txtfilter1 = txtfilter_open.read()
        txtfilter1 = pp.findall(txtfilter1)
        txtfilter1 = ''.join(txtfilter1)
        txtfilter2 = re.sub('\\n+', '\n', txtfilter1)
        textfilter = re.sub('', '', txtfilter2)
        txtfilter_out = open(txt_path + set_Filenum_of_Name(page_idx + 1) + ".txt", "w", -1, "utf-8")
        txtfilter_out.write(textfilter)

        print(">>>", page_idx + 1, "번째 PDF 슬라이드 텍스트 변환 완료")

    """for page_idx, page_layout in enumerate(extract_pages(pdf_path)):

        txtFile = open(txt_path + set_Filenum_of_Name(page_idx + 1) + ".txt", "w", -1, "utf-8")  # 번역한 내용을 저장할 텍스트 파일
        txtFile.write(str(page_idx + 1) + "번 슬라이드 해설 시작" + "\n" + "\n")

        result = ""
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                result += element.get_text() + "\n"

        imgcaption = imgExtract(page_idx)

        if(imgcaption=="이미지 없음"):
            print("이미지 없음")
            txtFile.write(result + "\n")
        else:
            imgcaption = " ".join(imgcaption)
            txtFile.write(result + imgcaption + "\n")

        txtFile.write(str(page_idx + 1) + "번 슬라이드 해설 끝")

        txtFile.close()

        # 텍스트 변환 필터링
        txtfilter_open = open(txt_path + set_Filenum_of_Name(page_idx + 1) + ".txt", "r", -1, "utf-8")
        pp = re.compile("[ㄱ-ㅣ가-힣A-Za-z0-9-+=.()~\s]")
        txtfilter1 = txtfilter_open.read()
        txtfilter1 = pp.findall(txtfilter1)
        txtfilter1 = ''.join(txtfilter1)
        txtfilter2 = re.sub('\\n+', '\n', txtfilter1)
        textfilter = re.sub('', '', txtfilter2)
        txtfilter_out = open(txt_path + set_Filenum_of_Name(page_idx + 1) + ".txt", "w", -1, "utf-8")
        txtfilter_out.write(textfilter)

        print(">>>", page_idx + 1, "번째 PDF 슬라이드 텍스트 변환 완료")"""

    Pdf.close()
    print("[PDF to TXT 변환 종료] 슬라이드 이미지를 텍스트로 변환을 종료합니다\n")

# pdf 파일에 이미지가 있는지 판단하는 함수
def imgExtract(page_index):  # 이미지 추출 함수
    # open the file
    pdf_file = fitz.open(pdf_path)

    # 디렉토리 유무 검사 및 디렉토리 생성
    try:
        if not os.path.exists(img_path):  # 디렉토리 없을 시 생성
            os.makedirs(img_path)
    except OSError:
        print('Error: Creating directory. ' + img_path)  # 디렉토리 생성 오류

    # get the page itself
    page = pdf_file[page_index]
    image_list = page.getImageList()
    # printing number of images found in this page
    if image_list:
        print(f"[+] Found a total of {len(image_list)} images in page {page_index}")

        caption_list_eng = []
        caption_list_kor = []

        for image_index, img in enumerate(page.getImageList(), start=1):
            # get the XREF of the image
            xref = img[0]
            # extract the image bytes
            base_image = pdf_file.extractImage(xref)
            image_bytes = base_image["image"]
            # get the image extension
            image_ext = base_image["ext"]
            # load it to PIL
            image = Image.open(io.BytesIO(image_bytes))

            if (image.size[0]<=50 or image.size[1]<=50 or image.size[0] >= 10000 or image.size[1] >= 10000):

                if image.size[0] < image.size[1]:
                    new_width  = 60
                    new_height = int(new_width * image.size[1] / image.size[0])

                else:
                    new_height = 60
                    new_width  = int(new_height * image.size[0] / image.size[1])

                resize_image = image.resize((new_width, new_height), Image.ANTIALIAS)
                imgfileName = "slide_" + set_Filenum_of_Name(page_index+1) +"img_"+ set_Filenum_of_Name(image_index)+ ".jpg"
                resize_image.save(img_path+imgfileName)

            else :
                imgfileName = "slide_" + set_Filenum_of_Name(page_index+1) +"img_"+ set_Filenum_of_Name(image_index)+ ".jpg"
                image.save(open(img_path+imgfileName, "wb"))

            image_caption = imgCaption(imgfileName)      # 이미지 캡션 함수 호출
            caption_list_eng.append(image_caption)
            image_caption = str(image_index) + "번째 이미지 " + eng2Kor(image_caption) + "\n"
            caption_list_kor.append(image_caption)

        print("이미지캡션 영어:", caption_list_eng)
        print("이미지캡션 한국어:", caption_list_kor)

    else:
        print("[!] No images found on page", page_index)
        caption_list_kor = "이미지 없음"

    return caption_list_kor

# 이미지의 캡션 자막(영어)를 생성하는 함수
def imgCaption(imgfileName):  # 이미지 캡션 함수

    subscription_key = '1f05b01e39814389bff7f16810cc0522'
    endpoint = 'https://commentor.cognitiveservices.azure.com/'

    analyze_url = endpoint + "vision/v3.1/analyze"
    image_data = open(img_path + imgfileName, "rb").read()
    headers = {'Ocp-Apim-Subscription-Key': subscription_key,
               'Content-Type': 'application/octet-stream'}
    params = {'visualFeatures': 'Categories,Description,Color'}
    response = requests.post(
        analyze_url, headers=headers, params=params, data=image_data)
    response.raise_for_status()

    analysis = response.json()

    image_caption = analysis["description"]["captions"][0]["text"].capitalize()

    return image_caption

# 영어 문자열을 한글 문자열로 번역하는 함수
def eng2Kor(image_caption):
    translator = Translator()
    trans1 = translator.translate(image_caption, src='en', dest='ko')
    return trans1.text

# 두 문자열 간 자카드 유사도 계산
def JaccardSimilarity(input_hash1, input_hash2):
    list_inp1 = list(input_hash1)
    list_inp2 = list(input_hash2)
    hash_union = set(list_inp1).union(set(list_inp2))
    hash_intersection = set(list_inp1).intersection(set(list_inp2))
    # print(hash_union)
    # print(hash_intersection)
    return len(hash_intersection) / len(hash_union)

# Perceptive Hash 변환을 통한 jaro 유사도 계산
def calImgPerHashSim(slide, capture):
    slide = slide_path + slide
    capture = capture_path + capture

    # 해시값 변환 (Perceptive Hash)
    # Perceptive Hash는 낮은 주파수 영역을 추출하여 유의미한 값으로 이미지 압축
    slide_hash = imagehash.phash(Image.open(slide))
    capture_hash = imagehash.phash(Image.open(capture))

    # 자카드 유사도 계산
    img_hash_distance = JaccardSimilarity(str(slide_hash), str(capture_hash))

    # print(img_hash_distance)
    return img_hash_distance

# Difference Hash 변환을 통한 유사도 계산
def calImgDifHashSim(slide, capture):
    slide = slide_path + slide
    capture = capture_path + capture

    # 해시값 변환 (Difference Hash)
    # Difference Hash는 정해진 사이즈로 압축한 이미지의 인전한 픽셀 값의 크기 비교를 평가
    slide_hash = imagehash.dhash(Image.open(slide))
    capture_hash = imagehash.dhash(Image.open(capture))

    # print(slide_hash)
    # print(capture_hash)

    # 해밍 유사도 계산
    # img_hash_distance = distance.jaccard(slide_hash, capture_hash)
    img_hash_distance = jellyfish.jaro_distance(str(slide_hash), str(capture_hash))

    return img_hash_distance

# 동영상 캡처 이미지와 원본 슬라이드 이미지 간 유사도를 계산하는 함수
def calSim_CapNSlide():
    print("\n[이미지 유사도 계산 시작] 캡쳐 화면과 슬라이드 이미지의 유사도 계산을 시작합니다")

    slideList = os.listdir(slide_path)
    slideList = [slide_file for slide_file in slideList if slide_file.endswith(".jpg")]  # jpg로 끝나는 것만 가져오기
    slideList.sort()
    print(">>> 슬라이드 파일 목록:", slideList)

    captureList = os.listdir(capture_path)
    captureList = [capture_file for capture_file in captureList if capture_file.endswith(".jpg")]  # jpg로 끝나는 것만 가져오기
    captureList.sort()
    print(">>> 캡쳐 파일 목록:", captureList)

    final_list = []
    for slide in slideList:
        first_candidate_list = []
        second_candidate_list = []

        # 강의 슬라이드와 동영상 캡쳐본 간 해시값 계산
        for capture in captureList:
            img_hash_distance = calImgPerHashSim(slide, capture)
            first_candidate_list.append(img_hash_distance)
            print(">>> >>>", slide, "vs", capture, ":", img_hash_distance)

        first_max_value = max(first_candidate_list)

        tmp_list = []
        for idx, v in enumerate(first_candidate_list):
            if v == first_max_value:
                # tmp_list.append(captureList[idx])
                tmp_list.append(idx)

        print(slide, tmp_list)

        # 중복되는 최대 유사도 값이 2개 이상일 때 해싱 한 번 더 계산
        print(">>> 최대 유사도 값을 가지는 요소 개수:", len(tmp_list))
        if len(tmp_list) >= 2:
            for idx in tmp_list:
                img_hash_distance = calImgDifHashSim(slide, captureList[idx])
                second_candidate_list.append(img_hash_distance)
                print(">>> >>> >>>", slide, "vs", idx, ":", img_hash_distance)

            max_idx = second_candidate_list.index(max(second_candidate_list))  # 최대 유사도 값 중 첫 번재 등장 인덱스
            final_list.append(tmp_list[max_idx])  # 해당 인덱스의 이미지 파일명 저장

        else:
            final_list.append(tmp_list[0])

    print(final_list)
    print("[이미지 유사도 계산 종료] 캡쳐 화면과 슬라이드 이미지의 유사도 계산을 종료합니다\n")

    return final_list

# 텍스트 파일을 기반으로 TTS 음성파일을 생성하는 함수
def txt2TTS():

    print("\n[TTS 시작] TTS 변환을 시작합니다")
    txt_list = os.listdir(txt_path)
    txt_list = [txt_file for txt_file in txt_list if txt_file.endswith(".txt")]  # jpg로 끝나는 것만 가져오기
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
        tts.save(tts_path + "tts_" + set_Filenum_of_Name(idx + 1) + ".mp3")

        print(set_Filenum_of_Name(idx + 1) + " MP3 file saved!")

    print("[TTS 종료] TTS 변환을 종료합니다\n")

# 동영상 내 화면 전환이 발생하는 시점을 기준으로 원본 오디오 파일을 자르는 함수
def cutLectureMp3():
    print("\n[lec 생성 시작] mp3 파일 변환 및 mp3 파일 CUT을 시작합니다")

    print(">>> mp4 영상 → mp3 오디오 변환 시작")
    # 원본 강의 mp4영상을 mp3 형식으로 변환
    clip = mp.VideoFileClip(video_path)  # 동영상 불러오기
    clip.audio.write_audiofile("lecture_audio.mp3")  # mp3 파일로 변환
    shutil.move("lecture_audio.mp3", audio_path)  # mp3 파일을 원하는 디렉토리로 이동
    print(">>> mp4 영상 → mp3 오디오 변환 완료")

    print(">>> mp3 오디오 CUT 시작")
    # 원본 강의 mp3 파일을 전환 시간에 맞추어 cut
    audio = AudioSegment.from_mp3(audio_path)
    time_csv = pd.read_csv(save_path)

    # 디렉토리 유무 검사 및 디렉토리 생성
    try:
        if not os.path.exists(lec_path):  # 디렉토리 없을 시 생성
            os.makedirs(lec_path)
    except OSError:
        print('Error: Creating directory. ' + lec_path)  # 디렉토리 생성 오류

    for i in range(len(time_csv["time"])):

        fileName = "lec_" + set_Filenum_of_Name(i + 1) + ".mp3"
        fileName = lec_path + fileName

        if i == (len(time_csv["time"]) - 1):  # 마지막 클립
            result = audio[int(time_csv["time"][i]) * 1000:]
            result.export(fileName, format='mp3')
        else:  # 처음, 중간 클립
            result = audio[int(time_csv["time"][i]) * 1000: int(time_csv["time"][i + 1]) * 1000]
            result.export(fileName, format='mp3')

        print(">>> >>>", i + 1, "번째 클립 mp3 파일 생성 완료")

    print("\n[lec 생성 종료] mp3 파일 변환 및 mp3 파일 CUT을 종료합니다")


# 메인함수
if __name__ == '__main__':
    time_list = []  # pdf 2 image, 장면 추출 시간, OCR 시간, ORB 유사도 시간
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

    # PDF to TXT 저장
    tmp_start = time.time()
    pdf2txt()
    tmp_sec = time.time() - tmp_start
    tmp_times = str(datetime.timedelta(seconds=tmp_sec)).split(".")
    time_list.append(tmp_times[0])

    # 슬라이드와 캡처본 간 이미지 유사도 계산
    tmp_start = time.time()
    tf_timeline_idx = calSim_CapNSlide()
    tmp_sec = time.time() - tmp_start
    tmp_times = str(datetime.timedelta(seconds=tmp_sec)).split(".")
    time_list.append(tmp_times[0])

    tf_timeline_list = []
    for i, idx_val in enumerate(tf_timeline_idx):
        tf_timeline_list.append(captured_timeline_list[idx_val])
        print("[" + str(i + 1) + "번째 슬라이드 등장시간]", int(captured_timeline_list[idx_val] / 60), "분",
              round(captured_timeline_list[idx_val] % 60), "초")

    df['time'] = tf_timeline_list
    df.to_csv(save_path, mode='w')

    # txt 2 TTS 파일 생성
    tmp_start = time.time()
    txt2TTS()
    tmp_sec = time.time() - tmp_start
    tmp_times = str(datetime.timedelta(seconds=tmp_sec)).split(".")
    time_list.append(tmp_times[0])

    # 원본 강의 영상을 mp3로 변환 및 전환 시점에 맞추어 cut
    tmp_start = time.time()
    cutLectureMp3()
    tmp_sec = time.time() - tmp_start
    tmp_times = str(datetime.timedelta(seconds=tmp_sec)).split(".")
    time_list.append(tmp_times[0])

    total_sec = time.time() - total_start
    total_times = str(datetime.timedelta(seconds=total_sec)).split(".")

    # 장면 추출 시간, PDF to TXT 시간, 이미지 유사도 매칭 시간, CUT 시간, 총 시간
    print("\n■ PDF2JPG 추출 시간:", time_list[0])
    print("■ 장면 추출 시간:", time_list[1])
    print("■ PDF to TXT 시간:", time_list[2])
    print("■ 이미지 유사도 매칭 시간:", time_list[3])
    print("■ TTS 시간:", time_list[4])
    print("■ mp3 변환 및 CUT 시간:", time_list[5])
    print("■□■ 총 소요 시간:", total_times[0])