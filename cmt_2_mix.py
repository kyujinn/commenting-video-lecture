"""
* 코드 파일 이름: cmt_2_mix.py
* 코드 작성자: 박동연, 강소정, 김유진
* 코드 설명: 동영상 강의 해설 파일을 생성하기 위해 전처리 파일을 조합하여 최종 해설 파일 생성
* 코드 최종 수정일: 2021/06/27 (박동연)
* 문의 메일: yeon0729@sookmyung.ac.kr
"""

# 패키지 및 라이브러리 호출
import sys
import time
import datetime
import pandas as pd
from pydub import AudioSegment
import os

pdf2image_module_path = "data/Release-21.03.0/poppler-21.03.0/Library/bin/"

# 경로 설정 (경로 내에 한글 디렉토리 및 한글 파일이 있으면 제대로 동작하지 않음 유의 !!!!!)
default_path = "Datamining/"
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

# 모든 오디오 파일을 조합하여 최종 믹싱을 하는 함수
def mix():
    print("\n[lec과 tts 병합 시작] 원본 mp3파일과 TTS mp3파일 병합을 시작합니다")
    tts_list = os.listdir(tts_path)
    tts_list = [tts_file for tts_file in tts_list if tts_file.endswith(".mp3")]  # tts 가져오기
    tts_list.sort()
    print(">>> tts mp3 파일 목록:", tts_list)

    lec_list = os.listdir(lec_path)
    lec_list = [lec_file for lec_file in lec_list if lec_file.endswith(".mp3")]  # lec 가져오기
    lec_list.sort()
    print(">>> lec mp3 파일 목록:", lec_list)

    combined = AudioSegment.from_mp3(default_path + "lec/lec_001.mp3")

    for i, tts_file in enumerate(tts_list):
        tts_list[i] = AudioSegment.from_mp3(default_path + "tts/" + "tts_" + set_Filenum_of_Name(i + 1) + ".mp3")

        # tts mp3 file 10 dB lower
        tts_list[i] = tts_list[i] - 10
        lec_list[i] = AudioSegment.from_mp3(default_path + "lec/" + "lec_" + set_Filenum_of_Name(i + 1) + ".mp3")

        if (i != 0):
            combined = combined + tts_list[i] + lec_list[i]
        else:
            combined = tts_list[i] + lec_list[i]

        print(set_Filenum_of_Name(i + 1) + "번째 tts+lec MP3 mixed!")

    try:
        if not os.path.exists(mix_path):  # 디렉토리 없을 시 생성
            os.makedirs(mix_path)
    except OSError:
        print('Error: Creating directory. ' + mix_path)  # 디렉토리 생성 오류

    # simple export
    file_handle = combined.export(mix_path + "mix.mp3", format="mp3")

    print("[lec과 tts 병합 종료] 원본 mp3파일과 TTS mp3파일 병합을 종료합니다\n")

# 메인함수
if __name__ == '__main__':
    time_list = []  # pdf 2 image, 장면 추출 시간, OCR 시간, ORB 유사도 시간
    total_start = time.time()

    # 모든 오디오 파일 병합 - mix
    tmp_start = time.time()
    mix()
    tmp_sec = time.time() - tmp_start
    tmp_times = str(datetime.timedelta(seconds=tmp_sec)).split(".")
    time_list.append(tmp_times[0])

    total_sec = time.time() - total_start
    total_times = str(datetime.timedelta(seconds=total_sec)).split(".")

    # 장면 추출 시간, PDF to TXT 시간, 이미지 유사도 매칭 시간, CUT 시간, MIX 시간, 총 시간
    print("\n■ MIX 시간:", time_list[0])
    print("■□■ 총 소요 시간:", total_times[0])