from __future__ import unicode_literals

import re
import os
import cv2
import time
import pytube
import numpy as np
from PIL import Image


class Video2Frames():
    def __init__(self, video):
        if self.youtube_url_validation(video):
            self.download_youtube(video)
        elif os.path.exists(video):
            self.videoFileName = video

    def download_youtube(self, url):
        yt = pytube.YouTube(url)
        print('Title:', yt.title)
        print('Author:', yt.author)
        print('Published date:', yt.publish_date.strftime("%Y-%m-%d"))
        print('Number of views:', yt.views)
        print('Length of video:', yt.length, 'seconds')
        self.videoFileName = 'Youtube-' + url.split('watch?v=')[-1] + '.mp4'
        yt.streams.filter(res='1080p', progressive=False).order_by(
            'resolution').desc().first().download(filename=self.videoFileName)
        print('Video successfullly downloaded from ', url)

    def youtube_url_validation(self, url):
        youtube_regex = (
            r'(https?://)?(www\.)?'
            '(youtube|youtu|youtube-nocookie)\.(com|be)/'
            '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')

        return re.match(youtube_regex, url)

    def float_to_str(self, seconds):
        hours = int(seconds / 3600)
        minutes = int((seconds-hours*3600) / 60)
        seconds = float(seconds % 60)

        if hours < 10:
            hours = f"0{hours}"
        if minutes < 10:
            minutes = f"0{minutes}"
        if int(seconds) < 10:
            seconds = f"0{seconds}"
        seconds = str(seconds)[:7]

        while True:
            if len(seconds) < 7:
                seconds = seconds + "0"
            else:
                break
        return f"{hours}h{minutes}m{seconds}s"

    def str_to_float(self, duaration):
        hours = int(duaration[:2])
        minutes = int(duaration[3:5])
        seconds = float(duaration[6:13])
        return hours*3600 + minutes*60 + seconds

    def check_similarity(self, img_path_1, img_path_2, threshold=0.9):
        i1 = np.asarray(Image.open(img_path_1)).flatten()
        i2 = np.asarray(Image.open(img_path_2)).flatten()
        print(f"Correlation: {np.corrcoef(i1, i2)[0, 1]}")
        return np.corrcoef(i1, i2)[0, 1] > threshold

    def toFrames(self):
        folder_name = '.'.join(self.videoFileName.split('.')[:-1])
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        video = cv2.VideoCapture(self.videoFileName)
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
        if int(major_ver) < 3:
            fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        else:
            fps = video.get(cv2.CAP_PROP_FPS)

        total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = float(total_frames) / float(fps)
        start_time = time.time()
        vidcap = cv2.VideoCapture(self.videoFileName)
        success, image = vidcap.read()

        count = 0
        base_image_path = ""

        while success:
            count += 1
            print(f"\nProcessing Frame: {count}/{int(total_frames)}")

            time_in_video = self.float_to_str(count/fps)
            file_name = f"{folder_name}-{time_in_video}.png"

            print(f"File Name: {file_name}")
            file_path = f"{folder_name}/{folder_name}-{time_in_video}.png"

            cv2.imwrite(file_path, image)
            print(f"Written: {file_path}")

            running_time = time.time() - start_time
            print(f"Running Time: {self.float_to_str(running_time)}")

            remaining_time = self.float_to_str(
                duration/self.str_to_float(time_in_video)*running_time)

            print(f"Estimated Remaining Time: {remaining_time}")

            if base_image_path == '':
                base_image_path = file_path
                print(f"Current Base Image: {base_image_path}")
            else:
                print(f"Current Base Image: {base_image_path}")
                if self.check_similarity(base_image_path, file_path):
                    os.remove(file_path)
                    print(f"Removed: {file_name}")

                else:
                    base_image_path = file_path

            success, image = vidcap.read()


URL = 'Youtube-fRsjv-JKyf8.mp4'
Video2Frames(URL).toFrames()
