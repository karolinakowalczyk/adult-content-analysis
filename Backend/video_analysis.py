import numpy as np
from flask import Flask, request, jsonify
from pymongo import MongoClient
import os
import uuid
import yt_dlp
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from threading import Thread
import queue
from emailService import sendEmail
from modelHelpers import getModel
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pydub import AudioSegment
import whisper
import matplotlib.pyplot as plt
import matplotlib
from utils import download_audio, download_video, anylyze_audio, analyse_video_frames

DEBUG = False

USE_MODEL = True
REMOVE_AFTER_PROCESSING = True
BATCH_SIZE = 30
FRAMES_PER_CHUNK = 3600
SAMPLE_RATE = 16000
# AUDIO_EXTENSION = 'wav'
# VIDEO_EXTENSION = 'mp4'


app = Flask(__name__)
CORS(app)

client = MongoClient('localhost', 27017)

db = client.flask_db

videodb = db.video
audiodb = db.audio



if USE_MODEL == True:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("cpu")
    print(device)
    image_model = getModel()
    image_model.to(device).eval()
    print("model loaded")



audio_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
audio_device = torch.device("cpu")
audioModel = whisper.load_model("base").to(audio_device)



@app.route('/youtube_audio',methods=['POST'])
def analyse_audio():
    if 'file' in request.files:
        file = request.files['file']
        file_name=secure_filename(file.filename)
        file.save(file_name)
        name=file_name.split(".")[0]
        sound = AudioSegment.from_file(file_name,format="mp4")
        print(file_name)
        sound.export(name+'.wav', format="wav")
        info_dict = {"webpage_url": "from file", "title": name}
        t = Thread(target=anylyze_audio, args=(audiodb,
                                               audioModel,
                                               name+'.wav',
                                               info_dict,request.args.get('email'),
                                               None, False, None, file_name))
        t.start()
    else:
        youtubeurl = request.args.get('youtube-url')

        out_wav_file, info_dict = download_audio(youtubeurl, '.')
        print(out_wav_file)
        t = Thread(target=anylyze_audio, args=(audiodb,
                                               audioModel,
                                               out_wav_file,
                                               info_dict,
                                               request.args.get('email')))
        t.start()
    return ('Your audio was succesfully downloaded and sent to analyze.\n After completion, you will receive a message with an email link to the results.', 200)

@app.route('/youtube_video',methods=['POST'])
def analyse_frames():
    if 'file' in request.files:
        file = request.files['file']
        file_name=secure_filename(file.filename)
        file.save(file_name)
        name=file_name.split(".")[0]
        info_dict = {"webpage_url": "from file", "title": name}
        t = Thread(target=analyse_video_frames, args=(file_name,
                                                      videodb,
                                                      image_model,
                                                      device,
                                                      info_dict,
                                                      request.args.get('email')))
        t.start()
    else:
        print("started")
        youtubeurl = request.args.get('youtube-url')

        out_mp4_file, info_dict = download_video(youtubeurl, '.')
        print(out_mp4_file)
        t = Thread(target=analyse_video_frames, args=(out_mp4_file,
                                                      videodb,
                                                      image_model,
                                                      device,
                                                      info_dict,
                                                      request.args.get('email')))
        t.start()
    return ('Your video was succesfully downloaded and sent to analyze.\n After completion, you will receive a message with an email link to the results.', 200)


@app.route('/youtube_video_audio',methods=['POST'])
def analyse_video_audio():
    if 'file' in request.files:
        file = request.files['file']
        file_name=secure_filename(file.filename)
        file.save(file_name)
        name=file_name.split(".")[0]
        sound = AudioSegment.from_file(file_name,format="mp4")
        sound.export(name+'.wav', format="wav")
        info_dict = {"webpage_url": "from file", "title": name}
        q = queue.Queue()
        a = Thread(target=anylyze_audio, args=(audiodb,
                                               audioModel,
                                               name+'.wav',
                                               info_dict,
                                               request.args.get('email'),
                                               0, True, q))
        v = Thread(target=analyse_video_frames, args=(file_name,
                                                      videodb,
                                                      image_model,
                                                      device,
                                                      info_dict,
                                                      request.args.get('email'),
                                                      1, True, q))

        a.start()
        v.start()
    else:
        youtubeurl = request.args.get('youtube-url')

        out_wav_file, info_dict = download_audio(youtubeurl, '.')
        print(out_wav_file)
        out_mp4_file, info_dict = download_video(youtubeurl, '.')
        print(out_mp4_file)
    
        q = queue.Queue()
        a = Thread(target=anylyze_audio, args=(audiodb,
                                               audioModel,
                                               out_wav_file,
                                               info_dict,
                                               request.args.get('email'),
                                               0, True, q))
        v = Thread(target=analyse_video_frames, args=(out_mp4_file,
                                                      videodb,
                                                      image_model,
                                                      device,
                                                      info_dict,
                                                      request.args.get('email'),
                                                      1, True, q))

        a.start()
        v.start()


    return ('Data was succesfully downloaded and sent to analyze.\n After completion, you will receive a message with an email link to the results.', 200)


@app.route('/youtube_audio/<id>', methods=['GET'])
def get_anyleze_audio(id):
    words = audiodb.find_one({"audio_id":id})
    if words is None:
      return  ('', 404)
    response = {
        "audio_id": words['audio_id'],
        "data": [{
            "word": d['word'],
            "start_time": d['start_time'],
            "end_time": d['end_time'],
            "censored": d['censored']
        } for d in words['data'] ],
        "url": words['url'],
        "title": words['title']
    }
    
    return (jsonify(response), 200)








@app.route('/youtube_video/<id>', methods=['GET'])
def get_anylyze_video(id):
    data = videodb.find_one({"video_id":id})
    if data is None:
      return  ('', 404)
    
    response = {}
    response["video_id"] = data['video_id']
    response["url"] = data['url']
    response["title"] = data['title']
    video_data = {}
    for d in data['data']:
        video_data.update({d['timestamp']:d['value']})
    response["data"] = video_data
    return (jsonify(response), 200)

if DEBUG:
    @app.route('/save_frames_yt',methods=['GET'])
    def save_frames_from_video():
        youtubeurl = request.args.get('youtube-url')
        skipframes = int(request.args.get('skip-frames'))
        bad = int(request.args.get('bad'))
        stream = "video"

        out_mp4_file, info_dict = download_video(youtubeurl, '.')

        video = torchvision.io.VideoReader(out_mp4_file, stream)
        name = out_mp4_file.split(".")[0]
        frames_n = 0
        saved_n = 0
        for frame in video:
            frames_n += 1
            if frames_n % skipframes == 0:
                img = frame['data'].numpy().transpose((1, 2, 0))
                if bad == 1:
                    matplotlib.image.imsave(os.getcwd() + "\\bad\\{}_{}.png".format(name, frames_n), img)
                else:
                    matplotlib.image.imsave(os.getcwd() + "\\images\\{}_{}.png".format(name, frames_n), img)
                saved_n += 1
        
        video.container.close()
        os.remove(out_mp4_file)
        
        return ("saved {} images".format(saved_n))


if DEBUG:
    @app.route('/youtube_video_info',methods=['GET'])
    def get_video_info():

        youtubeurl = request.args.get('youtube-url')
        ydl = yt_dlp.YoutubeDL({})
        info_dict = ydl.extract_info(youtubeurl, download=False)
        # # print(info_dict)

        formats = info_dict["formats"]

        best_video = next(f for f in formats
                        if f['vcodec'] != 'none' and f['acodec'] == 'none')
        
        audio_ext = {'mp4': 'm4a'}[best_video['ext']]
        best_audio = next(f for f in formats if (
            f['acodec'] != 'none' and f['vcodec'] == 'none' and f['ext'] == audio_ext))

        ydl_opts = {
            'outtmpl': os.getcwd() + '/%(id)s.%(ext)s',
            "acodec": "mp4a.40.2",
            'ext': 'mp4',
            "format": "18",
            'format_id': '18' #{
                #'format_id': '18',
                # 'ext': 'mp4',
                # 'requested_formats':  [best_video, best_audio],
                # 'protocol': f'{best_video["protocol"]}+{best_audio["protocol"]}'
            #}
        }

        path = download_video(youtubeurl, os.getcwd(), ydl_opts=ydl_opts)
        # title = info_dict.get('title')
        # link = info_dict.get('webpage_url')

        return jsonify(info_dict)

if __name__ == "__main__":
    app.run(debug=True)