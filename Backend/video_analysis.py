import numpy as np
from flask import Flask, request, jsonify
from pymongo import MongoClient
import os
import uuid
import yt_dlp
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import librosa
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm
from threading import Thread
from emailService import sendEmail

USE_MODEL = False
REMOVE_AFTER_PROCESSING = False
BATCH_SIZE = 4

app = Flask(__name__)

client = MongoClient('localhost', 27017)

db = client.flask_db

videodb = db.video
audiodb = db.audio


if USE_MODEL == False:
    # with open('./labels_map.txt') as f:
    #     labels_map = json.load(f)
    # labels_map = [labels_map[str(i)] for i in range(1000)]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("cpu")
    print(device)
    image_model = EfficientNet.from_pretrained('efficientnet-b4')
    image_model.to(device).eval()

tfms = transforms.Compose([transforms.Resize(224, antialias=True),
                        transforms.Normalize(
                                [0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225]),
                        ])

AUDIO_EXTENSION = 'wav'
VIDEO_EXTENSION = 'mp4'

model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")
tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")


from torch.utils.data import Dataset

class CustomTensorDataset(Dataset):
  def __init__(self, dataset, transform_list=None):
    self.tensors = dataset
    self.transforms = transform_list

  def __getitem__(self, index):
    x = self.tensors[index]

    if self.transforms:
      x = self.transforms(x)

    return x

  def __len__(self):
    return self.tensors.size(0)


def analyse_frame(image):
    if USE_MODEL == True:
        img = tfms(image).to(device)
        output = image_model(img)
        class_idx = output.argmax(dim=-1)
        return class_idx
    else:
        if image.dim() == 4:
            return np.random.beta(1, 5, image.shape[0])
        else:
            return np.random.beta(1, 5)
    

def download_audio(url, output_path):
    ydl_opts = {
        'outtmpl': output_path + '/%(id)s.%(ext)s',
        'format': 'm4a/bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': AUDIO_EXTENSION,
    }]}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        info_with_audio_extension = dict(info)
        info_with_audio_extension['ext'] = AUDIO_EXTENSION
        return ydl.prepare_filename(info_with_audio_extension), info

def download_video(url, output_path):
    
    ydl_opts = {
        'outtmpl': output_path + '/%(id)s.%(ext)s',
        'format': '160',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        info_with_video_extension = dict(info)
        info_with_video_extension['ext'] = VIDEO_EXTENSION
        return ydl.prepare_filename(info_with_video_extension), info

@app.route('/youtube_audio',methods=['POST'])
def analyse_audio():

    youtubeurl = request.args.get('youtube-url')

    out_wav_file, info_dict = download_audio(youtubeurl, '.')
    print(out_wav_file)
    t = Thread(target=anylyze_youtube_audio, args=(out_wav_file, info_dict,request.args.get('email')))
    t.start()
    return ('Your audio was succesfully downloaded and sent to analyze.\n After completion, you will receive a message with an email link to the results.', 200)

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


def anylyze_youtube_audio(out_wav_file, info_dict, mail):
    speech, rate = librosa.load(out_wav_file, sr=16000)
    input_values = feature_extractor(speech, return_tensors="pt").input_values
    logits = model(input_values).logits[0]
    pred_ids = torch.argmax(logits, axis=-1)

    transcriptions = tokenizer.decode(pred_ids, output_word_offsets=True)
    time_offset = model.config.inputs_to_logits_ratio / feature_extractor.sampling_rate

    audio_id = str(uuid.uuid4())
    words = {
        "audio_id": audio_id,
        "data": [{
            "word": d["word"],
            "start_time": round(d["start_offset"] * time_offset, 2),
            "end_time": round(d["end_offset"] * time_offset, 2),
            "censored": 1 if np.random.beta(1, 5) > 0.5 else 0
        } for d in transcriptions.word_offsets ],
        "url": info_dict.get('webpage_url'),
        "title": info_dict.get('title')
    }
    audiodb.insert_one(words)
    os.remove(out_wav_file)

    link = "http://localhost:5000/youtube_audio/"+audio_id
    sendEmail(mail, "Audio analysis has been completed",link)


    

@app.route('/youtube_video',methods=['POST'])
def analyse_frames():

    
    youtubeurl = request.args.get('youtube-url')

    out_mp4_file, info_dict = download_video(youtubeurl, '.')
    print(out_mp4_file)
    t = Thread(target=analyse_video_frames, args=(out_mp4_file, info_dict,request.args.get('email')))
    t.start()
    return ('Your video was succesfully downloaded and sent to analyze.\n After completion, you will receive a message with an email link to the results.', 200)


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

def analyse_video_frames(out_mp4_file, info_dict, mail):

    stream = "video"

    torch.cuda.empty_cache()
    video_frames = torch.empty(0)
    video_pts = []
    video = torchvision.io.VideoReader(out_mp4_file, stream)
    meta_data = video.get_metadata()
    print(meta_data)
    fps = round(meta_data['video']['fps'][0])

    video.set_current_stream("video")
    frames = []
    for frame in video:
        frames.append(frame['data'].type(dtype=torch.float32))
        video_pts.append(frame['pts'])
    if len(frames) > 0:
        
        video_frames = torch.stack(frames, 0)

    ds = CustomTensorDataset(dataset=video_frames, transform_list=tfms)
    

    video.container.close()

    dataloader = DataLoader(ds, batch_size=BATCH_SIZE)

    if REMOVE_AFTER_PROCESSING == True:
        os.remove(out_mp4_file)

    
    values = torch.empty(0) if USE_MODEL else []
    i = 0

    with tqdm(dataloader, unit="batch") as tepoch:
        for imgs in tepoch:
            outputs = analyse_frame(imgs)
            if USE_MODEL:
                values = torch.hstack([values, outputs.detach().cpu()])
            else:
                values = [*values, *outputs]
    data = {}
    i = 0
    for pts in video_pts:
        data[round(pts, 3)] = int(values[i].item()) if USE_MODEL else round(values[i], 3)
        i += 1

    video_id = str(uuid.uuid4())
    analysis = {}
    analysis["video_id"] = video_id
    analysis["data"] = [{
            "timestamp": d,
            "value": data[d]
        } for d in data ]
    analysis["url"] = info_dict.get('webpage_url')
    analysis["title"] = info_dict.get('title')

    videodb.insert_one(analysis)

    link = "http://localhost:5000/youtube_video/"+video_id
    sendEmail(mail, "Video analysis has been completed",link)

@app.route('/youtube_video_info',methods=['GET'])
def get_video_info():

    youtubeurl = request.args.get('youtube-url')

    ydl = yt_dlp.YoutubeDL({})
    info_dict = ydl.extract_info(youtubeurl, download=False)
    title = info_dict.get('title')
    link = info_dict.get('webpage_url')

    return jsonify({"title": title, "link": link})

if __name__ == "__main__":
    app.run(debug=True)