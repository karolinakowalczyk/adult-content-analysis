import numpy as np
from flask import Flask, request, jsonify
import os
import yt_dlp
import torchvision.transforms as transforms
import torch
import librosa
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC
from efficientnet_pytorch import EfficientNet
import cv2

USE_MODEL = False

app = Flask(__name__)

if USE_MODEL == True:
    # with open('./labels_map.txt') as f:
    #     labels_map = json.load(f)
    # labels_map = [labels_map[str(i)] for i in range(1000)]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("cpu")
    print(device)
    image_model = EfficientNet.from_pretrained('efficientnet-b4')
    image_model.to(device).eval()

tfms = transforms.Compose([transforms.ToPILImage(),
                        transforms.Resize(224),
                        transforms.ToTensor(),
                        transforms.Normalize(
                                [0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225]),
                        ])

AUDIO_EXTENSION = 'wav'

model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")
tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

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

def frame_to_milisec(fps, frame_num):
    return int(float(frame_num / fps) * 100 + int(frame_num % fps / 30. * 100))

@app.route('/youtube_audio',methods=['POST'])
def analyse_audio():

    youtubeurl = request.args.get('youtube-url')

    out_wav_file, info_dict = download_audio(youtubeurl, '.')
    print(out_wav_file)

    speech, rate = librosa.load(out_wav_file, sr=16000)
    input_values = feature_extractor(speech, return_tensors="pt").input_values
    logits = model(input_values).logits[0]
    pred_ids = torch.argmax(logits, axis=-1)

    transcriptions = tokenizer.decode(pred_ids, output_word_offsets=True)
    time_offset = model.config.inputs_to_logits_ratio / feature_extractor.sampling_rate

    words = {
        "data": [{
            "word": d["word"],
            "start_time": round(d["start_offset"] * time_offset, 2),
            "end_time": round(d["end_offset"] * time_offset, 2),
            "censored": 1 if np.random.beta(1, 5) > 0.5 else 0
        } for d in transcriptions.word_offsets ],
        "url": info_dict.get('webpage_url'),
        "title": info_dict.get('title')
    }

    os.remove(out_wav_file)

    response = jsonify(words)
    return response

@app.route('/youtube_video',methods=['POST'])
def analyse_frames():

    youtubeurl = request.args.get('youtube-url')

    ydl = yt_dlp.YoutubeDL({})
    info_dict = ydl.extract_info(youtubeurl, download=False)
    formats = info_dict.get('formats')[::-1]
    
    for f in formats:
        if f.get('format_note', None) == '144p' and f.get('format_id', None) == '278':
            url = f.get('url',None)
            fps = f.get('fps',None)

            cap = cv2.VideoCapture(url)

            if not cap.isOpened():
                print('video not opened')
                exit(-1)

            frame_num = 0
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                img = np.copy(frame)
                img = tfms(frame)

                frames.append(img)
                    
                frame_num += 1

                if cv2.waitKey(30)&0xFF == ord('q'):
                    break

            cap.release()


    cv2.destroyAllWindows()

    analysis = {}
    analysis["data"] = {}
    frame_num = 0
    for fr in frames:
        if USE_MODEL == True:
            im = fr[None,:,:,:].to(device)
            output = image_model(im)
            class_idx = output.argmax(dim=-1)
            value = np.random.beta(1, 5, class_idx.size())[0]
        value = np.random.beta(1, 5)
        analysis["data"]["{}".format(frame_to_milisec(fps, frame_num))] = value
        frame_num += 1

    analysis["url"] = info_dict.get('webpage_url')
    analysis["title"] = info_dict.get('title')

    response = jsonify(analysis)
    return response

@app.route('/youtube_video_info',methods=['POST'])
def get_video_info():

    youtubeurl = request.args.get('youtube-url')

    ydl = yt_dlp.YoutubeDL({})
    info_dict = ydl.extract_info(youtubeurl, download=False)
    title = info_dict.get('title')
    link = info_dict.get('webpage_url')

    return jsonify({"title": title, "link": link})

if __name__ == "__main__":
    app.run(debug=True)