import yt_dlp
import os
import uuid
from emailService import sendEmail
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm



AUDIO_EXTENSION = 'wav'
VIDEO_EXTENSION = 'mp4'
FRAMES_PER_CHUNK = 3600
REMOVE_AFTER_PROCESSING = True
USE_MODEL = True
BATCH_SIZE = 30

profanity_words_file = open("profanity_words_list.txt", "r")
data = profanity_words_file.read()
profanity_words_list = data.split("\n")
profanity_words_file.close()
print("loaded profanity words list")

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

def download_video(url, output_path, ydl_opts=None):

    ydl_opts = {
        'outtmpl': output_path + '/%(id)s.%(ext)s',
        'format': '134',
    } if ydl_opts == None else ydl_opts
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        info_with_video_extension = dict(info)
        info_with_video_extension['ext'] = VIDEO_EXTENSION
        return ydl.prepare_filename(info_with_video_extension), info
    

def anylyze_audio(audiodb,
                  audioModel,
                  out_wav_file,
                  info_dict,
                  mail,
                  id=None,
                  video_audio=False,
                  result_queue=None,
                  video_file=None):


    result = audioModel.transcribe(out_wav_file, word_timestamps=True, verbose=True)

    for s in result["segments"]:
        text = (" " + s["text"] + " ").lower().replace(".", "").replace(",", "").replace("?", "")
        value = False
        for word in profanity_words_list:
            value = (" " + word + " ") in text
            if value:
                break
        s["censored"] = 1 if value else 0

    audio_id = str(uuid.uuid4())
    words = {
        "audio_id": audio_id,
        "data": [{
            "word": s["text"],
            "start_time": s["start"],
            "end_time": s["end"],
            "censored": s["censored"]
        } for s in result["segments"] ],
        "url": info_dict.get('webpage_url'),
        "title": info_dict.get('title')
    }

    audiodb.insert_one(words)

    if REMOVE_AFTER_PROCESSING:
        os.remove(out_wav_file)
    if video_file != None:
        os.remove(video_file)

    if video_audio:
        if not result_queue.empty():
            video_link = result_queue.get()[1]
            audio_link = "http://localhost:3000/youtube_audio/"+audio_id
            sendEmail(mail, "Analysis has been completed", video_link, audio_link, info_dict.get('title'))
        else:
            result_queue.put((id, "http://localhost:3000/youtube_audio/"+audio_id))
    else:
        link = "http://localhost:3000/youtube_audio/"+audio_id
        sendEmail(mail, "Audio analysis has been completed", None, link, info_dict.get('title'))

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

def analyse_frame(image, image_model, device):
    if USE_MODEL == True:
        with torch.no_grad():
            image_model.eval()
            image = image.to(device)
            output = image_model(image)
        return output
    else:
        return np.random.beta(1, 5, image.shape[0])

def analyse_video_frames(out_mp4_file, videodb, image_model, device, info_dict, mail, id=None, video_audio=False, result_queue=None):

    tfms = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(320, interpolation=transforms.InterpolationMode("bicubic")),
            transforms.CenterCrop(size=(320, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    stream = "video"

    video_frames = torch.empty(0)
    video_pts = []
    video = torchvision.io.VideoReader(out_mp4_file, stream)
    meta_data = video.get_metadata()
    print(meta_data)

    total_frames = int(meta_data['video']['duration'][0] * meta_data['video']['fps'][0])
    print("Frames: ", total_frames)
    total_steps = int(total_frames / FRAMES_PER_CHUNK)
    step = 0
    curr_frame = 0

    frames = []
    values = []
    # print("Video frames: ", len(video.))
    for frame in video:
        frames.append(frame['data'])
        video_pts.append(frame['pts'])
        curr_frame += 1
        if len(frames) % FRAMES_PER_CHUNK == 0:
            video_frames = torch.stack(frames, 0)
            ds = CustomTensorDataset(dataset=video_frames, transform_list=tfms)
            dataloader = DataLoader(ds, batch_size=BATCH_SIZE)
            
            print("STEP: ", "{}/{}".format(step, total_steps))
            with tqdm(dataloader, unit="batch") as tepoch:
                for imgs in tepoch:
                    if USE_MODEL:
                        outputs = analyse_frame(imgs, image_model, device).detach().cpu()
                        for o in outputs:
                            values.append(o.item())
                    else:
                        outputs = np.random.beta(1, 5, imgs.shape[0])
                        values = [*values, *outputs]
            step += 1
            frames = []
    if frames != []:
        video_frames = torch.stack(frames, 0)
        ds = CustomTensorDataset(dataset=video_frames, transform_list=tfms)
        dataloader = DataLoader(ds, batch_size=BATCH_SIZE)
        
        print("STEP: ", "{}/{}".format(step, total_steps))
        with tqdm(dataloader, unit="batch") as tepoch:
            for imgs in tepoch:
                if USE_MODEL:
                    outputs = analyse_frame(imgs, image_model, device).detach().cpu()
                    for o in outputs:
                        values.append(o.item())
                else:
                    outputs = np.random.beta(1, 5, imgs.shape[0])
                    values = [*values, *outputs]
        step += 1
        frames = []

    video.container.close()

    

    if REMOVE_AFTER_PROCESSING == True:
        os.remove(out_mp4_file)

    data = {}
    i = 0
    for pts in video_pts:
        data[round(pts, 3)] = round(values[i], 3)
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

    if video_audio:
        if not result_queue.empty():
            audio_link = result_queue.get()[1]
            video_link = "http://localhost:3000/youtube_video/"+video_id
            sendEmail(mail, "Analysis has been completed", video_link, audio_link, info_dict.get('title'))
        else:
            result_queue.put((id, "http://localhost:3000/youtube_video/"+video_id))
    else:
        link = "http://localhost:3000/youtube_video/"+video_id
        sendEmail(mail, "Video analysis has been completed", link, None, info_dict.get('title'))