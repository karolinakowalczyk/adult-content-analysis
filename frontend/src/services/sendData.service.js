import api from './api'

const sendDataWithUrlForVideoAudioAnalysis = (url, email, file) => {
    let formdata = new FormData();
    formdata.append('file',file[0]);
    return api.post(`/youtube_video_audio?youtube-url=${url}&email=${email}`, formdata, {
        headers: {
          "Content-Type": "multipart/form-data",
        }
      })
}

const sendDataWithUrlForAudioAnalysis = (url, email, file) => {
    let formdata = new FormData();
    formdata.append('file',file[0]);
    return api.post(`/youtube_audio?youtube-url=${url}&email=${email}`, formdata, {
        headers: {
          "Content-Type": "multipart/form-data",
        }
      })
}

const sendDataWithUrlForVideoAnalysis = (url, email, file) => {
    let formdata = new FormData();
    formdata.append('file',file[0]);
    return api.post(`/youtube_video?youtube-url=${url}&email=${email}`, formdata, {
        headers: {
          "Content-Type": "multipart/form-data",
        }
      })
}

const sendDataService = {
    sendDataWithUrlForVideoAudioAnalysis,
    sendDataWithUrlForAudioAnalysis,
    sendDataWithUrlForVideoAnalysis
};

export default sendDataService;