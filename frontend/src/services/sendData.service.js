import api from './api'

const sendDataWithUrlForVideoAudioAnalysis = (url, email) => {
    return api.post(`/youtube_video_audio?youtube-url=${url}&email=${email}`)
}

const sendDataWithUrlForAudioAnalysis = (url, email) => {
    return api.post(`/youtube_audio?youtube-url=${url}&email=${email}`)
}

const sendDataWithUrlForVideoAnalysis = (url, email) => {
    return api.post(`/youtube_video?youtube-url=${url}&email=${email}`)
}

const sendDataService = {
    sendDataWithUrlForVideoAudioAnalysis,
    sendDataWithUrlForAudioAnalysis,
    sendDataWithUrlForVideoAnalysis
};

export default sendDataService;