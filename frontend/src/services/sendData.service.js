import api from './api'

const sendDataWithUrl = (url, email) => {
    return api.post(`/youtube_video?youtube-url=${url}&email=${email}`)
}

const sendDataService = {
    sendDataWithUrl
};

export default sendDataService;