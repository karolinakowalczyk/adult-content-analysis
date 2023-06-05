import api from "./api";

const getAnalysedVideo = (videoId) => {
  return api.get(`/youtube_video/${videoId}`);
};

const getAnalysedAudio = (audioId) => {
  return api.get(`/youtube_audio/${audioId}`);
};

const getAnalysedDataService = {
  getAnalysedVideo,
  getAnalysedAudio,
};

export default getAnalysedDataService;
