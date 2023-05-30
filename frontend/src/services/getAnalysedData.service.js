import api from "./api";

const getAnalysedVideo = (videoId) => {
  return api.get(`/youtube_video/${videoId}`);
};

const getAnalysedDataService = {
  getAnalysedVideo,
};

export default getAnalysedDataService;
