import { createBrowserRouter } from "react-router-dom";
import { LineChart } from "./components/line-chart/LineChart";
import App from "./App";
import { TimelineChart } from "./components/timeline-chart/TimelineChart";

const router = createBrowserRouter([
  {
    path: "/",
    element: <App />,
  },
  {
    path: "youtube_video/:video_id",
    element: <LineChart />,
  },
  {
    path: "youtube_audio/:audio_id",
    element: <TimelineChart />,
  },
]);

export default router;
