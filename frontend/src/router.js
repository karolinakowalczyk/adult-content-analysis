import { createBrowserRouter } from "react-router-dom";
import { LineChart } from "./components/line-chart/LineChart";
import App from "./App";

const router = createBrowserRouter([
  {
    path: "/",
    element: <App />,
  },
  {
    path: "youtube_video/:video_id",
    element: <LineChart />,
  },
]);

export default router;
