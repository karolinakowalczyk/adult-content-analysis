import React, { useState, useEffect } from "react";
import GetAnalysedDataService from "../../services/getAnalysedData.service";
import "./LineChart.scss";
import Container from "react-bootstrap/Container";
import Alert from "react-bootstrap/Alert";
import { useParams } from "react-router-dom";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  Title,
  Tooltip,
  LineElement,
  Legend,
  CategoryScale,
  LinearScale,
  PointElement,
} from "chart.js";
ChartJS.register(
  Title,
  Tooltip,
  LineElement,
  Legend,
  CategoryScale,
  LinearScale,
  PointElement
);

export function LineChart() {
  const [videoId, setVideoId] = useState();
  const [chartLabels, setChartLabels] = useState([]);
  const [chartValues, setChartValues] = useState([]);
  const [displayChart, setDisplayChart] = useState(false);
  const [videoTitle, setVideoTitle] = useState("");
  const [videoUrl, setVideoUrl] = useState("");
  const [error, setError] = useState("");
  const id = useParams();

  useEffect(() => {
    console.log("tu");
    setVideoId(id.video_id);
    if (videoId) {
      GetAnalysedDataService.getAnalysedVideo(videoId)
        .then(
          (response) => {
            setVideoTitle(response.data.title);
            setVideoUrl(response.data.url);
            let timetamps = [];
            let results = [];
            for (let d in response.data.data) {
              timetamps.push(d);
              results.push(response.data.data[d]);
            }
            setChartLabels(timetamps);
            setChartValues(results);
            setDisplayChart(true);
          },
          (error) => {
            const resMessage =
              (error.response &&
                error.response.data &&
                error.response.data.message) ||
              error.message ||
              error.toString();
            setError(resMessage);
          }
        )
        .catch((error) => setError(error));
    }
  }, [videoId, displayChart, id]);

  const data = {
    labels: chartLabels,
    datasets: [
      {
        label: videoTitle,
        data: chartValues,
        backgroundColor: "violet",
        borderColor: "violet",
      },
    ],
  };
  return (
    <Container>
      <a href={videoUrl} target="_blank" rel="noreferrer">
        Link to analized video
      </a>
      {displayChart && (
        <Line
          data={data}
          options={{
            scales: {
              y: {
                beginAtZero: true,
                title: {
                  display: true,
                  text: "Adult content intensity",
                },
              },
              x: {
                title: {
                  display: true,
                  text: "Video Timestamp [s]",
                },
              },
            },
          }}
        />
      )}
      {error && <Alert variant="danger">Something went wrong...</Alert>}
    </Container>
  );
}
