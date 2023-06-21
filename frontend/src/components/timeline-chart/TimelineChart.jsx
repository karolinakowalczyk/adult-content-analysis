import React, { useState, useEffect } from "react";
import { useParams } from "react-router-dom";
import GetAnalysedDataService from "../../services/getAnalysedData.service";
import Alert from "react-bootstrap/Alert";
import "./TimelineChart.scss";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  Title,
  Tooltip,
  LineElement,
  Legend,
  CategoryScale,
  LinearScale,
  PointElement,
  BarElement,
  TimeScale,
} from "chart.js";
import Container from "react-bootstrap/esm/Container";
ChartJS.register(
  Title,
  Tooltip,
  LineElement,
  Legend,
  CategoryScale,
  LinearScale,
  PointElement,
  BarElement,
  TimeScale
);

export function TimelineChart() {
  const [audioId, setAudioId] = useState();
  const [error, setError] = useState("");
  const [displayChart, setDisplayChart] = useState(false);
  const [goodWords, setGoodWords] = useState([]);
  const [badWords, setBadWords] = useState([]);
  const [audioTitle, setAudioTitle] = useState("");
  const [audioUrl, setAudioUrl] = useState("");
  const id = useParams();

  useEffect(() => {
    setAudioId(id.audio_id);
    if (audioId) {
      GetAnalysedDataService.getAnalysedAudio(audioId)
        .then(
          (response) => {
            let goodBufor = [];
            let badBufor = [];
            // let words = [];

            response.data.data.forEach((element) => {
              if (element["censored"] === 1) {
                badBufor.push({
                  word: element["word"],
                  x: [element["start_time"], element["end_time"]],
                  y: "Censored",
                });
              } else if (element["censored"] === 0) {
                goodBufor.push({
                  word: element["word"],
                  x: [element["start_time"], element["end_time"]],
                  y: "Censored",
                });
              }
            });
            setAudioTitle(response.data.title);
            setAudioUrl(response.data.url);
            setGoodWords(goodBufor);
            setBadWords(badBufor);
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
        .catch((error) => {
          setError(error);
        });
    }
  }, [displayChart, audioId, id]);
  const data = {
    labels: ["Censored"],
    datasets: [
      {
        label: "Yes",
        data: badWords,
        backgroundColor: ["rgb(255, 99, 132)"],
        borderColor: ["rgb(255, 99, 132)"],
        borderWidth: 1,
      },
      {
        label: "No",
        data: goodWords,
        backgroundColor: ["rgb(75, 192, 192)"],
        borderColor: ["rgb(75, 192, 192)"],
        borderWidth: 1,
      },
    ],
  };
  return (
    <Container>
      <a href={audioUrl} target="_blank" rel="noreferrer">
        Link to analized audio
      </a>
      {displayChart && (
        <Bar
          data={data}
          options={{
            indexAxis: "y",
            barPercentage: 1,
            categoryPercentage: 1,
            scales: {
              x: {
                title: {
                  display: true,
                  text: "Audio Timestamp [s]",
                },
              },
              y: {
                beginAtZero: true,
                stacked: true,
              },
            },
            plugins: {
              title: {
                display: true,
                text: audioTitle,
              },
              tooltip: {
                callbacks: {
                  title: function (context) {
                    console.log()
                    return `${context[0].raw.word}`;
                  },
                },
              },
              legend: {
                display: false,
              },
            },
          }}
        />
      )}
      {error && <Alert variant="danger">Something went wrong...</Alert>}
    </Container>
  );
}
