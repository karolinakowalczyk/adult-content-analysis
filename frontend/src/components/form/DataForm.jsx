import React, { useState } from "react";
import Container from "react-bootstrap/Container";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import "./DataForm.scss";
import Form from "react-bootstrap/Form";
import Button from "react-bootstrap/Button";
import Alert from "react-bootstrap/Alert";
import { useForm } from "react-hook-form";
import SendDataService from "../../services/sendData.service";
import { ToastMessage } from "../toast/ToastMessage";

export function DataForm() {
  const [url, setUrl] = useState("");
  const [file, setFile] = useState("");
  const [email, setEmail] = useState("");
  const [message, setMessage] = useState("");
  const [selectedOption, setSelectedOption] = useState(false);
  const [isError, setIsError] = useState(false);
  const [isSubmit, setIsSubmit] = useState(false);

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm();

  const isUrlOrFile = () => {
    if (!url && !file) {
      return false;
    } else if (url && file) {
      setUrl("");
      document.getElementById("file").value = "";
      setFile("");
      return false;
    }
    return true;
  };

  const checkFileExtention = (e) => {
    if (!file) {
      return;
    }
    switch (file.type) {
      case "video/mp4":
        return true;
      case "audio/mpeg":
        return true;
      default:
        document.getElementById("file").value = "";
        setFile("");
        return false;
    }
  };

  const handleChange = (e) => {
    e.persist();
    console.log(e.target.value);
    setSelectedOption(e.target.value);
  };

  const onSubmit = (data) => {
    setIsSubmit(true);
    if (selectedOption === "audioframes") {
      SendDataService.sendDataWithUrlForVideoAudioAnalysis(
        data.url,
        data.email,
        data.file
      ).then(
        (res) => {
          setUrl("");
          setEmail("");
          setIsSubmit(false);
          setMessage(res.data);
          setIsError(false);
        },
        (error) => {
          setIsSubmit(false);
          setMessage(error.message);
          setIsError(true);
        }
      );
    } else if (selectedOption === "audio") {
      console.log("audio");
      SendDataService.sendDataWithUrlForAudioAnalysis(
        data.url,
        data.email,
        data.file
      ).then(
        (res) => {
          setUrl("");
          setEmail("");
          setIsSubmit(false);
          setMessage(res.data);
          setIsError(false);
        },
        (error) => {
          setIsSubmit(false);
          setMessage(error.message);
          setIsError(true);
        }
      );
    } else if (selectedOption === "video") {
      SendDataService.sendDataWithUrlForVideoAnalysis(
        data.url,
        data.email,
        data.file
      ).then(
        (res) => {
          setUrl("");
          setEmail("");
          setIsSubmit(false);
          setMessage(res.data);
          setIsError(false);
        },
        (error) => {
          setIsSubmit(false);
          setMessage(error.message);
          setIsError(true);
        }
      );
    }
  };

  return (
    <Container className="form-container">
      {message && (
        <ToastMessage message={message} isError={isError}></ToastMessage>
      )}
      <Form onSubmit={handleSubmit(onSubmit)}>
        <Row>
          <Col>
            <h2>Provide media to analize content</h2>
          </Col>
        </Row>
        <Row>
          <Col>
            <Form.Group className="mb-3" controlId="url">
              <Form.Label>Video URL: </Form.Label>
              <Form.Control
                type="text"
                placeholder="Provide video URL"
                value={url}
                {...register("url", {
                  validate: { isUrlOrFile },
                  onChange: (e) => {
                    e.preventDefault();
                    setUrl(e.target.value);
                  },
                })}
                aria-invalid={errors.url ? "true" : "false"}
              />
            </Form.Group>
          </Col>
        </Row>
        <Row>
          <Col>
            <Form.Group controlId="file" className="mb-3">
              <Form.Label>Video file: </Form.Label>
              <Form.Control
                type="file"
                {...register("file", {
                  validate: { isUrlOrFile, checkFileExtention },
                  onChange: (e) => {
                    e.preventDefault();
                    setFile(e.target.files[0]);
                  },
                })}
                aria-invalid={errors.file ? "true" : "false"}
              />
            </Form.Group>
            {errors.file?.type === "checkFileExtention" && (
              <Alert variant="danger">It's not valid video format!</Alert>
            )}
          </Col>
        </Row>
        <Row>
          <Col>
            <Form.Group className="mb-3" controlId="email">
              <Form.Label>Your email: </Form.Label>
              <Form.Control
                type="email"
                placeholder="Provide your email"
                value={email}
                {...register("email", {
                  required: true,
                  onChange: (e) => {
                    e.preventDefault();
                    setEmail(e.target.value);
                  },
                })}
                aria-invalid={errors.email ? "true" : "false"}
              />
            </Form.Group>
            {errors.email?.type === "required" && (
              <Alert variant="danger">E-mail is required!</Alert>
            )}
          </Col>
        </Row>
        {errors.url?.type === "isUrlOrFile" && (
          <Alert variant="danger">
            URL or video file is required! You can't provide both.
          </Alert>
        )}
        <Col className="mb-3">
          <Row>
            <Row>
              <Form.Check
                value="audioframes"
                className="mb-3"
                inline
                label="Audio & Video Analysis"
                name="group1"
                type="radio"
                id="audioframes"
                defaultChecked
                onChange={handleChange}
              />
            </Row>
            <Row>
              <Form.Check
                value="video"
                className="mb-3"
                inline
                label="Video Analysis"
                name="group1"
                type="radio"
                id="video"
                onChange={handleChange}
              />
            </Row>
            <Row>
              <Form.Check
                value="audio"
                inline
                label="Audio Analysis"
                name="group1"
                type="radio"
                id="audio"
                onChange={handleChange}
              />
            </Row>
          </Row>
        </Col>

        <Button type="submit" disabled={isSubmit}>
          {isSubmit && <span className="spinner-grow spinner-grow-sm"></span>}
          Submit form
        </Button>
      </Form>
    </Container>
  );
}
