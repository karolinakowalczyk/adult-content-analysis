import React, { useState } from "react";
import Container from "react-bootstrap/Container";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import "./DataForm.scss";
import Form from "react-bootstrap/Form";
import Button from "react-bootstrap/Button";
import Alert from "react-bootstrap/Alert";
import { useForm } from "react-hook-form";

//  TO DO
// POST data

export function DataForm() {
  const [url, setUrl] = useState("");
  const [file, setFile] = useState("");
  const [email, setEmail] = useState("");

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

  const onSubmit = (data) => {
    console.log(data);
  };

  return (
    <Container className="form-container">
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
                //value={file}
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
        <Button type="submit">Submit form</Button>
      </Form>
    </Container>
  );
}
