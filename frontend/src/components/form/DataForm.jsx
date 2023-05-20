import React, { useEffect, useState } from "react";
import Container from "react-bootstrap/Container";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import "./DataForm.scss";
import Form from "react-bootstrap/Form";
import Button from "react-bootstrap/Button";
import Alert from "react-bootstrap/Alert";
import { Controller, useForm } from "react-hook-form";
export function DataForm() {
  const [url, setUrl] = useState("");
  const [file, setFile] = useState("");
  const [email, setEmail] = useState("");

  const {
    register,
    handleSubmit,
    watch,
    formState: { errors },
  } = useForm();
  const isUrlOrFile = (url, file) => {
    if (!url && !file) {
      return false;
    }
    return true;
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
                  onChange: (e) => {
                    e.preventDefault();
                    setUrl(e.target.value);
                  },
                })}
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
                value={file}
                {...register("file", {
                  onChange: (e) => {
                    e.preventDefault();
                    setFile(e.target.value);
                  },
                })}
              />
            </Form.Group>
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
        <Button type="submit">Submit form</Button>
      </Form>
    </Container>
  );
}
