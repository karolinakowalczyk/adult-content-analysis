import React, { useState, useEffect } from "react";
import Toast from "react-bootstrap/Toast";

export function ToastMessage({ message, isError }) {
  const [show, setShow] = useState(true);
  useEffect(() => {
    const timeId = setTimeout(() => {
      setShow(false);
    }, 5000);

    return () => {
      clearTimeout(timeId);
    };
  }, []);

  const close = () => setShow(!show);

  return (
    <Toast show={show} onClose={close}>
      <Toast.Header>
        {isError ? (
          <strong className="me-auto">ERROR</strong>
        ) : (
          <strong className="me-auto">SUCCESS</strong>
        )}
      </Toast.Header>
      <Toast.Body>{message}</Toast.Body>
    </Toast>
  );
}
