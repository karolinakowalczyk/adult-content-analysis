import "./App.scss";
import React from "react";
import { DataForm } from "./components/form/DataForm";
import { Header } from "./components/header/Header";

function App() {
  return (
    <div className="main-app">
      <Header />
      <div className="app-container">
        <DataForm></DataForm>
      </div>
    </div>
  );
}

export default App;
