import React from "react";
import ReactDOM from "react-dom/client";
import { MotionConfig } from "framer-motion";
import "@fontsource-variable/readex-pro";
import App from "./App";
import "./styles.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <MotionConfig reducedMotion="user">
      <App />
    </MotionConfig>
  </React.StrictMode>,
);
