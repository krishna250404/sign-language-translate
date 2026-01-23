import { useRef, useState } from "react";
import Webcam from "react-webcam";

function App() {
  const webcamRef = useRef(null);
  const [imageSrc, setImageSrc] = useState(null);
  const [status, setStatus] = useState("Idle");

  const capture = () => {
    if (!webcamRef.current) return;
    const screenshot = webcamRef.current.getScreenshot();
    setImageSrc(screenshot);
    setStatus("Frame captured (placeholder)");
  };

  return (
    <div style={styles.container}>
      <h1>Webcam ML App (Placeholder)</h1>

      <Webcam
        ref={webcamRef}
        audio={false}
        screenshotFormat="image/jpeg"
        videoConstraints={videoConstraints}
        style={styles.webcam}
      />

      <div style={styles.controls}>
        <button onClick={capture}>Capture Frame</button>
        <button onClick={() => setImageSrc(null)}>Clear</button>
      </div>

      <p>Status: {status}</p>

      {imageSrc && (
        <div style={styles.preview}>
          <h3>Captured Frame</h3>
          <img src={imageSrc} alt="Captured" />
        </div>
      )}
    </div>
  );
}

const videoConstraints = {
  width: 640,
  height: 480,
  facingMode: "user"
};

const styles = {
  container: {
    textAlign: "center",
    padding: "20px"
  },
  webcam: {
    width: "640px",
    borderRadius: "12px",
    marginBottom: "10px"
  },
  controls: {
    display: "flex",
    gap: "10px",
    justifyContent: "center",
    marginBottom: "10px"
  },
  preview: {
    marginTop: "20px"
  }
};

export default App;
