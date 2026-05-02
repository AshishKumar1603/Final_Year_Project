import { useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";

const emojiMap = {
  thumbs_up: "👍",
  stop: "✋",
  victory: "✌️",
  fist: "👊",
  one: "☝️",
  point_right: "👉",
  high_five: "✋",
  thumbs_down: "👎",
  okay: "👌",
  neutral: "😐",
  none: "😐",
  error: "⚠️",
};

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const isSending = useRef(false);

  const [gesture, setGesture] = useState("neutral");
  const [objects, setObjects] = useState([]);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const interval = setInterval(() => {
      sendFrame();
    }, 300); // 🔥 faster + smoother

    return () => clearInterval(interval);
  }, []);

  const sendFrame = async () => {
    if (isSending.current) return;
    isSending.current = true;

    try {
      const webcam = webcamRef.current;
      if (!webcam) return;

      const screenshot = webcam.getScreenshot();
      if (!screenshot) return;

      const blob = await fetch(screenshot).then((res) => res.blob());

      const formData = new FormData();
      formData.append("file", blob, "frame.jpg");

      const res = await fetch("http://127.0.0.1:8000/predict-image", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("API error");

      const data = await res.json();

      setGesture(data.gesture);
      setObjects(data.objects || []);
      setConnected(true);

      drawBoxes(data.objects || []);
    } catch (err) {
      console.log("API error:", err);
      setConnected(false);
    } finally {
      isSending.current = false;
    }
  };

  // 🔥 DRAW BOUNDING BOXES
  const drawBoxes = (objects) => {
    const canvas = canvasRef.current;
    const video = webcamRef.current?.video;

    if (!canvas || !video) return;

    const ctx = canvas.getContext("2d");

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    objects.forEach((obj) => {
      const [x1, y1, x2, y2] = obj.box;

      ctx.strokeStyle = "#00FF00";
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      ctx.fillStyle = "#00FF00";
      ctx.font = "16px Arial";
      ctx.fillText(obj.label, x1, y1 - 5);
    });
  };

  return (
    <div className="h-screen bg-gradient-to-br from-black via-gray-900 to-gray-800 flex items-center justify-center text-white">
      {" "}
      <div className="flex gap-10 items-center">
        {/* 🎥 Webcam + Canvas */}
        <div className="relative rounded-3xl overflow-hidden border border-white/20 shadow-2xl">
          <Webcam
            ref={webcamRef}
            audio={false}
            screenshotFormat="image/jpeg"
            className="w-[400px] h-[300px] object-cover"
          />

          <canvas
            ref={canvasRef}
            className="absolute top-0 left-0 w-[400px] h-[300px]"
          />
        </div>

        {/* 🤖 UI */}
        <div className="w-[350px] p-8 rounded-3xl bg-white/10 backdrop-blur-xl border border-white/20 shadow-2xl text-center">
          <h1 className="text-2xl font-bold mb-4">🤖 Gesture + Object AI</h1>

          {/* Status */}
          <div className="flex items-center justify-center gap-2 mb-6">
            <div
              className={`w-3 h-3 rounded-full ${connected ? "bg-green-400" : "bg-red-500"} animate-pulse`}
            />
            <span className="text-sm">
              {connected ? "Connected" : "Disconnected"}
            </span>
          </div>

          {/* Gesture */}
          <div className="bg-black/40 rounded-2xl p-6 shadow-inner mb-4">
            <div className="text-6xl mb-3">{emojiMap[gesture] || "❓"}</div>

            <h2 className="text-xl font-bold text-green-400 capitalize">
              {gesture}
            </h2>
          </div>

          {/* Objects */}
          <div className="bg-black/40 rounded-2xl p-4 shadow-inner">
            <h3 className="text-sm text-gray-400 mb-2">Detected Objects</h3>

            {objects.length > 0 ? (
              <div className="flex flex-wrap gap-2 justify-center">
                {objects.map((obj, index) => (
                  <span
                    key={index}
                    className="px-3 py-1 bg-blue-500/20 border border-blue-400 text-blue-300 rounded-full text-xs"
                  >
                    {obj.label}
                  </span>
                ))}
              </div>
            ) : (
              <p className="text-gray-500 text-xs">No objects</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
