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
  const isSending = useRef(false);
  const prevGesture = useRef("");

  const [gesture, setGesture] = useState("neutral");
  const [objects, setObjects] = useState([]);
  const [boxes, setBoxes] = useState([]);
  const [connected, setConnected] = useState(false);
  const [fps, setFps] = useState(0);

  // 🔊 SOUND
  const playSound = () => {
    try {
      const audio = new Audio("/click.mp3");
      audio.volume = 0.3;
      audio.play();
    } catch {
      console.log("sound error");
    }
  };

  // 🎯 gesture change detect
  useEffect(() => {
    if (gesture !== prevGesture.current) {
      playSound();
      prevGesture.current = gesture;
    }
  }, [gesture]);

  useEffect(() => {
    let lastTime = Date.now();

    const interval = setInterval(() => {
      const now = Date.now();
      setFps(Math.round(1000 / (now - lastTime)));
      lastTime = now;

      sendFrame();
    }, 400);

    return () => clearInterval(interval);
  }, []);

  const sendFrame = async () => {
    if (isSending.current) return;
    isSending.current = true;

    try {
      const screenshot = webcamRef.current.getScreenshot();
      if (!screenshot) return;

      const blob = await fetch(screenshot).then((res) => res.blob());

      const formData = new FormData();
      formData.append("file", blob, "frame.jpg");

      const res = await fetch("http://127.0.0.1:8000/predict-image", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      setGesture(data.gesture);
      setObjects(data.objects || []);
      setBoxes(data.objects || []);
      setConnected(true);
    } catch {
      setConnected(false);
    } finally {
      isSending.current = false;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-black via-gray-900 to-gray-800 text-white">
      {/* 🔥 HEADER */}
      <div className="flex justify-between items-center px-8 py-5 border-b border-white/10">
        {/* LEFT */}
        <div>
          <h1 className="text-xl font-bold">🤖 AI Vision Dashboard</h1>
          <p className="text-xs text-gray-400">
            Real-Time Gesture & Object Detection
          </p>
        </div>

        {/* CENTER - TEAM */}
        <div className="text-center">
          <h2 className="text-sm font-semibold text-gray-300">Team Members</h2>
          <p className="text-xs text-gray-400">Ashish Kumar</p>
          <p className="text-xs text-gray-400">Vishal Kumar</p>
          <p className="text-xs text-gray-400">Rohit Kumar</p>
        </div>

        {/* RIGHT */}
        <div className="text-right text-sm">
          <p>FPS: {fps}</p>
          <div className="flex items-center justify-end gap-2">
            <div
              className={`w-2.5 h-2.5 rounded-full ${connected ? "bg-green-400" : "bg-red-500"}`}
            />
            <span>{connected ? "Connected" : "Disconnected"}</span>
          </div>
        </div>
      </div>

      {/* 🔥 MAIN */}
      <div className="flex p-6 gap-6">
        {/* 🎥 CAMERA */}
        <div className="relative w-[70%] aspect-video rounded-2xl overflow-hidden border border-white/10 shadow-xl">
          <Webcam
            ref={webcamRef}
            audio={false}
            screenshotFormat="image/jpeg"
            className="w-full h-full object-cover"
          />

          {/* BOXES */}
          {boxes.map((obj, i) => {
            if (!obj.box) return null;
            const [x1, y1, x2, y2] = obj.box;

            return (
              <div
                key={i}
                className="absolute border-2 border-green-400"
                style={{
                  left: `${x1}px`,
                  top: `${y1}px`,
                  width: `${x2 - x1}px`,
                  height: `${y2 - y1}px`,
                }}
              >
                <span className="bg-green-400 text-black text-xs px-1">
                  {obj.label}
                </span>
              </div>
            );
          })}

          {/* Gesture overlay */}
          <div className="absolute top-4 left-4 bg-black/60 px-3 py-1 rounded-lg text-xs">
            Gesture: <span className="text-green-400">{gesture}</span>
          </div>
        </div>

        {/* SIDE PANEL */}
        <div className="w-[30%] flex flex-col gap-5">
          {/* Gesture */}
          <div className="bg-white/10 rounded-xl p-5 text-center border border-white/10">
            <div className="text-4xl">{emojiMap[gesture]}</div>
            <p className="text-green-400 mt-2">{gesture}</p>
          </div>

          {/* Objects */}
          <div className="bg-white/10 rounded-xl p-5 border border-white/10">
            <p className="text-xs text-gray-400 mb-2">Detected Objects</p>

            {objects.length > 0 ? (
              <div className="flex flex-wrap gap-2">
                {objects.map((obj, i) => (
                  <span
                    key={i}
                    className="px-2 py-1 bg-blue-500/20 border border-blue-400 text-xs rounded-full"
                  >
                    {typeof obj === "string" ? obj : obj.label}
                  </span>
                ))}
              </div>
            ) : (
              <p className="text-xs text-gray-500">No objects</p>
            )}
          </div>

          {/* Stats */}
          <div className="bg-white/10 rounded-xl p-5 border border-white/10 text-xs">
            <p>Latency: ~400ms</p>
            <p>Model: YOLOv8 + MediaPipe</p>
            <p>Status: Active</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
