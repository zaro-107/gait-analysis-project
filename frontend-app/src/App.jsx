import { useEffect, useMemo, useState } from "react";
import "./App.css";
import GaitDashboard from "./GaitDashboard";

const API_BASE = "/api";

function pretty(obj) {
  try {
    return JSON.stringify(obj, null, 2);
  } catch {
    return String(obj);
  }
}

function Badge({ children }) {
  return (
    <span
      style={{
        display: "inline-block",
        padding: "6px 10px",
        borderRadius: 999,
        border: "1px solid rgba(255,255,255,0.12)",
        background: "rgba(255,255,255,0.06)",
        fontSize: 12,
      }}
    >
      {children}
    </span>
  );
}

function Card({ title, subtitle, children, right }) {
  return (
    <div
      style={{
        background: "rgba(255,255,255,0.06)",
        border: "1px solid rgba(255,255,255,0.10)",
        borderRadius: 16,
        padding: 18,
        boxShadow: "0 10px 25px rgba(0,0,0,0.18)",
      }}
    >
      <div style={{ display: "flex", alignItems: "flex-start", gap: 12 }}>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 18, fontWeight: 700 }}>{title}</div>
          {subtitle ? (
            <div style={{ opacity: 0.8, marginTop: 4, fontSize: 13 }}>{subtitle}</div>
          ) : null}
        </div>
        {right ? <div>{right}</div> : null}
      </div>
      <div style={{ marginTop: 14 }}>{children}</div>
    </div>
  );
}

function ProgressBar({ value }) {
  const pct = Math.max(0, Math.min(1, value ?? 0));
  return (
    <div
      style={{
        height: 10,
        borderRadius: 999,
        background: "rgba(255,255,255,0.08)",
        overflow: "hidden",
        border: "1px solid rgba(255,255,255,0.10)",
      }}
    >
      <div
        style={{
          width: `${pct * 100}%`,
          height: "100%",
          background: "linear-gradient(90deg, #60a5fa, #34d399)",
        }}
      />
    </div>
  );
}

function SectionTitle({ children }) {
  return (
    <div style={{ fontSize: 13, opacity: 0.85, marginBottom: 8, fontWeight: 700 }}>
      {children}
    </div>
  );
}

function KeyValue({ label, value }) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        gap: 12,
        padding: "8px 10px",
        borderRadius: 10,
        background: "rgba(255,255,255,0.04)",
        border: "1px solid rgba(255,255,255,0.08)",
      }}
    >
      <div style={{ opacity: 0.8 }}>{label}</div>
      <div style={{ fontWeight: 700, textAlign: "right" }}>{String(value)}</div>
    </div>
  );
}

async function safeFetchJson(url, options = {}, timeoutMs = 120000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(url, {
      ...options,
      signal: controller.signal,
    });

    const rawText = await res.text();
    let data = null;

    try {
      data = rawText ? JSON.parse(rawText) : null;
    } catch {
      data = { raw: rawText };
    }

    if (!res.ok) {
      const detail =
        data?.detail ||
        data?.reason ||
        data?.message ||
        data?.raw ||
        `HTTP ${res.status} ${res.statusText}`;
      throw new Error(detail);
    }

    return data;
  } catch (e) {
    if (e.name === "AbortError") {
      throw new Error("Request timed out. Video processing took too long.");
    }
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

export default function App() {
  const [sensorFile, setSensorFile] = useState(null);
  const [mediaFile, setMediaFile] = useState(null);

  const [loadingSensor, setLoadingSensor] = useState(false);
  const [loadingMedia, setLoadingMedia] = useState(false);
  const [checkingApi, setCheckingApi] = useState(false);

  const [sensorResult, setSensorResult] = useState(null);
  const [mediaResult, setMediaResult] = useState(null);
  const [healthResult, setHealthResult] = useState(null);

  const [sensorError, setSensorError] = useState("");
  const [mediaError, setMediaError] = useState("");
  const [apiError, setApiError] = useState("");

  const [showDashboard, setShowDashboard] = useState(false);

  const mediaPreviewUrl = useMemo(() => {
    if (!mediaFile) return "";
    return URL.createObjectURL(mediaFile);
  }, [mediaFile]);

  useEffect(() => {
    return () => {
      if (mediaPreviewUrl) {
        URL.revokeObjectURL(mediaPreviewUrl);
      }
    };
  }, [mediaPreviewUrl]);

  async function checkApiHealth() {
    setApiError("");
    setCheckingApi(true);
    try {
      const data = await safeFetchJson(`${API_BASE}/health`, {
        method: "GET",
      }, 15000);
      setHealthResult(data);
    } catch (e) {
      setHealthResult(null);
      setApiError(String(e.message || e));
    } finally {
      setCheckingApi(false);
    }
  }

  async function runSensorPredict() {
    setSensorError("");
    setSensorResult(null);

    if (!sensorFile) {
      setSensorError("Please select a sensor .txt or .csv file.");
      return;
    }

    const fd = new FormData();
    fd.append("file", sensorFile);

    setLoadingSensor(true);
    try {
      const data = await safeFetchJson(`${API_BASE}/predict`, {
        method: "POST",
        body: fd,
      }, 120000);
      setSensorResult(data);
    } catch (e) {
      setSensorError(String(e.message || e));
    } finally {
      setLoadingSensor(false);
    }
  }

  async function runMediaPredict() {
    setMediaError("");
    setMediaResult(null);
    setShowDashboard(false); // Reset dashboard on new upload

    if (!mediaFile) {
      setMediaError("Please select a video or image file.");
      return;
    }

    const fd = new FormData();
    fd.append("file", mediaFile); // Fixed: changed "media" to "file"

    setLoadingMedia(true);
    try {
      const data = await safeFetchJson(`${API_BASE}/predict_media`, {
        method: "POST",
        body: fd,
      }, 180000);
      setMediaResult(data);
    } catch (e) {
      setMediaError(String(e.message || e));
    } finally {
      setLoadingMedia(false);
    }
  }

  function handleDownloadPDF() {
    if (mediaResult?.report_file) {
      window.open(`${API_BASE}/download_report/${mediaResult.report_file}`, '_blank');
    }
  }

  const confidence = sensorResult?.confidence;
  const prediction = sensorResult?.prediction;

  const rich = mediaResult?.gait_features_rich || {};
  const preds = mediaResult?.predictions || {};

  return (
    <div
      style={{
        minHeight: "100vh",
        background:
          "radial-gradient(1200px 600px at 20% 10%, rgba(99,102,241,0.35), transparent), radial-gradient(900px 500px at 90% 20%, rgba(16,185,129,0.25), transparent), radial-gradient(900px 500px at 50% 90%, rgba(236,72,153,0.20), transparent), #0b1020",
        color: "white",
      }}
    >
      <div style={{ maxWidth: 1150, margin: "0 auto", padding: "28px 18px 44px" }}>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 12, alignItems: "center" }}>
          <div style={{ fontSize: 28, fontWeight: 800 }}>GAIT Analyzer</div>
          <Badge>Sensor (.txt/.csv)</Badge>
          <Badge>Video (.mp4/.mov/.avi/.mkv/.webm)</Badge>
          <Badge>Image (.jpg/.png/.jpeg/.webp)</Badge>
          <div style={{ marginLeft: "auto", opacity: 0.85, fontSize: 13 }}>
            API: {API_BASE}
          </div>
        </div>

        <div style={{ marginTop: 18, opacity: 0.85, fontSize: 14, lineHeight: 1.45 }}>
          Upload either a <b>sensor file</b> for activity prediction, or a <b>video/image</b> for
          pose-based gait and posture feature extraction.
        </div>

        <div style={{ marginTop: 16, display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
          <button
            onClick={checkApiHealth}
            disabled={checkingApi}
            style={{
              padding: "10px 14px",
              borderRadius: 12,
              border: "1px solid rgba(255,255,255,0.14)",
              background: "rgba(255,255,255,0.10)",
              color: "white",
              cursor: "pointer",
              fontWeight: 700,
            }}
          >
            {checkingApi ? "Checking API..." : "Check API Health"}
          </button>

          {healthResult ? (
            <>
              <Badge>Backend: OK</Badge>
              <Badge>Device: {healthResult.device}</Badge>
              <Badge>Pose task: {String(healthResult.pose_task_exists)}</Badge>
            </>
          ) : null}

          {apiError ? <div style={{ color: "#ffb4b4" }}>{apiError}</div> : null}
        </div>

        <div
          style={{
            marginTop: 18,
            display: "grid",
            gridTemplateColumns: "repeat(12, 1fr)",
            gap: 14,
          }}
        >
          <div style={{ gridColumn: "span 6" }}>
            <Card
              title="Sensor Prediction"
              subtitle="Upload sensor time-series (.txt/.csv) → CNN embeddings → ExtraTrees → predicted activity"
              right={<Badge>{loadingSensor ? "Running..." : "Ready"}</Badge>}
            >
              <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
                <input
                  type="file"
                  accept=".txt,.csv"
                  onChange={(e) => setSensorFile(e.target.files?.[0] || null)}
                />
                <button
                  onClick={runSensorPredict}
                  disabled={loadingSensor}
                  style={{
                    padding: "10px 14px",
                    borderRadius: 12,
                    border: "1px solid rgba(255,255,255,0.14)",
                    background: "rgba(255,255,255,0.10)",
                    color: "white",
                    cursor: "pointer",
                    fontWeight: 700,
                  }}
                >
                  {loadingSensor ? "Predicting..." : "Predict"}
                </button>
                {sensorFile ? (
                  <div style={{ opacity: 0.85, fontSize: 13 }}>
                    Selected: <b>{sensorFile.name}</b>
                  </div>
                ) : null}
              </div>

              {sensorError ? (
                <div style={{ marginTop: 12, color: "#ffb4b4" }}>{sensorError}</div>
              ) : null}

              {sensorResult ? (
                <div style={{ marginTop: 14 }}>
                  <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
                    <Badge>
                      Prediction: <b>{prediction}</b>
                    </Badge>
                    <Badge>
                      Confidence: <b>{confidence}</b>
                    </Badge>
                    <Badge>
                      Windows: <b>{sensorResult.num_windows}</b>
                    </Badge>
                  </div>

                  <div style={{ marginTop: 10 }}>
                    <div style={{ opacity: 0.85, fontSize: 12, marginBottom: 6 }}>Confidence bar</div>
                    <ProgressBar value={typeof confidence === "number" ? confidence : 0} />
                  </div>

                  <div style={{ marginTop: 12 }}>
                    <SectionTitle>Summary</SectionTitle>
                    <div style={{ display: "grid", gap: 8 }}>
                      <KeyValue label="File" value={sensorResult.filename || "-"} />
                      <KeyValue label="Rows" value={sensorResult.num_rows || "-"} />
                      <KeyValue label="Windows" value={sensorResult.num_windows || "-"} />
                      <KeyValue label="Prediction" value={sensorResult.prediction || "-"} />
                    </div>
                  </div>

                  <div style={{ marginTop: 12 }}>
                    <div style={{ opacity: 0.85, fontSize: 12, marginBottom: 6 }}>Raw response</div>
                    <pre
                      style={{
                        background: "rgba(0,0,0,0.30)",
                        border: "1px solid rgba(255,255,255,0.10)",
                        borderRadius: 14,
                        padding: 12,
                        overflowX: "auto",
                        fontSize: 12,
                      }}
                    >
                      {pretty(sensorResult)}
                    </pre>
                  </div>
                </div>
              ) : null}
            </Card>
          </div>

          <div style={{ gridColumn: "span 6" }}>
            <Card
              title="Video / Image Analysis"
              subtitle="Upload a walking video or a full-body image → MediaPipe Pose → gait/posture features"
              right={<Badge>{loadingMedia ? "Running..." : "Ready"}</Badge>}
            >
              <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
                <input
                  type="file"
                  accept="video/*,image/*,.webm"
                  onChange={(e) => setMediaFile(e.target.files?.[0] || null)}
                />
                <button
                  onClick={runMediaPredict}
                  disabled={loadingMedia}
                  style={{
                    padding: "10px 14px",
                    borderRadius: 12,
                    border: "1px solid rgba(255,255,255,0.14)",
                    background: "rgba(255,255,255,0.10)",
                    color: "white",
                    cursor: "pointer",
                    fontWeight: 700,
                  }}
                >
                  {loadingMedia ? "Analyzing..." : "Analyze"}
                </button>
                {mediaFile ? (
                  <div style={{ opacity: 0.85, fontSize: 13 }}>
                    Selected: <b>{mediaFile.name}</b>
                  </div>
                ) : null}
              </div>

              {mediaPreviewUrl ? (
                <div style={{ marginTop: 12 }}>
                  {mediaFile?.type?.startsWith("image/") ? (
                    <img
                      src={mediaPreviewUrl}
                      alt="preview"
                      style={{
                        width: "100%",
                        maxHeight: 260,
                        objectFit: "contain",
                        borderRadius: 14,
                        border: "1px solid rgba(255,255,255,0.10)",
                        background: "rgba(0,0,0,0.20)",
                      }}
                    />
                  ) : (
                    <video
                      src={mediaPreviewUrl}
                      controls
                      style={{
                        width: "100%",
                        maxHeight: 260,
                        borderRadius: 14,
                        border: "1px solid rgba(255,255,255,0.10)",
                        background: "black",
                      }}
                    />
                  )}
                </div>
              ) : null}

              {mediaError ? (
                <div style={{ marginTop: 12, color: "#ffb4b4", whiteSpace: "pre-wrap" }}>
                  {mediaError}
                </div>
              ) : null}

              {mediaResult ? (
                <div style={{ marginTop: 14 }}>
                  <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
                    <Badge>
                      Type: <b>{mediaResult.type || "-"}</b>
                    </Badge>
                    <Badge>
                      File: <b>{mediaResult.filename || "-"}</b>
                    </Badge>
                    {mediaResult?.avg_cadence != null ? (
                      <Badge>
                        Avg cadence: <b>{Number(mediaResult.avg_cadence).toFixed(2)}</b>
                      </Badge>
                    ) : null}
                    {mediaResult?.symmetry_index != null ? (
                      <Badge>
                        Symmetry index: <b>{Number(mediaResult.symmetry_index).toFixed(3)}</b>
                      </Badge>
                    ) : null}
                  </div>

                  {mediaResult.type === "video" ? (
                    <>
                      <div style={{ marginTop: 12 }}>
                        <SectionTitle>Basic gait features</SectionTitle>
                        <div style={{ display: "grid", gap: 8 }}>
                          <KeyValue label="Frames used" value={mediaResult?.frames_used ?? "-"} />
                          <KeyValue label="Effective FPS" value={mediaResult?.fps_effective != null ? Number(mediaResult.fps_effective).toFixed(2) : "-"} />
                          <KeyValue
                            label="Average cadence (spm)"
                            value={mediaResult?.avg_cadence != null ? Number(mediaResult.avg_cadence).toFixed(2) : "-"}
                          />
                          <KeyValue
                            label="Symmetry index"
                            value={mediaResult?.symmetry_index != null ? Number(mediaResult.symmetry_index).toFixed(3) : "-"}
                          />
                        </div>
                      </div>

                      <div style={{ marginTop: 12 }}>
                        <SectionTitle>Predictions</SectionTitle>
                        <div style={{ display: "grid", gap: 8 }}>
                          <KeyValue
                            label="Rule-based"
                            value={preds?.rule_based?.label || preds?.rule_based?.status || "-"}
                          />
                          <KeyValue
                            label="Normal/Abnormal ML"
                            value={preds?.normal_abnormal_ml?.label || preds?.normal_abnormal_ml?.status || "-"}
                          />
                          <KeyValue
                            label="Classical gait model"
                            value={preds?.video_gait_classical?.label || preds?.video_gait_classical?.status || "-"}
                          />
                          <KeyValue
                            label="Video LSTM gait type"
                            value={preds?.video_lstm_gait_type?.gait_type || preds?.video_lstm_gait_type?.status || "-"}
                          />
                        </div>
                      </div>

                      {rich?.status === "failed" ? (
                        <div style={{ marginTop: 12, color: "#ffb4b4" }}>
                          Rich feature extraction failed: {rich.reason}
                        </div>
                      ) : null}

                      {/* --- ACTION BAR FOR REPORTS --- */}
                      <div style={{ 
                        marginTop: 20, 
                        padding: 16, 
                        background: "rgba(255,255,255,0.05)", 
                        borderRadius: 12,
                        display: "flex",
                        gap: 12,
                        flexWrap: "wrap"
                      }}>
                        {/* Button 1: Download PDF */}
                        {mediaResult.report_file && (
                          <button
                            onClick={handleDownloadPDF}
                            style={{
                              flex: 1,
                              padding: "12px",
                              borderRadius: "8px",
                              backgroundColor: "#3b82f6",
                              color: "white",
                              fontWeight: "bold",
                              border: "none",
                              cursor: "pointer",
                            }}
                          >
                            📄 Download PDF Report
                          </button>
                        )}

                        {/* Button 2: Toggle Interactive Dashboard */}
                        {rich?.series && (
                          <button
                            onClick={() => setShowDashboard(!showDashboard)}
                            style={{
                              flex: 1,
                              padding: "12px",
                              borderRadius: "8px",
                              backgroundColor: showDashboard ? "#ef4444" : "#10b981",
                              color: "white",
                              fontWeight: "bold",
                              border: "none",
                              cursor: "pointer",
                            }}
                          >
                            {showDashboard ? "✖ Close Interactive Report" : "📊 View Interactive Report"}
                          </button>
                        )}
                      </div>

                      {/* --- RENDER THE DASHBOARD IF TOGGLED --- */}
                      {showDashboard && rich?.series ? (
                        <GaitDashboard richFeatures={rich} />
                      ) : null}
                    </>
                  ) : null}

                  {mediaResult.type === "image" ? (
                    <div style={{ marginTop: 12 }}>
                      <SectionTitle>Posture summary</SectionTitle>
                      <div style={{ display: "grid", gap: 8 }}>
                        <KeyValue
                          label="Pose detected"
                          value={String(mediaResult.pose_detected)}
                        />
                        <KeyValue
                          label="Left knee angle"
                          value={mediaResult?.posture_features?.left_knee_angle_deg ?? "-"}
                        />
                        <KeyValue
                          label="Right knee angle"
                          value={mediaResult?.posture_features?.right_knee_angle_deg ?? "-"}
                        />
                        <KeyValue
                          label="Left hip angle"
                          value={mediaResult?.posture_features?.left_hip_angle_deg ?? "-"}
                        />
                        <KeyValue
                          label="Right hip angle"
                          value={mediaResult?.posture_features?.right_hip_angle_deg ?? "-"}
                        />
                      </div>
                    </div>
                  ) : null}

                  <div style={{ marginTop: 12 }}>
                    <div style={{ opacity: 0.85, fontSize: 12, marginBottom: 6 }}>Raw response</div>
                    <pre
                      style={{
                        background: "rgba(0,0,0,0.30)",
                        border: "1px solid rgba(255,255,255,0.10)",
                        borderRadius: 14,
                        padding: 12,
                        overflowX: "auto",
                        fontSize: 12,
                        whiteSpace: "pre-wrap",
                        wordBreak: "break-word",
                      }}
                    >
                      {pretty(mediaResult)}
                    </pre>
                  </div>
                </div>
              ) : null}
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
