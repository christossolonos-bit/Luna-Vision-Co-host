import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { VRMLoaderPlugin, VRMUtils } from "@pixiv/three-vrm";
import {
  VRMAnimationLoaderPlugin,
  createVRMAnimationClip,
} from "@pixiv/three-vrm-animation";

const isObsOverlay =
  window.location.pathname.startsWith("/obs") ||
  new URLSearchParams(window.location.search).get("obs") === "1";

const lunaChannel = new BroadcastChannel("luna-cohost");

const obsSync = {
  overlayActive: false,
};

if (isObsOverlay) {
  document.body.classList.add("obs-overlay");
}

const state = {
  busy: false,
  watchTimer: null,
  screenCaptureEnabled: false,
  mediaRecorder: null,
  micStream: null,
  recordedChunks: [],
  listenEnabled: false,
  vadPaused: false,
  speechActive: false,
  transcribing: false,
};

const els = {
  canvas: document.getElementById("vrm-canvas"),
  status: document.getElementById("status-pill"),
  settingsBtn: document.getElementById("settings-btn"),
  settingsPanel: document.getElementById("settings-panel"),
  styleSelect: document.getElementById("style-select"),
  voiceMood: document.getElementById("voice-mood"),
  useScreen: document.getElementById("use-screen"),
  screenCaptureSetting: document.getElementById("screen-capture-setting"),
  speakReplies: document.getElementById("speak-replies"),
  watchEnabled: document.getElementById("watch-enabled"),
  watchInterval: document.getElementById("watch-interval"),
  clearMemory: document.getElementById("clear-memory"),
  captureSource: document.getElementById("capture-source"),
  captureTrigger: document.getElementById("capture-trigger"),
  captureMenu: document.getElementById("capture-menu"),
  captureSelect: document.getElementById("capture-select"),
  screenCaptureToggle: document.getElementById("screen-capture-toggle"),
  analyzeBtn: document.getElementById("analyze-btn"),
  micBtn: document.getElementById("mic-btn"),
  micStatus: document.getElementById("mic-status"),
  chatLog: document.getElementById("chat-log"),
  chatForm: document.getElementById("chat-form"),
  chatInput: document.getElementById("chat-input"),
};

const voiceListen = {
  stream: null,
  audioContext: null,
  analyser: null,
  mimeType: "",
  vadTimer: null,
  calibrationStart: 0,
  noiseSamples: [],
  noiseFloor: 0.008,
  silenceStartedAt: null,
  speechStartedAt: 0,
};

const VAD = {
  intervalMs: 60,
  calibrationMs: 900,
  silenceMs: 900,
  minSpeechMs: 500,
  maxSpeechMs: 18000,
  minLevel: 0.012,
  speechMultiplier: 2.6,
};

const vrmState = {
  vrm: null,
  mixer: null,
  clock: new THREE.Clock(),
  mouthKeys: ["aa", "ih", "ou", "A", "I", "U", "O"],
  audioContext: null,
  analyser: null,
  lipSyncValue: 0,
  modelOffsetY: 0,
  dragActive: false,
  dragLastY: 0,
};

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || response.statusText);
  }
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return response.json();
  }
  return response;
}

function appendMessage(role, content) {
  const node = document.createElement("div");
  node.className = `msg ${role}`;
  const label = role === "assistant" ? "Luna" : role === "user" ? "You" : "System";
  node.innerHTML = `<span class="role">${label}</span>${escapeHtml(content)}`;
  els.chatLog.appendChild(node);
  els.chatLog.scrollTop = els.chatLog.scrollHeight;
}

function showCaptureFeedback(data) {
  if (!data?.capture_preview_b64) return;

  const node = document.createElement("div");
  node.className = "msg system capture-preview";
  let text = `Luna saw: ${data.capture_label || "screen"}`;
  if (data.capture_warning) {
    text +=
      " — capture looks black or unreadable. Try borderless windowed mode or select the monitor showing the game.";
  }
  node.innerHTML = `<span class="role">Capture</span>${escapeHtml(text)}<img alt="Capture preview" src="data:image/jpeg;base64,${data.capture_preview_b64}" />`;
  els.chatLog.appendChild(node);
  els.chatLog.scrollTop = els.chatLog.scrollHeight;
}

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function setStatus(text, ok = true) {
  els.status.textContent = text;
  els.status.classList.toggle("ok", ok);
  els.status.classList.toggle("error", !ok);
}

function statusWithCapture(baseMessage, ok = true) {
  const capture = state.screenCaptureEnabled ? "Screen ON" : "Screen OFF";
  const sep = baseMessage.includes("|") ? " | " : " — ";
  return `${baseMessage}${sep}${capture}`;
}

function payloadBase() {
  return {
    capture_source: els.captureSource.value,
    style: els.styleSelect.value,
    speak: els.speakReplies.checked,
  };
}

async function refreshStatus() {
  try {
    const data = await api("/api/status");
    if (typeof data.screen_capture_enabled === "boolean") {
      state.screenCaptureEnabled = data.screen_capture_enabled;
      updateScreenCaptureUi(false);
    }
    setStatus(statusWithCapture(data.message, data.ok), data.ok);
  } catch (error) {
    setStatus(`Offline — ${error.message}`, false);
  }
}

async function loadCaptureSources() {
  const sources = await api("/api/capture-sources");
  const previous = els.captureSource.value;
  els.captureMenu.innerHTML = "";

  for (const source of sources) {
    const item = document.createElement("li");
    item.className = "custom-select-option";
    item.dataset.value = source.id;
    item.textContent = source.label;
    item.setAttribute("role", "option");
    item.addEventListener("click", () => selectCaptureSource(source.id, source.label));
    els.captureMenu.appendChild(item);
  }

  if (sources.length === 0) {
    selectCaptureSource("monitor:1", "Screen 1");
    return;
  }

  const stillValid = sources.find((item) => item.id === previous);
  const preferred = stillValid || sources.find((item) => item.id === "monitor:1") || sources[0];
  selectCaptureSource(preferred.id, preferred.label);
}

async function selectCaptureSource(id, label) {
  els.captureSource.value = id;
  els.captureTrigger.textContent = label;
  closeCaptureMenu();
  for (const option of els.captureMenu.querySelectorAll(".custom-select-option")) {
    option.classList.toggle("selected", option.dataset.value === id);
  }
  if (!state.screenCaptureEnabled) return;
  try {
    await api(`/api/preview?source=${encodeURIComponent(id)}`);
  } catch {
    // Preview sync is best-effort; chat still passes capture_source.
  }
}

function updateScreenCaptureUi(notify = true) {
  const on = state.screenCaptureEnabled;
  if (els.screenCaptureToggle) {
    els.screenCaptureToggle.textContent = on ? "Screen: ON" : "Screen: OFF";
    els.screenCaptureToggle.classList.toggle("active", on);
    els.screenCaptureToggle.title = on
      ? "Turn screen capture off (saves GPU/CPU with VSeeFace open)"
      : "Turn screen capture on so Luna can see your screen";
  }
  if (els.screenCaptureSetting) {
    els.screenCaptureSetting.checked = on;
  }
  if (els.captureSelect) {
    els.captureSelect.classList.toggle("disabled", !on);
  }
  if (els.captureTrigger) {
    els.captureTrigger.disabled = !on;
  }
  if (els.analyzeBtn) {
    els.analyzeBtn.disabled = !on;
  }
  if (els.useScreen) {
    els.useScreen.disabled = !on;
    if (!on) {
      els.useScreen.checked = false;
    }
  }
  if (els.watchEnabled) {
    if (!on) {
      els.watchEnabled.checked = false;
      els.watchEnabled.disabled = true;
    } else {
      els.watchEnabled.disabled = false;
    }
  }
  syncWatchTimer();
  if (notify && !isObsOverlay) {
    appendMessage(
      "system",
      on
        ? "Screen capture ON — Luna can see your screen (uses GPU/CPU)."
        : "Screen capture OFF — vision paused to save performance.",
    );
  }
}

async function setScreenCapture(enabled, { notify = true } = {}) {
  const previous = state.screenCaptureEnabled;
  state.screenCaptureEnabled = enabled;
  updateScreenCaptureUi(false);
  if (els.screenCaptureToggle) {
    els.screenCaptureToggle.disabled = true;
  }

  try {
    const data = await api("/api/screen-capture", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled }),
    });
    state.screenCaptureEnabled = Boolean(data.enabled);
    localStorage.setItem("luna-screen-capture", state.screenCaptureEnabled ? "1" : "0");
    updateScreenCaptureUi(notify);
    await refreshStatus();
  } catch (error) {
    state.screenCaptureEnabled = previous;
    updateScreenCaptureUi(false);
    appendMessage("system", `Screen capture error: ${error.message}`);
    throw error;
  } finally {
    if (els.screenCaptureToggle) {
      els.screenCaptureToggle.disabled = false;
    }
  }
}

async function toggleScreenCapture() {
  await setScreenCapture(!state.screenCaptureEnabled);
}

function openCaptureMenu() {
  els.captureMenu.classList.remove("hidden");
  els.captureTrigger.setAttribute("aria-expanded", "true");
}

function closeCaptureMenu() {
  els.captureMenu.classList.add("hidden");
  els.captureTrigger.setAttribute("aria-expanded", "false");
}

function setupCaptureSelect() {
  els.captureTrigger.addEventListener("click", async () => {
    if (els.captureMenu.classList.contains("hidden")) {
      await loadCaptureSources();
      openCaptureMenu();
    } else {
      closeCaptureMenu();
    }
  });

  document.addEventListener("click", (event) => {
    if (!els.captureSelect.contains(event.target)) {
      closeCaptureMenu();
    }
  });
}

function audioMimeFromBytes(bytes, audioFormat = null) {
  if (audioFormat === "wav") return "audio/wav";
  if (audioFormat === "mpeg" || audioFormat === "mp3") return "audio/mpeg";
  if (
    bytes.length >= 4 &&
    bytes[0] === 0x52 &&
    bytes[1] === 0x49 &&
    bytes[2] === 0x46 &&
    bytes[3] === 0x46
  ) {
    return "audio/wav";
  }
  return "audio/mpeg";
}

async function playAssistantAudio(audioB64, audioFormat = null) {
  if (!audioB64) return;

  const binary = atob(audioB64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  const blob = new Blob([bytes], { type: audioMimeFromBytes(bytes, audioFormat) });
  const url = URL.createObjectURL(blob);

  if (!vrmState.audioContext) {
    vrmState.audioContext = new AudioContext();
  }
  if (vrmState.audioContext.state === "suspended") {
    await vrmState.audioContext.resume();
  }

  const audio = new Audio(url);
  audio.crossOrigin = "anonymous";
  const source = vrmState.audioContext.createMediaElementSource(audio);
  vrmState.analyser = vrmState.audioContext.createAnalyser();
  vrmState.analyser.fftSize = 256;
  source.connect(vrmState.analyser);
  vrmState.analyser.connect(vrmState.audioContext.destination);

  await audio.play();
  await new Promise((resolve) => {
    audio.onended = () => {
      URL.revokeObjectURL(url);
      vrmState.analyser = null;
      resolve();
    };
  });
}

function isObsOverlayActive() {
  return obsSync.overlayActive;
}

async function refreshObsActive() {
  try {
    const data = await api("/api/obs/active");
    obsSync.overlayActive = Boolean(data.active);
  } catch {
    obsSync.overlayActive = false;
  }
}

async function routeAssistantAudio(audioB64, audioFormat = null) {
  if (!audioB64) return;

  state.vadPaused = true;
  try {
    if (isObsOverlay) {
      await playAssistantAudio(audioB64, audioFormat);
      return;
    }

    await refreshObsActive();
    if (isObsOverlayActive()) {
      await api("/api/obs/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ audio_b64: audioB64 }),
      });
      return;
    }

    await playAssistantAudio(audioB64, audioFormat);
  } finally {
    state.vadPaused = false;
  }
}

async function sendChat(message, { systemLabel = null } = {}) {
  if (!message.trim() || state.busy) return;
  state.busy = true;
  state.vadPaused = true;
  appendMessage("user", systemLabel || message);
  els.chatInput.disabled = true;

  try {
    const body = {
      message: message.trim(),
      use_screen: els.useScreen.checked,
      ...payloadBase(),
    };
    const data = await api("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (body.use_screen) {
      showCaptureFeedback(data);
    }
    appendMessage("assistant", data.reply);
    await routeAssistantAudio(data.audio_b64, data.audio_format);
  } catch (error) {
    appendMessage("system", error.message);
  } finally {
    state.busy = false;
    state.vadPaused = false;
    els.chatInput.disabled = false;
    els.chatInput.focus();
  }
}

async function analyzeScreen() {
  if (state.busy) return;
  state.busy = true;
  appendMessage("system", "Analyzing screen…");

  try {
    const data = await api("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payloadBase()),
    });
    showCaptureFeedback(data);
    appendMessage("assistant", data.reply);
    await routeAssistantAudio(data.audio_b64, data.audio_format);
  } catch (error) {
    appendMessage("system", error.message);
  } finally {
    state.busy = false;
  }
}

async function watchTick() {
  if (state.busy || !els.watchEnabled.checked) return;
  state.busy = true;
  try {
    const data = await api("/api/watch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payloadBase()),
    });
    if (data.skipped || !data.reply) return;
    showCaptureFeedback(data);
    appendMessage("assistant", `[Watch ${data.timestamp}] ${data.reply}`);
    await routeAssistantAudio(data.audio_b64, data.audio_format);
  } catch (error) {
    appendMessage("system", error.message);
  } finally {
    state.busy = false;
  }
}

function syncWatchTimer() {
  if (state.watchTimer) {
    clearInterval(state.watchTimer);
    state.watchTimer = null;
  }
  if (!els.watchEnabled.checked || !state.screenCaptureEnabled) return;
  const seconds = Math.max(5, Number(els.watchInterval.value) || 8);
  state.watchTimer = setInterval(watchTick, seconds * 1000);
}

function updateListenButton() {
  if (state.listenEnabled) {
    els.micBtn.textContent = state.speechActive ? "🎙 Speaking…" : "🎤 Listen: ON";
    els.micBtn.title = "Turn voice listening off";
    els.micBtn.classList.add("listening");
    if (state.speechActive) {
      els.micBtn.classList.add("recording");
    } else {
      els.micBtn.classList.remove("recording");
    }
  } else {
    els.micBtn.textContent = "🎤 Listen: OFF";
    els.micBtn.title = "Turn voice listening on";
    els.micBtn.classList.remove("listening", "recording");
  }
}

function getMicLevel() {
  if (!voiceListen.analyser) return 0;
  const data = new Uint8Array(voiceListen.analyser.fftSize);
  voiceListen.analyser.getByteTimeDomainData(data);
  let sum = 0;
  for (let i = 0; i < data.length; i += 1) {
    const sample = (data[i] - 128) / 128;
    sum += sample * sample;
  }
  return Math.sqrt(sum / data.length);
}

function getSpeechThreshold() {
  return Math.max(VAD.minLevel, voiceListen.noiseFloor * VAD.speechMultiplier);
}

function beginSpeechCapture() {
  if (state.speechActive || state.transcribing || !voiceListen.stream) return;

  state.speechActive = true;
  state.recordedChunks = [];
  voiceListen.speechStartedAt = Date.now();
  voiceListen.silenceStartedAt = null;

  state.mediaRecorder = voiceListen.mimeType
    ? new MediaRecorder(voiceListen.stream, { mimeType: voiceListen.mimeType })
    : new MediaRecorder(voiceListen.stream);
  state.mediaRecorder.ondataavailable = (event) => {
    if (event.data.size > 0) state.recordedChunks.push(event.data);
  };
  state.mediaRecorder.start();
  els.micStatus.textContent = "Hearing you…";
  updateListenButton();
}

async function finishSpeechCapture() {
  if (!state.speechActive || !state.mediaRecorder) return;

  const duration = Date.now() - voiceListen.speechStartedAt;
  state.speechActive = false;
  voiceListen.silenceStartedAt = null;
  updateListenButton();

  if (duration < VAD.minSpeechMs) {
    if (state.mediaRecorder.state === "recording") {
      state.mediaRecorder.stop();
    }
    state.mediaRecorder = null;
    if (state.listenEnabled) {
      els.micStatus.textContent = "Listening for speech…";
    }
    return;
  }

  const recorder = state.mediaRecorder;
  await new Promise((resolve) => {
    recorder.onstop = resolve;
    recorder.stop();
  });
  state.mediaRecorder = null;

  const blob = new Blob(state.recordedChunks, {
    type: voiceListen.mimeType || "audio/webm",
  });
  state.recordedChunks = [];

  if (blob.size < 4096) {
    if (state.listenEnabled) {
      els.micStatus.textContent = "Listening for speech…";
    }
    return;
  }

  state.transcribing = true;
  els.micStatus.textContent = "Transcribing…";

  const formData = new FormData();
  formData.append("file", blob, "speech.webm");

  try {
    const data = await api("/api/stt", { method: "POST", body: formData });
    if (data.text) {
      els.micStatus.textContent = "Heard you.";
      await sendChat(data.text);
    } else if (state.listenEnabled) {
      els.micStatus.textContent = "Listening for speech…";
    }
  } catch (error) {
    els.micStatus.textContent = "Speech error.";
    appendMessage("system", error.message);
  } finally {
    state.transcribing = false;
    if (state.listenEnabled && !state.speechActive) {
      voiceListen.calibrationStart = Date.now();
      voiceListen.noiseSamples = [];
      els.micStatus.textContent = "Listening for speech…";
    }
  }
}

function tickVoiceActivity() {
  if (!state.listenEnabled || state.vadPaused || state.transcribing) return;

  const level = getMicLevel();
  const now = Date.now();

  if (now - voiceListen.calibrationStart < VAD.calibrationMs) {
    voiceListen.noiseSamples.push(level);
    const average =
      voiceListen.noiseSamples.reduce((sum, value) => sum + value, 0) /
      voiceListen.noiseSamples.length;
    voiceListen.noiseFloor = Math.max(0.004, average * 1.15);
    return;
  }

  const threshold = getSpeechThreshold();
  const speaking = level > threshold;

  if (speaking && !state.speechActive) {
    beginSpeechCapture();
    return;
  }

  if (!state.speechActive) return;

  if (speaking) {
    voiceListen.silenceStartedAt = null;
    if (now - voiceListen.speechStartedAt > VAD.maxSpeechMs) {
      finishSpeechCapture();
    }
    return;
  }

  if (!voiceListen.silenceStartedAt) {
    voiceListen.silenceStartedAt = now;
  } else if (now - voiceListen.silenceStartedAt >= VAD.silenceMs) {
    finishSpeechCapture();
  }
}

async function startVoiceListen() {
  if (state.listenEnabled) return;

  try {
    voiceListen.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
  } catch (error) {
    els.micStatus.textContent = "Mic blocked.";
    appendMessage("system", `Microphone error: ${error.message}`);
    return;
  }

  voiceListen.mimeType = MediaRecorder.isTypeSupported("audio/webm") ? "audio/webm" : "";
  voiceListen.audioContext = new AudioContext();
  const source = voiceListen.audioContext.createMediaStreamSource(voiceListen.stream);
  voiceListen.analyser = voiceListen.audioContext.createAnalyser();
  voiceListen.analyser.fftSize = 2048;
  source.connect(voiceListen.analyser);

  voiceListen.calibrationStart = Date.now();
  voiceListen.noiseSamples = [];
  voiceListen.noiseFloor = 0.008;
  voiceListen.silenceStartedAt = null;
  state.speechActive = false;

  state.listenEnabled = true;
  els.micStatus.textContent = "Calibrating noise…";
  updateListenButton();

  voiceListen.vadTimer = window.setInterval(tickVoiceActivity, VAD.intervalMs);
  window.setTimeout(() => {
    if (state.listenEnabled && !state.speechActive && !state.transcribing) {
      els.micStatus.textContent = "Listening for speech…";
    }
  }, VAD.calibrationMs + 100);
}

function stopVoiceListen() {
  if (!state.listenEnabled) return;

  state.listenEnabled = false;
  state.speechActive = false;
  state.transcribing = false;

  if (voiceListen.vadTimer) {
    clearInterval(voiceListen.vadTimer);
    voiceListen.vadTimer = null;
  }

  if (state.mediaRecorder?.state === "recording") {
    state.mediaRecorder.stop();
  }
  state.mediaRecorder = null;
  state.recordedChunks = [];

  if (voiceListen.stream) {
    for (const track of voiceListen.stream.getTracks()) {
      track.stop();
    }
    voiceListen.stream = null;
  }

  if (voiceListen.audioContext) {
    voiceListen.audioContext.close().catch(() => {});
    voiceListen.audioContext = null;
  }
  voiceListen.analyser = null;

  els.micStatus.textContent = "Voice listen off.";
  updateListenButton();
}

function setupMicControls() {
  updateListenButton();
  els.micBtn.addEventListener("click", async () => {
    if (state.listenEnabled) {
      stopVoiceListen();
    } else {
      await startVoiceListen();
    }
  });
}

function setupUi() {
  setupCaptureSelect();
  els.settingsBtn.addEventListener("click", () => {
    els.settingsPanel.classList.toggle("hidden");
  });

  if (els.voiceMood) {
    els.voiceMood.addEventListener("change", async () => {
      try {
        await api("/api/voice/mood", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ mood: els.voiceMood.value }),
        });
      } catch (error) {
        appendMessage("system", `Voice mood error: ${error.message}`);
      }
    });
  }

  els.chatForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const text = els.chatInput.value;
    els.chatInput.value = "";
    sendChat(text);
  });

  els.analyzeBtn?.addEventListener("click", analyzeScreen);
  els.screenCaptureToggle?.addEventListener("click", () => {
    toggleScreenCapture().catch(() => {});
  });
  els.screenCaptureSetting?.addEventListener("change", () => {
    const want = els.screenCaptureSetting.checked;
    if (want === state.screenCaptureEnabled) return;
    setScreenCapture(want).catch(() => {});
  });
  els.clearMemory.addEventListener("click", async () => {
    await api("/api/reset", { method: "POST" });
    els.chatLog.innerHTML = "";
    appendMessage("system", "Memory cleared.");
  });

  els.watchEnabled.addEventListener("change", syncWatchTimer);
  els.watchInterval.addEventListener("change", syncWatchTimer);
}

function updateLipSync() {
  if (!vrmState.vrm?.expressionManager) return;

  let target = 0;
  if (vrmState.analyser) {
    const buffer = new Uint8Array(vrmState.analyser.frequencyBinCount);
    vrmState.analyser.getByteFrequencyData(buffer);
    const avg = buffer.reduce((sum, value) => sum + value, 0) / buffer.length;
    target = Math.min(1, avg / 90);
  }

  vrmState.lipSyncValue += (target - vrmState.lipSyncValue) * 0.35;
  for (const key of vrmState.mouthKeys) {
    if (vrmState.vrm.expressionManager.getExpression(key) != null) {
      vrmState.vrm.expressionManager.setValue(key, vrmState.lipSyncValue);
    }
  }
  vrmState.vrm.expressionManager.update();
}

const OBS_RENDER_FPS = 60;

// Browsers pause requestAnimationFrame when a window is minimized. A timer loop plus
// inaudible audio keeps the OBS overlay animating for window capture / background use.
function startObsBackgroundKeepAlive() {
  if (!isObsOverlay) return;

  let ctx = null;
  try {
    ctx = new AudioContext();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.frequency.value = 20000;
    gain.gain.value = 0.001;
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start(0);
  } catch {
    return;
  }

  const resume = () => {
    void ctx?.resume();
  };
  resume();
  document.addEventListener("visibilitychange", resume);
  window.addEventListener("focus", resume);
}

function startVrmRenderLoop(renderer, tick) {
  if (!isObsOverlay) {
    renderer.setAnimationLoop(tick);
    return;
  }

  startObsBackgroundKeepAlive();

  const intervalMs = 1000 / OBS_RENDER_FPS;
  setInterval(tick, intervalMs);

  document.addEventListener("visibilitychange", () => {
    vrmState.clock.getDelta();
  });
}

function setupVrmVerticalDrag(canvas, vrm, controls) {
  canvas.addEventListener("contextmenu", (event) => event.preventDefault());

  const applyDrag = (deltaY) => {
    const model = vrmState.vrm;
    if (!model) return;
    vrmState.modelOffsetY -= deltaY * 0.003;
    model.scene.position.y = vrmState.modelOffsetY;
    lunaChannel.postMessage({ type: "vrm-y", offsetY: vrmState.modelOffsetY });
  };

  canvas.addEventListener(
    "pointerdown",
    (event) => {
      if (event.button !== 2) return;
      event.preventDefault();
      event.stopPropagation();
      vrmState.dragActive = true;
      vrmState.dragLastY = event.clientY;
      controls.enabled = false;
      canvas.setPointerCapture(event.pointerId);
      canvas.classList.add("vrm-dragging");
    },
    { capture: true },
  );

  canvas.addEventListener(
    "pointermove",
    (event) => {
      if (!vrmState.dragActive) return;
      event.preventDefault();
      event.stopPropagation();
      const deltaY = event.clientY - vrmState.dragLastY;
      applyDrag(deltaY);
      vrmState.dragLastY = event.clientY;
    },
    { capture: true },
  );

  const stopDrag = (event) => {
    if (!vrmState.dragActive) return;
    if (event.type === "pointerup" && event.button !== 2) return;
    event.stopPropagation();
    vrmState.dragActive = false;
    controls.enabled = true;
    if (canvas.hasPointerCapture(event.pointerId)) {
      canvas.releasePointerCapture(event.pointerId);
    }
    canvas.classList.remove("vrm-dragging");
  };

  canvas.addEventListener("pointerup", stopDrag, { capture: true });
  canvas.addEventListener("pointercancel", stopDrag, { capture: true });
}

async function initVrmViewer() {
  const renderer = new THREE.WebGLRenderer({
    canvas: els.canvas,
    antialias: true,
    alpha: true,
  });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  if (isObsOverlay) {
    renderer.setClearColor(0x00ff00, 1);
  }

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 20);
  camera.position.set(0, 1.25, 2.2);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 1.05, 0);
  controls.enablePan = false;
  controls.minDistance = 1.2;
  controls.maxDistance = 4.5;
  controls.mouseButtons = {
    LEFT: THREE.MOUSE.ROTATE,
    MIDDLE: THREE.MOUSE.DOLLY,
    RIGHT: null,
  };
  controls.update();

  setupVrmVerticalDrag(renderer.domElement, null, controls);

  scene.add(new THREE.AmbientLight(0xffffff, 0.65));
  const keyLight = new THREE.DirectionalLight(0xffffff, 1.1);
  keyLight.position.set(1, 2, 2);
  scene.add(keyLight);
  const rimLight = new THREE.DirectionalLight(0x88ccff, 0.55);
  rimLight.position.set(-2, 1.5, -1);
  scene.add(rimLight);

  const loader = new GLTFLoader();
  loader.register((parser) => new VRMLoaderPlugin(parser));
  loader.register((parser) => new VRMAnimationLoaderPlugin(parser));

  const vrmGltf = await loader.loadAsync("/api/assets/vrm");
  const vrm = vrmGltf.userData.vrm;
  VRMUtils.removeUnnecessaryVertices(vrm.scene);
  VRMUtils.combineSkeletons(vrm.scene);
  vrm.scene.rotation.y = Math.PI;
  scene.add(vrm.scene);
  vrmState.vrm = vrm;

  const vrmaGltf = await loader.loadAsync("/api/assets/vrma");
  const animations = vrmaGltf.userData.vrmAnimations;
  if (animations?.length) {
    const clip = createVRMAnimationClip(animations[0], vrm);
    vrmState.mixer = new THREE.AnimationMixer(vrm.scene);
    const action = vrmState.mixer.clipAction(clip);
    action.play();
  }

  function resize() {
    const width = els.canvas.clientWidth;
    const height = els.canvas.clientHeight;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }

  window.addEventListener("resize", resize);
  resize();

  startVrmRenderLoop(renderer, () => {
    const delta = vrmState.clock.getDelta();
    vrmState.mixer?.update(delta);
    vrm.update(delta);
    updateLipSync();
    controls.update();
    renderer.render(scene, camera);
  });
}

function setupObsSync() {
  lunaChannel.onmessage = (event) => {
    const data = event.data;
    if (!data) return;

    if (data.type === "vrm-y" && typeof data.offsetY === "number") {
      vrmState.modelOffsetY = data.offsetY;
      if (vrmState.vrm) {
        vrmState.vrm.scene.position.y = data.offsetY;
      }
    }
  };
}

function startObsRelay() {
  const tick = async () => {
    try {
      await api("/api/obs/heartbeat", { method: "POST" });
      const data = await api("/api/obs/tts");
      if (data.audio_b64) {
        await playAssistantAudio(data.audio_b64);
      }
    } catch {
      // Server may be restarting; retry on next tick.
    }
  };

  tick();
  setInterval(tick, 400);

  window.addEventListener("beforeunload", () => {
    fetch("/api/obs/disconnect", { method: "POST", keepalive: true }).catch(() => {});
  });
}

async function boot() {
  setupObsSync();
  let appConfig = { player_name: "solonaras" };

  if (!isObsOverlay) {
    setupUi();
    setupMicControls();
    appendMessage("system", "Loading Luna…");
    await Promise.all([refreshStatus(), loadCaptureSources()]);
    appConfig = await api("/api/config");
    els.styleSelect.value = appConfig.style;
    els.watchInterval.value = appConfig.watch_interval_sec;
    const savedCapture = localStorage.getItem("luna-screen-capture");
    const wantCapture =
      savedCapture === "1"
        ? true
        : savedCapture === "0"
          ? false
          : Boolean(appConfig.screen_capture_enabled);
    if (wantCapture !== Boolean(appConfig.screen_capture_enabled)) {
      await setScreenCapture(wantCapture, { notify: false });
    } else {
      state.screenCaptureEnabled = Boolean(appConfig.screen_capture_enabled);
      updateScreenCaptureUi(false);
    }
    if (els.useScreen && savedCapture === "1") {
      els.useScreen.checked = true;
    }
    if (els.voiceMood && appConfig.voice_mood) {
      els.voiceMood.value = appConfig.voice_mood;
    }

    const obsLink = document.getElementById("obs-url");
    if (obsLink) {
      obsLink.href = `${window.location.origin}/obs`;
    }
  } else {
    startObsRelay();
  }

  try {
    await initVrmViewer();
    if (!isObsOverlay) {
      const player = appConfig.player_name || "player";
      appendMessage(
        "assistant",
        `Hey ${player}! I'm Luna, your co-host. Share your screen — games, Suno, whatever you're up to — and let's go.`,
      );
    }
  } catch (error) {
    if (!isObsOverlay) {
      appendMessage("system", `VRM load failed: ${error.message}`);
    }
  }

  if (!isObsOverlay) {
    syncWatchTimer();
    setInterval(refreshStatus, 15000);
  }
}

boot();
