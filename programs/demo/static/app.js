// 動画の顔匿名化デモの画面の処理．
// 入力 (カメラでの撮影またはファイル) と方式を選び，サーバーへジョブを送り，進捗を表示して結果を再生する．
"use strict";

const state = {
  methods: [],
  methodId: null,
  params: {},
  inputBlob: null,     // 新たにアップロードする動画
  inputName: "",
  lastJobId: null,     // 再実行で入力を再利用するジョブ
  history: [],
  ready: false,
  busy: false,
  maxDuration: 60,
};

const $ = (id) => document.getElementById(id);

/** 秒数を「m:ss」の形式にする． */
function formatTime(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

/** 方式とパラメータを短い文字列にする． */
function describe(methodId, params) {
  const values = Object.entries(params).map(([k, v]) => `${k === "gamma" ? "γ" : k === "alpha" ? "α" : k}=${v}`);
  return values.length ? `${methodId} (${values.join(", ")})` : methodId;
}

/** サーバーの状態 (モデルの読み込み) を確かめ，読み込み中なら繰り返し確かめる． */
async function pollServer() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();
    state.maxDuration = status.limits.max_duration_sec;
    document.querySelectorAll(".limit-duration").forEach((el) => (el.textContent = status.limits.max_duration_sec));
    document.querySelectorAll(".limit-height").forEach((el) => (el.textContent = status.limits.max_height));
    const box = $("server-status");
    if (status.error) {
      box.className = "status status-error";
      box.textContent = `モデルを読み込めませんでした: ${status.error}`;
      return;
    }
    state.ready = status.ready;
    box.className = status.ready ? "status status-ready" : "status status-loading";
    box.textContent = status.ready ? "準備完了：動画を用意して匿名化できます．" : "モデルを読み込み中です…";
    updateRunButton();
    if (!status.ready) setTimeout(pollServer, 2000);
  } catch (error) {
    $("server-status").className = "status status-error";
    $("server-status").textContent = "サーバーに接続できません．";
    setTimeout(pollServer, 3000);
  }
}

/** 方式の一覧を読み込んで表示する． */
async function loadMethods() {
  state.methods = await (await fetch("/api/methods")).json();
  const list = $("method-list");
  list.innerHTML = "";
  state.methods.forEach((method, index) => {
    const label = document.createElement("label");
    label.className = "method";
    label.innerHTML = `<input type="radio" name="method" value="${method.id}"><span class="label"></span><span class="desc"></span>`;
    label.querySelector(".label").textContent = method.label;
    label.querySelector(".desc").textContent = method.description;
    label.querySelector("input").addEventListener("change", () => selectMethod(method.id));
    list.appendChild(label);
    if (index === 0) {
      label.querySelector("input").checked = true;
    }
  });
  selectMethod(state.methods[0].id);
}

/** 方式を選び，そのパラメータの操作部を表示する． */
function selectMethod(methodId) {
  state.methodId = methodId;
  const method = state.methods.find((m) => m.id === methodId);
  document.querySelectorAll(".method").forEach((el) => {
    el.classList.toggle("selected", el.querySelector("input").value === methodId);
  });
  state.params = {};
  const container = $("method-params");
  container.innerHTML = "";
  method.params.forEach((spec) => {
    state.params[spec.name] = spec.default;
    const wrapper = document.createElement("div");
    wrapper.className = "param";
    wrapper.innerHTML = `<label><span></span><span class="value"></span></label>
      <input type="range" min="${spec.min}" max="${spec.max}" step="${spec.step}" value="${spec.default}">`;
    wrapper.querySelector("label span").textContent = spec.label;
    const value = wrapper.querySelector(".value");
    const slider = wrapper.querySelector("input");
    const show = () => (value.textContent = Number(slider.value).toFixed(spec.step === 1 ? 0 : 2));
    slider.addEventListener("input", () => { state.params[spec.name] = Number(slider.value); show(); });
    slider.dataset.name = spec.name;
    show();
    container.appendChild(wrapper);
  });
  const presetNames = { weak: "弱", medium: "中", strong: "強" };
  const presetKeys = Object.keys(method.presets || {});
  if (presetKeys.length) {
    const presets = document.createElement("div");
    presets.className = "presets";
    presets.innerHTML = `<span class="muted">匿名化の強さ：</span>`;
    presetKeys.forEach((key) => {
      const button = document.createElement("button");
      button.textContent = presetNames[key] || key;
      button.type = "button";
      button.addEventListener("click", () => applyPreset(method.presets[key]));
      presets.appendChild(button);
    });
    container.prepend(presets);
  }
}

/** プリセットの値をパラメータの操作部に反映する． */
function applyPreset(values) {
  Object.entries(values).forEach(([name, value]) => {
    const slider = document.querySelector(`#method-params input[data-name="${name}"]`);
    if (!slider) return;
    slider.value = value;
    slider.dispatchEvent(new Event("input"));
  });
}

/** 匿名化を開始できるかに応じてボタンの状態を更新する． */
function updateRunButton() {
  const hasInput = Boolean(state.inputBlob) || Boolean(state.lastJobId);
  $("run").disabled = !(state.ready && state.inputBlob && !state.busy);
  $("run-again").classList.toggle("hidden", !state.lastJobId);
  $("run-again").disabled = !(state.ready && state.lastJobId && !state.busy);
  return hasInput;
}

/** 処理する動画を設定し，プレビューを表示する． */
function setInput(blob, name) {
  state.inputBlob = blob;
  state.inputName = name;
  state.lastJobId = null;
  const preview = $("input-preview");
  if (preview.src) URL.revokeObjectURL(preview.src);
  preview.src = URL.createObjectURL(blob);
  $("input-name").textContent = `${name}（${(blob.size / 1024 / 1024).toFixed(1)} MB）`;
  $("input-ready").classList.remove("hidden");
  updateRunButton();
}

// ---------------------------------------------------------------- カメラでの撮影
let stream = null;
let recorder = null;
let chunks = [];
let timerId = null;

/** カメラの欄にメッセージを表示する．kind は "info"，"ok"，"error" のいずれか． */
function showCameraMessage(message, kind) {
  const box = $("camera-message");
  box.textContent = message;
  box.className = `notice${kind === "error" ? " notice-error" : kind === "ok" ? " notice-ok" : ""}`;
}

/**
 * このページでカメラを使えるかを調べる．使えない場合は理由を返す．
 * ブラウザは，localhost 以外の http で開いたページにはカメラを使わせない (navigator.mediaDevices が存在しない)．
 */
function cameraUnavailableReason() {
  if (!window.isSecureContext) {
    return `このページは ${location.origin} で開かれているため，ブラウザの制限によりカメラを使えません．` +
      `サーバーと同じ PC のブラウザで http://localhost:${location.port || 80} を開くか，「ファイルを選択」を使ってください．`;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || !window.MediaRecorder) {
    return "このブラウザはカメラでの録画に対応していません．Chrome，Edge，Firefox，Safari の最新版で開くか，「ファイルを選択」を使ってください．";
  }
  return null;
}

/** カメラの起動に失敗したときの理由を，エラーの種類に応じた説明にする． */
function describeCameraError(error) {
  const embedded = window.self !== window.top;
  switch (error && error.name) {
    case "NotAllowedError":
    case "SecurityError":
      return "カメラの使用が許可されませんでした．アドレスバーのカメラのアイコン (またはサイトの設定) から許可して，もう一度押してください．" +
        (embedded ? "VS Code の内蔵ブラウザなど，ページを埋め込んで表示する画面ではカメラを使えないことがあります．通常のブラウザで開いてください．" : "");
    case "NotFoundError":
    case "OverconstrainedError":
      return "カメラが見つかりませんでした．カメラが接続されているか確認してください．";
    case "NotReadableError":
    case "AbortError":
      return "カメラを開けませんでした．他のアプリ (ビデオ会議など) がカメラを使っていないか確認してください．";
    default:
      return `カメラを起動できませんでした (${error && error.name ? error.name + ": " : ""}${error && error.message ? error.message : error})．`;
  }
}

/** カメラとマイクを起動してプレビューを表示する． */
async function startCamera() {
  const reason = cameraUnavailableReason();
  if (reason) {
    showCameraMessage(reason, "error");
    return;
  }
  const button = $("camera-start");
  button.disabled = true;
  button.textContent = "起動中…";
  showCameraMessage("ブラウザからカメラの使用の許可を求められたら，「許可」を選んでください．", "info");
  try {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: true });
    } catch (error) {
      if (error && error.name === "NotAllowedError") throw error;
      // マイクが無い・使えない場合は映像だけで録画する
      stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    }
  } catch (error) {
    button.disabled = false;
    button.textContent = "カメラを起動";
    showCameraMessage(describeCameraError(error), "error");
    return;
  }
  const preview = $("camera-preview");
  preview.srcObject = stream;
  await preview.play().catch(() => {});
  button.textContent = "カメラ起動中";
  $("record-start").disabled = false;
  const audio = stream.getAudioTracks().length ? "" : "（マイクが使えないため，音声なしで録画します）";
  showCameraMessage(`カメラを起動しました．「録画開始」を押すと録画が始まります．${audio}`, "ok");
}

/** ブラウザが対応している録画形式を選ぶ． */
function pickMimeType() {
  const candidates = ["video/webm;codecs=vp9,opus", "video/webm;codecs=vp8,opus", "video/webm", "video/mp4"];
  return candidates.find((type) => window.MediaRecorder && MediaRecorder.isTypeSupported(type)) || "";
}

/** 録画を開始する．上限の秒数に達したら自動で停止する． */
function startRecording() {
  chunks = [];
  const mimeType = pickMimeType();
  recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
  recorder.ondataavailable = (event) => event.data.size && chunks.push(event.data);
  recorder.onstop = () => {
    const type = recorder.mimeType || "video/webm";
    const extension = type.includes("mp4") ? "mp4" : "webm";
    setInput(new Blob(chunks, { type }), `camera_${new Date().toISOString().replace(/[:.]/g, "-")}.${extension}`);
  };
  recorder.start(500);
  showCameraMessage("録画中です．「録画停止」を押すと終了し，下に処理する動画として表示されます．", "info");
  const started = Date.now();
  $("record-start").disabled = true;
  $("record-stop").disabled = false;
  timerId = setInterval(() => {
    const elapsed = (Date.now() - started) / 1000;
    $("record-timer").textContent = `録画中 ${formatTime(elapsed)} / ${formatTime(state.maxDuration)}`;
    if (elapsed >= state.maxDuration) stopRecording();
  }, 250);
}

/** 録画を停止する． */
function stopRecording() {
  if (recorder && recorder.state !== "inactive") recorder.stop();
  clearInterval(timerId);
  $("record-timer").textContent = "";
  $("record-start").disabled = false;
  $("record-stop").disabled = true;
  showCameraMessage("録画を終えました．下の「処理する動画」を確認して，方式を選んで匿名化を開始してください．もう一度録画することもできます．", "ok");
}

// ---------------------------------------------------------------- ジョブの送信と進捗
/** エラーを表示し，見える位置までスクロールする． */
function showError(message) {
  $("error").textContent = message;
  $("error").classList.remove("hidden");
  $("error").scrollIntoView({ behavior: "smooth", block: "center" });
}

/** タブを切り替える．name は "camera" または "file"． */
function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) => {
    t.classList.toggle("active", t.dataset.tab === name);
    t.setAttribute("aria-selected", t.dataset.tab === name ? "true" : "false");
  });
  $("panel-camera").classList.toggle("hidden", name !== "camera");
  $("panel-file").classList.toggle("hidden", name !== "file");
}

/** 画面を開いた時点でカメラを使えるかを調べ，使えない場合は理由を表示してファイルの選択に切り替える． */
function checkCamera() {
  const reason = cameraUnavailableReason();
  if (reason) {
    showCameraMessage(reason, "error");
    $("camera-start").disabled = true;
    switchTab("file");
  } else if (window.self !== window.top) {
    showCameraMessage("ページが埋め込みで表示されています (VS Code の内蔵ブラウザなど)．カメラを使えない場合は通常のブラウザで開いてください．", "info");
  }
}

/** ジョブを送信する．reuse が真の場合は直前の入力を再利用する． */
async function submit(reuse) {
  $("error").classList.add("hidden");
  const form = new FormData();
  form.append("method", state.methodId);
  form.append("params", JSON.stringify(state.params));
  form.append("seed", String(Number($("seed").value) || 0));
  form.append("fix_norm", $("fix-norm").checked ? "true" : "false");
  if (reuse) {
    form.append("source_job", state.lastJobId);
  } else {
    form.append("file", state.inputBlob, state.inputName);
  }
  state.busy = true;
  updateRunButton();
  $("progress").classList.remove("hidden");
  setProgress(0, reuse ? "送信中…" : "動画をアップロード中…");
  try {
    const response = await fetch("/api/jobs", { method: "POST", body: form });
    const job = await response.json();
    if (!response.ok) throw new Error(job.detail || "送信に失敗しました．");
    state.inputBlob = null;
    await followJob(job.id);
  } catch (error) {
    showError(error.message);
  } finally {
    state.busy = false;
    updateRunButton();
  }
}

/** 進捗バーと説明を更新する． */
function setProgress(fraction, label) {
  $("progress-fill").style.width = `${Math.round(fraction * 100)}%`;
  $("progress-label").textContent = label;
}

/** ジョブが終わるまで状態を確かめ続ける． */
async function followJob(jobId) {
  const started = Date.now();
  for (;;) {
    const job = await (await fetch(`/api/jobs/${jobId}`)).json();
    const elapsed = formatTime((Date.now() - started) / 1000);
    const queue = job.queue_position > 1 ? `（順番待ち ${job.queue_position - 1} 件）` : "";
    setProgress(job.progress, `${job.stage_label} ${Math.round(job.progress * 100)}%  経過 ${elapsed} ${queue}`);
    if (job.stage === "done") {
      state.lastJobId = job.id;
      showResult(job);
      addHistory(job);
      return;
    }
    if (job.stage === "error") throw new Error(job.error || "処理に失敗しました．");
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
}

/** 結果の動画と統計を表示する． */
function showResult(job) {
  const s = job.summary;
  const stamp = Date.now();
  $("result-original").src = `/api/jobs/${job.id}/video/original?t=${stamp}`;
  $("result-anonymized").src = `/api/jobs/${job.id}/video/anonymized?t=${stamp}`;
  $("download").href = `/api/jobs/${job.id}/video/anonymized?download=true`;
  $("result-caption").textContent = `匿名化後：${describe(job.method, job.params)}`;
  const timings = s.timings_sec;
  const stats = [
    ["方式", describe(job.method, job.params) + (job.fix_norm ? "，ノルム補正あり" : "") + `，シード ${job.seed}`],
    ["動画", `${s.width}×${s.height}，${s.fps.toFixed(1)} fps，${s.num_frames} フレーム（${s.duration.toFixed(1)} 秒）`],
    ["検出した人物", s.num_people === 0 ? "顔が検出されませんでした" : `${s.num_people} 人（各人物の顔の数：${s.faces_per_person.join("，")}）`],
    ["顔が映っているフレーム", `${s.frames_with_faces} / ${s.num_frames}`],
    ["元の顔ベクトルとの類似度", s.cos_source_vs_original.length ? s.cos_source_vs_original.map((c) => c.toFixed(2)).join("，") : "-"],
    ["処理時間", `${s.total_sec.toFixed(1)} 秒（検出 ${timings.detect.toFixed(1)}，追跡 ${timings.track.toFixed(1)}，生成 ${timings.generate.toFixed(1)}，貼り戻し ${timings.paste.toFixed(1)}）`],
  ];
  const list = $("result-stats");
  list.innerHTML = "";
  stats.forEach(([key, value]) => {
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = key;
    dd.textContent = value;
    list.append(dt, dd);
  });
  $("result").classList.remove("hidden");
  setProgress(1, `完了（${s.total_sec.toFixed(1)} 秒）`);
}

/** 実行履歴に追加する．履歴をクリックすると，その結果を再表示する． */
function addHistory(job) {
  state.history.unshift(job);
  const list = $("history");
  list.innerHTML = "";
  state.history.forEach((item) => {
    const li = document.createElement("li");
    li.innerHTML = "<span></span><span class='muted'></span>";
    li.children[0].textContent = describe(item.method, item.params);
    li.children[1].textContent = `${item.input_name}・${item.summary.num_people} 人`;
    li.classList.toggle("current", item.id === job.id);
    li.addEventListener("click", () => {
      document.querySelectorAll("#history li").forEach((el) => el.classList.remove("current"));
      li.classList.add("current");
      showResult(item);
    });
    list.appendChild(li);
  });
  $("history-card").classList.remove("hidden");
}

// ---------------------------------------------------------------- 画面の初期化
function setupTabs() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => switchTab(tab.dataset.tab));
  });
}

function setupFileInput() {
  const zone = $("drop-zone");
  $("file-input").addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (file) setInput(file, file.name);
  });
  ["dragenter", "dragover"].forEach((name) => zone.addEventListener(name, (event) => {
    event.preventDefault();
    zone.classList.add("dragover");
  }));
  ["dragleave", "drop"].forEach((name) => zone.addEventListener(name, (event) => {
    event.preventDefault();
    zone.classList.remove("dragover");
  }));
  zone.addEventListener("drop", (event) => {
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith("video/")) setInput(file, file.name);
    else showError("動画ファイルを選択してください．");
  });
}

function setup() {
  setupTabs();
  setupFileInput();
  $("camera-start").addEventListener("click", startCamera);
  $("record-start").addEventListener("click", startRecording);
  $("record-stop").addEventListener("click", stopRecording);
  $("run").addEventListener("click", () => submit(false));
  $("run-again").addEventListener("click", () => submit(true));
  $("play-both").addEventListener("click", () => {
    const a = $("result-original");
    const b = $("result-anonymized");
    a.currentTime = 0;
    b.currentTime = 0;
    b.muted = true;
    a.play();
    b.play();
  });
  checkCamera();
  loadMethods();
  pollServer();
}

setup();
