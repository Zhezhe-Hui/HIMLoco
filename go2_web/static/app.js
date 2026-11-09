"use strict";
const $ = id => document.getElementById(id);
const state = {experiments: [], selected: null, index: 0, mode: "history", layer: "obstacles",
  playing: false, rate: 1, lastTick: 0, replayElapsed: 0, status: null, lastActive: null, toastTimer: null,
  cameraPhase:"disabled",cameraPreview:-1,referenceHistoryId:"ours_ready_shadow_20261030",lease:null,calLoaded:false,lastRealOutcome:null};

async function api(path, method = "GET", body) {
  const response = await fetch(path, {method, cache: "no-store",
    ...(body === undefined ? {} : {headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)})});
  const value = await response.json();
  if(response.status===401){location.replace("/");throw new Error("请重新登录");}
  if (!response.ok) throw new Error(value.error || "请求失败");
  return value;
}
function toast(message, error = false) {
  clearTimeout(state.toastTimer);
  $("toast").textContent = message;
  $("toast").classList.toggle("error", error);
  $("toast").hidden = false;
  state.toastTimer = setTimeout(() => $("toast").hidden = true, 4500);
}
const safe = value => String(value ?? "").replace(/[&<>"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[ch]));
$("logout").addEventListener("click",async()=>{try{await api("/api/auth/logout","POST",{});location.replace("/");}catch(error){toast(error.message,true);}});
fetch("/api/auth",{cache:"no-store"}).then(r=>r.json()).then(s=>{$("logout").hidden=!s.required;}).catch(()=>{});
const fixed = (value, digits=2) => Number.isFinite(Number(value)) && value !== null && value !== undefined ? Number(value).toFixed(digits) : "—";
function showView(view) {
  document.body.dataset.view=view;
  document.querySelectorAll(".view").forEach(el => el.classList.toggle("active", el.id === `view-${view}`));
  document.querySelectorAll(".nav-item").forEach(el => el.classList.toggle("active", el.dataset.view === view));
  $("breadcrumb").textContent = {workspace:"工作台",records:"实验记录",environment:"参数与环境"}[view];
  if (view === "workspace") requestAnimationFrame(drawAll);
}
document.querySelectorAll(".nav-item").forEach(button => button.addEventListener("click", () => showView(button.dataset.view)));
document.querySelector(".brand").addEventListener("click", event => {event.preventDefault(); showView("workspace");});
document.querySelectorAll("[data-pane]").forEach(button=>button.addEventListener("click",()=>{
  document.body.dataset.workspacePane=button.dataset.pane;
  document.querySelectorAll("[data-pane]").forEach(tab=>tab.setAttribute("aria-selected",String(tab===button)));
  requestAnimationFrame(drawAll);
}));

function params() {
  const result = {};
  for (const name of ["forward","lateral","max_vx","duration"]) {
    if (!$(name).checkValidity() || $(name).value.trim() === "") {
      $(name).reportValidity(); throw new Error("请检查测试参数范围");
    }
    result[name] = Number($(name).value);
  }
  return result;
}
function applyParams(values) {
  for (const name of ["forward","lateral","max_vx","duration"]) $(name).value = values[name];
}
$("saveDraft").addEventListener("click", async () => {
  try {
    await api("/api/parameters", "POST", params());
    $("draftStatus").textContent = "已保存 · 下次任务启动生效";
    toast("参数已保存，运行中的目标不会改变");
  } catch (error) {toast(error.message, true);}
});
for (const name of ["forward","lateral","max_vx","duration"]) $(name).addEventListener("input", () => $("draftStatus").textContent = "参数已修改 · 尚未保存");

function setMode(mode) {
  state.mode = mode;
  state.playing = false;
  $("playReplay").textContent = "▶";
  $("historyMode").classList.toggle("active", mode === "history");
  $("simulationMode").classList.toggle("active", mode === "simulation");
  $("liveMode").classList.toggle("active", mode === "live");
  const wired=state.status?.real?.enabled, live=mode==='live';
  $("realActions").hidden=!wired || !live;
  $("simulationActions").hidden=wired && live;
  $("realRecording").hidden=!wired || !live;
  $("simulationRecording").hidden=wired && live;
  $("controlScope").textContent=live?'实机 · 手动启动':'仅模拟';
  $("modeNotice").textContent = mode === "live" ? "有线实机模式" : mode === "history" ? "历史回放模式" : "离线流程模拟";
  $("modeDescription").textContent = mode === "live" ? "真实相机与本体状态；手动启动规划或测试。" : mode === "history" ? "使用已保存的实机数据。页面操作不会向机器狗发送指令。" : "数据由流程模型生成，不运行真实规划器，不连接机器狗。";
  $("replayHint").textContent = mode === "history" ? "时间轴回放遥测；图像和地图来自最近留存快照，不是连续录像。" : "模拟任务验证操作流程；相机来源以画面标签为准，模拟任务不会控制机器人。";
  $("runSelect").disabled = mode !== "history";
  $("playReplay").disabled = mode !== "history";
  $("timeline").disabled = mode !== "history";
  $("replayRate").disabled = mode !== "history";
  renderTelemetry(); drawAll();
}
$("historyMode").addEventListener("click", () => {
  const id = state.selected?.source === "historical" ? state.selected.id : $("runSelect").value;
  if (id) loadExperiment(id).catch(e=>toast(e.message,true));
  else setMode("history");
});
$("simulationMode").addEventListener("click", () => setMode("simulation"));
$("liveMode").addEventListener("click",()=>{setMode('live');updateReal();});
$("openCalibration").addEventListener("click",()=>showView('environment'));

async function refreshExperiments() {
  const {items} = await api("/api/experiments");
  state.experiments = items;
  const selectedId = state.selected?.source === "historical" ? state.selected.id : "ours_ready_shadow_20261030";
  $("runSelect").replaceChildren();
  for (const entry of items.filter(e => e.source === "historical")) {
    const option = document.createElement("option"); option.value = entry.id; option.textContent = entry.title;
    option.selected = entry.id === selectedId;
    $("runSelect").append(option);
  }
  $("navCount").textContent = items.length;
  renderRecords();
}
function recordRow(entry, full = false) {
  const sim = entry.source === "simulation";
  const warning = entry.outcome === "no_steps" || entry.outcome === "simulated_timeout";
  return `<tr><td>${safe(entry.title)}${full ? `<small>${safe(entry.id)}</small>` : ""}</td><td><span class="badge ${sim ? "blue" : "muted"}">${sim ? "流程模拟" : "真实历史"}</span></td><td>${fixed(entry.duration,1)} s <span class="subtle">/ ${entry.sample_count} 帧</span></td><td><span class="result-pill ${warning ? "warning" : sim ? "sim" : ""}">${safe(entry.outcome_label)}</span></td><td><div class="row-actions"><button class="text-button" data-exp="${safe(entry.id)}">查看</button><a class="text-button" href="/api/experiments/${encodeURIComponent(entry.id)}/export" download>导出 ↓</a></div></td></tr>`;
}
function renderRecords() {
  const search = $("recordSearch").value.trim().toLowerCase();
  const items = state.experiments.filter(e => !search || (e.title+e.id).toLowerCase().includes(search));
  $("allRows").innerHTML = items.length ? items.map(e => recordRow(e,true)).join("") : '<tr><td colspan="5">没有匹配的实验记录</td></tr>';
}
$("recordSearch").addEventListener("input", renderRecords);
$("refreshRecords").addEventListener("click", () => refreshExperiments().then(() => toast("实验记录已刷新")).catch(e => toast(e.message,true)));
$("allRows").addEventListener("click", event => {
  const button = event.target.closest("button[data-exp]");
  if (button) {showView("workspace"); loadExperiment(button.dataset.exp).catch(e => toast(e.message,true));}
});
async function loadExperiment(id) {
  const entry = await api(`/api/experiments/${encodeURIComponent(id)}`);
  state.selected = entry;
  state.index = Math.max(0, entry.rows.length-1);
  if (entry.snapshot_frame !== null) {
    const found = entry.rows.findIndex(row => row.frame === entry.snapshot_frame);
    if (found >= 0) state.index = found;
  }
  if (entry.source === "historical") {
    state.referenceHistoryId=id;
    $("runSelect").value = id;
    if (!state.status?.camera?.has_frame) {
      for (const [element,kind] of [["rgbImage","rgb"],["depthImage","depth"]]) {
        $(element).classList.toggle("no-image", !entry.snapshot);
        if (entry.snapshot) $(element).src = `/api/assets/${encodeURIComponent(id)}/${kind}`;
      }
      $("rgbCaption").textContent = `历史快照${entry.snapshot_frame ? " · 帧 "+entry.snapshot_frame : ""} · 非实时视频`;
    }
  }
  setMode(entry.source === "historical" ? "history" : "simulation");
  $("timeline").max = Math.max(0,entry.rows.length-1);
  renderTelemetry(); drawAll();
}
$("runSelect").addEventListener("change", () => loadExperiment($("runSelect").value).catch(e => toast(e.message,true)));
$("timeline").addEventListener("input", () => {state.index = Number($("timeline").value); state.playing=false; $("playReplay").textContent="▶"; renderTelemetry(); drawSpeed();});
$("playReplay").addEventListener("click", () => {
  if (!state.selected?.rows.length || state.mode !== "history") return;
  state.playing = !state.playing;
  if (state.playing) {
    if (state.index >= state.selected.rows.length-1) state.index = 0;
    state.replayElapsed = state.selected.rows[state.index].elapsed || 0;
    state.lastTick = performance.now();
  }
  $("playReplay").textContent = state.playing ? "Ⅱ" : "▶";
  $("playReplay").setAttribute("aria-label", state.playing ? "暂停历史遥测" : "播放历史遥测");
});
$("replayRate").addEventListener("click", () => {state.rate = state.rate === 1 ? 2 : state.rate === 2 ? .5 : 1; $("replayRate").textContent=state.rate+"×";});

async function startTask(kind) {
  try {
    const result = await api("/api/tasks/start", "POST", {kind,parameters:params()});
    state.lastActive = result.task.id;
    state.selected = null;
    setMode("simulation");
    await refreshStatus(); await refreshExperiments();
    toast(kind === "shadow" ? "影子流程模拟已开始，不计算真实规划" : "模拟任务已开始，未发送机器人指令");
  } catch (error) {toast(error.message, true);}
}
$("startSimulation").addEventListener("click", () => startTask("simulation"));
$("startShadow").addEventListener("click", () => startTask("shadow"));
$("stopTask").addEventListener("click", async () => {
  try {
    state.lease=null;
    if(state.status?.real?.enabled)await api('/api/real/stop','POST',{});
    const result = await api("/api/tasks/stop","POST",{}); await refreshStatus(); await refreshExperiments(); toast(state.status?.real?.enabled?'任务已停止':result.message);
  }
  catch (error) {toast(error.message,true);}
});
$("recording").addEventListener("change", async () => {
  try {const result=await api("/api/recording","POST",{enabled:$("recording").checked}); toast(result.enabled ? "开始记录模拟遥测；不录制实机数据" : "模拟遥测记录已暂停"); await refreshStatus();}
  catch(error) {toast(error.message,true); await refreshStatus();}
});
async function refreshStatus() {
  const previous = state.status?.active?.outcome;
  state.status = await api("/api/state");
  updateCamera();
  updateReal();
  const active = state.status.active;
  const running = active?.outcome === "running";
  const realRunning=state.status.real?.task?.outcome==='running';
  $("startSimulation").disabled = running||realRunning; $("startShadow").disabled = running||realRunning; $("stopTask").disabled = !running&&!realRunning;
  $("recording").checked = state.status.recording;
  $("recordLight").classList.toggle("active", running && state.status.recording);
  $("taskState").classList.toggle("running", running);
  if (active && !realRunning && state.mode!=='live') {
    $("taskTitle").textContent = active.outcome_label;
    $("taskMessage").textContent = `${active.kind === "shadow" ? "影子流程" : "流程模拟"} · ${active.summary.recorded_samples || active.rows.length} 条记录 · 非实机`;
    $("taskElapsed").textContent = `${fixed(active.duration,1)} s`;
    $("taskProgress").style.width = `${Math.min(100, active.duration/active.parameters.duration*100)}%`;
  }
  if (state.mode === "simulation") {renderTelemetry(); drawAll();}
  if (previous === "running" && active?.outcome !== "running") {
    await refreshExperiments();
    toast(`${active.outcome_label}，数据仅用于离线流程检查`);
  }
}
function updateCamera() {
  const camera=state.status.camera || {enabled:false,phase:"disabled"};
  const onboard=state.status.runtime?.location==="robot";
  $("serviceLocation").textContent=onboard?"机器狗算力盒子":"当前电脑";
  $("deploymentBadge").textContent=onboard?"板载服务":"电脑服务";
  $("dataConnection").textContent=onboard?"D435i 板载 USB · 历史 / 模拟":"D435i 经 SSH · 历史 / 模拟";
  $("captureCamera").disabled=state.capturing || !camera.connected;
  const wasLive=["connecting","streaming","stale","error"].includes(state.cameraPhase);
  $("runtimeBadge").textContent=camera.connected?"相机实时 · 控制离线":"离线控制";
  const busy=camera.phase==="connecting";
  const active=!["disabled","idle"].includes(camera.phase);
  $("connectCamera").hidden=busy || camera.connected;
  $("connectCamera").disabled=!camera.enabled;
  $("connectCamera").textContent=camera.enabled?(camera.phase==="error"?"重试相机连接（只读）":"连接实时相机（只读）"):"实时入口未启用";
  $("connectCamera").title=state.status?.real?.enabled?"读取D435i和本体状态；连接不会发送运动指令":camera.enabled?"仅读取D435i，不读取或控制机器人运动":"使用 one_click.sh web-live 准备只读相机预览";
  $("disconnectCamera").hidden=!active;
  $("disconnectCamera").textContent=busy?"取消相机连接":"断开相机";
  if(camera.has_frame){
    const live=camera.connected;
    $("rgbSourceBadge").textContent=live?"实时 · 只读":"断流 · 定格";
    $("depthSourceBadge").textContent=live?"实时 · 0–4 m":"断流 · 定格";
    $("rgbCaption").textContent=`${live?"实时相机":"最后一帧，非实时"} · 帧 ${camera.preview_frame} · 采集 ${fixed(camera.received_fps,1)} / 显示 ${fixed(state.displayFps,1)} fps`;
    $("depthCaption").textContent=`最近接收 ${fixed(camera.received_age_ms,0)} ms 前 · 预览已着色，非原始深度`;
    $("modeNotice").textContent=live?"相机实时只读":"相机流中断";
    $("modeDescription").textContent=live?(state.mode==="history"?"只有相机实时；下方地图和遥测仍为历史回放，未发送运动指令。":"相机实时；测试任务仍为流程模拟，未发送运动指令。"):("画面已定格，请检查相机和网络；地图与遥测仍非实时。"+(camera.error?" "+camera.error.slice(0,160):""));
  }else if(busy){
    $("rgbSourceBadge").textContent="连接中";
    $("depthSourceBadge").textContent="连接中";
    $("modeNotice").textContent="只读相机连接中";
    $("modeDescription").textContent="正在等待 D435i 图像；仍保留历史画面，不发送运动指令。";
  }else{
    $("rgbSourceBadge").textContent="定格快照";
    $("depthSourceBadge").textContent="历史 · 0–4 m";
    if(wasLive){
      for(const [element,kind] of [["rgbImage","rgb"],["depthImage","depth"]]) $(element).src=`/api/assets/${encodeURIComponent(state.referenceHistoryId)}/${kind}`;
      $("rgbCaption").textContent="历史留存快照 · 非实时视频";
      $("depthCaption").textContent="蓝 → 近 / 暖色 → 远 · 历史快照";
      state.cameraPreview=-1;
      $("modeNotice").textContent=state.mode==="history"?"历史回放模式":"离线流程模拟";
      $("modeDescription").textContent="相机已断开，当前不发送运动指令。";
    }
    if(camera.phase==="error"){
      $("modeNotice").textContent="相机连接失败";
      $("modeDescription").textContent=(onboard?"保留历史快照。请检查相机 USB、占用进程与采集程序。":"保留历史快照。请检查机器人网络、SSH 登录及相机程序。")+(camera.error?" "+camera.error.slice(0,160):"");
    }
  }
  document.querySelectorAll(".camera-panel").forEach(panel=>panel.classList.toggle("stream-stale",camera.has_frame&&!camera.connected));
  state.cameraPhase=camera.phase;
}
$("connectCamera").addEventListener("click",async()=>{
  try{state.cameraPreview=-1;await api(state.status?.real?.enabled?'/api/real/connect':"/api/camera/connect","POST",{});if(state.status?.real?.enabled)setMode('live');await refreshStatus();toast("相机与状态连接已请求；不会发送运动指令");}
  catch(error){toast(error.message,true);}
});
$("disconnectCamera").addEventListener("click",async()=>{
  try{state.lease=null;await api(state.status?.real?.enabled?'/api/real/disconnect':"/api/camera/disconnect","POST",{});await refreshStatus();toast("实机链路已断开");}
  catch(error){toast(error.message,true);}
});
function currentEntry() {
  if(state.mode==='live')return {rows:state.status?.real?.rows||[],duration:state.status?.real?.task?.duration||0,current:state.status?.real?.current};
  if (state.mode === "simulation") {
    if (state.selected?.source === "simulation" && state.status?.active?.id !== state.selected.id) return state.selected;
    return state.status?.active || (state.selected?.source === "simulation" ? state.selected : null);
  }
  return state.selected;
}

function updateReal(){
  const real=state.status?.real;
  $("liveMode").hidden=!real?.enabled;
  $("saveCalibration").disabled=$("fitGround").disabled=$("approveCalibration").disabled=!real?.enabled;
  const board=state.status.runtime?.location==='robot';
  $("environmentMode").textContent=real?.enabled?(board?'板载实机感知 / 规划 / 受限执行':'有线实机感知 / 规划 / 受限执行'):'历史 / 流程模拟';
  $("motionInterface").textContent=real?.enabled?(real.task?.kind==='execute'&&real.task?.outcome==='running'?'受限速度桥运行中':'只读状态桥；实际执行需手动启动'):'未启用';
  if(!real?.enabled)return;
  $("connectCamera").textContent='连接相机与本体状态';
  $("serviceLocation").textContent=board?'机器狗算力盒子':'当前电脑 · 127.0.0.1';
  $("dataConnection").textContent=board?'D435i 本地 USB / 本体有线 DDS / 浏览器无线':'网线 / D435i SSH / Go2 DDS';
  $("captureCamera").textContent=board?'截图并下载':'截图到桌面';
  $("captureCamera").title=board?'同帧RGB和深度保存在盒子，并下载到当前设备':'保存同一实时帧的RGB和深度到电脑桌面';
  $("sidebarCompute").textContent=board?'盒子运行感知与规划':'电脑运行感知与规划';
  $("sidebarConnection").textContent=board?'板载运行 · 连接状态见顶部':'有线开发 · 连接状态见顶部';
  const task=real.task,running=task?.outcome==='running';
  $("startRealShadow").disabled=running||!real.inputs_fresh||state.status.active?.outcome==='running';
  $("openExecute").disabled=running||!real.inputs_fresh;
  updateExecuteReadiness();
  $("recordRGBD").disabled=running;
  $("commandCount").textContent=real.bridge_writes||0;
  $("commandFoot").textContent='写入次数，不代表实际迈步或厂商确认';
  const cal=real.calibration,values=cal.values;
  if(!state.calLoaded){
    $("calForward").value=values.camera_translation_body_m[0];$("calLeft").value=values.camera_translation_body_m[1];
    $("calYaw").value=values.mounting_yaw_deg||0;$("xyMeasured").checked=!!values.xy_physical_confirmed;$("yawConfirmed").checked=!!values.mounting_yaw_confirmed;
    state.calLoaded=true;
  }
  $("calibrationBadge").textContent=cal.approved_this_session?'完整标定已确认':cal.ground_checked_this_session?'地面已复核 · 测量待确认':'初步值 · 本次地面未复核';
  $("calibrationMessage").textContent=cal.job.message||'先连接实机，在水平地面静止站立至少2秒，再复核地面。';
  $("calibrationValues").textContent=`相机位置 [前、左、上]：${values.camera_translation_body_m.map(v=>fixed(v,4)).join(', ')} m\n站立机身高度：${fixed(values.standing_base_height_m,4)} m\n相机离地高度：${fixed(values.ground_camera_height_m,4)} m\n地面内点：${fixed(values.ground_inlier_fraction*100,1)}% · 残差：${fixed(values.ground_rms_m*1000,2)} mm\n前后/左右：${values.xy_physical_confirmed?'已实测':'估计，未实测确认'} · 安装偏航：${values.mounting_yaw_confirmed?'已核对':'未确认'}\n额外足迹裕量：${fixed(values.extra_footprint_margin_m,2)} m`;
  $("fitGround").disabled=running||!real.inputs_fresh||cal.job.phase==='running';
  $("saveCalibration").disabled=running||cal.job.phase==='running';
  $("approveCalibration").disabled=running||!cal.ground_checked_this_session||!values.xy_physical_confirmed||!values.mounting_yaw_confirmed;
  if(state.mode==='live'){
    $("runtimeBadge").textContent=running?(task.kind==='execute'?'实际执行中':'真实影子规划'):(real.connected?'实机状态已连接':'实机未连接');
    $("modeNotice").textContent=running?(task.kind==='execute'?'实际短距离试验':'真实影子规划 · 不发速度'):(board?'板载实机模式':'有线实机模式');
    $("modeDescription").textContent=running?'目标固定在启动时本体里程计坐标系；地图与遥测来自本次实机。':'先连接相机与状态、检查标定，再手动启动。地图尚未规划或显示上次任务帧。';
    if(!state.status.camera.has_frame){
      $("rgbImage").classList.add('no-image');$("depthImage").classList.add('no-image');
      $("rgbSourceBadge").textContent=$("depthSourceBadge").textContent='等待真实图像';
      $("rgbCaption").textContent=$("depthCaption").textContent='实机未收到图像，不使用历史图像代替';
    }
    $("taskState").classList.toggle('running',running);
    $("taskTitle").textContent=task?.outcome_label||'等待真实任务';
    $("taskMessage").textContent=real.bridge_error||(!real.inputs_fresh?'连接相机并等候新鲜状态':running?'真实双层代价 FMM + 标准足迹 MPPI':'初步标定不可当作完整标定');
    $("taskElapsed").textContent=task?`${fixed(task.duration,1)} s`:'—';
    $("taskProgress").style.width=task?`${Math.min(100,100*task.duration/task.parameters.duration)}%`:'0%';
    renderTelemetry();drawAll();
  }
  if(state.lastRealOutcome==='running' && task?.outcome!=='running'){
    state.lease=null;refreshExperiments().catch(error=>toast(error.message,true));
    toast(task?.outcome_label||'真实任务已结束');
  }
  state.lastRealOutcome=task?.outcome;
}

async function startReal(kind){
  try{
    const takeover=kind==='execute'&&$("executeProfile").value==='takeover';
    const result=await api('/api/real/start','POST',{kind,parameters:takeover?takeoverParams():params(),record_rgbd:$("recordRGBD").checked,
      takeover_test:takeover,remote_verified:takeover&&$("remoteVerified").checked,
      operator_confirmed:kind==='execute'&&$("operatorConfirmed").checked,preliminary_ack:kind==='execute'&&$("preliminaryAck").checked});
    state.lease=result.result.lease;setMode('live');
    if($("executeDialog").open)$("executeDialog").close();
    await refreshStatus();toast(kind==='shadow'?'已启动真实影子规划，不发送速度':'已启动本次实际试验，可随时停止');
  }catch(error){toast(error.message,true);$("executeReadiness").textContent=error.message;}
}
$("startRealShadow").addEventListener('click',()=>startReal('shadow'));
function takeoverParams(){return {forward:2,lateral:0,max_vx:.30,duration:Number($("takeoverDuration").value)};}
function updateExecuteReadiness(){
  const real=state.status?.real, take=$("executeProfile").value==='takeover', rc=real?.remote;
  $("takeoverDurationRow").hidden=$("remoteVerifiedRow").hidden=!take;
  $("remoteReadout").hidden=!take;
  const axes=rc?.axes?.map(v=>fixed(v,2)).join(' / ')||'—';
  $("remoteReadout").textContent=`手柄数据：${rc?.ready?'新鲜且已归中':rc?.fresh?'请松开按键并归中0.5秒':'未收到新鲜数据'}；摇杆 ${axes}；按键 ${rc?.keys??'—'}；操作验证：${rc?.input_verified?'已检测到近期输入':'请先只读核对实际操作，再归中'}。核对时网页不发程序指令，手柄仍能控制狗，请保持空旷。测试中摇杆超过约20%或任意按键终止本轮，不自动恢复。数据通道超时也停止，但数据新鲜不等于手柄无线链路在线，请保持手柄开机连接。`;
  const p=take?takeoverParams():params();
  $("executeSummary").textContent=take?`手柄接管测试：目标前方2m，直行0.30m/s，最长${p.duration}s；到点、障碍、数据超时或手柄操作都会提前结束。轻推摇杆并回中，确认程序退出后再按日常方式遥控。请保持前方至少4米空旷。`:`目标：前方${p.forward}m，侧向${p.lateral}m；速度${p.max_vx}m/s，时长${p.duration}s。普通执行仍限0.5m / 0.10m/s / 5s。`;
  const ground=!!real?.calibration?.ground_checked_this_session;
  $("executeReadiness").textContent=!ground?'先到参数页完成本次静态地面复核。':take&&!rc?.ready?'手柄数据须新鲜且归中；尚不允许启动。':take&&!rc?.input_verified?'两分钟内未检测到手柄操作，请先只读核对输入并归中。':'确认现场条件后手动启动；程序不会自动重试。';
  const acknowledged=$("operatorConfirmed").checked&&(real?.calibration?.approved_this_session||$("preliminaryAck").checked);
  $("confirmExecute").disabled=!real?.inputs_fresh||real?.task?.outcome==='running'||!ground||!acknowledged||
    (take&&(!rc?.ready||!rc?.input_verified||!$("remoteVerified").checked||p.duration<4||p.duration>8||!Number.isFinite(p.duration)));
}
$("openExecute").addEventListener('click',()=>{
  $("operatorConfirmed").checked=$("preliminaryAck").checked=false;
  $("remoteVerified").checked=false;
  $("preliminaryRow").hidden=!!state.status?.real?.calibration?.approved_this_session;
  updateExecuteReadiness();
  $("executeDialog").showModal();
});
for(const id of ['executeProfile','takeoverDuration','operatorConfirmed','preliminaryAck','remoteVerified'])$(id).addEventListener('change',updateExecuteReadiness);
$("cancelExecute").addEventListener('click',()=>$("executeDialog").close());
$("confirmExecute").addEventListener('click',()=>startReal('execute'));
$("saveCalibration").addEventListener('click',async()=>{
  try{await api('/api/real/calibration/save','POST',{forward_m:Number($("calForward").value),left_m:Number($("calLeft").value),yaw_deg:Number($("calYaw").value),physically_measured:$("xyMeasured").checked,yaw_confirmed:$("yawConfirmed").checked});await refreshStatus();toast('测量已保存，完整标定需要重新复核。');}catch(error){toast(error.message,true);}
});
$("fitGround").addEventListener('click',async()=>{
  try{await api('/api/real/calibration/fit','POST',{});await refreshStatus();toast('正在复核静态地面，不发送运动指令');}catch(error){toast(error.message,true);}
});
$("approveCalibration").addEventListener('click',async()=>{
  try{await api('/api/real/calibration/approve','POST',{confirmed:true});await refreshStatus();toast('本次完整标定已确认');}catch(error){toast(error.message,true);}
});
function leaveExecution(){
  if(state.lease){state.lease=null;navigator.sendBeacon('/api/real/stop',new Blob(['{}'],{type:'application/json'}));}
}
window.addEventListener('pagehide',leaveExecution);
document.addEventListener('visibilitychange',()=>{if(document.hidden)leaveExecution();});
function currentRow() {
  if(state.mode==='live')return state.status?.real?.current;
  const entry = currentEntry();
  if (!entry) return null;
  return state.mode === "simulation" ? entry.current || entry.rows.at(-1) : entry.rows[state.index];
}
function renderTelemetry() {
  const entry = currentEntry(), row = currentRow();
  const simulation = state.mode === "simulation";
  $("metricSource").textContent = simulation ? "离线生成数据" : "历史实机数据";
  $("metricSourceFoot").textContent = simulation ? "流程模型 · 无物理执行" : entry?.title || "等待历史数据";
  $("metricPlanLabel").textContent = state.mode==='live'?'规划耗时 · 单帧':'规划耗时 · P95';
  $("metricPlan").textContent = simulation ? "—" : fixed(entry?.summary?.plan_ms_p95,1);
  $("metricPlanFoot").textContent = simulation ? "未运行真实 FMM / MPPI" : `平均 ${fixed(entry?.summary?.plan_ms_mean,1)} ms · 历史统计`;
  $("metricProposed").textContent = fixed(row?.proposed_vx,3);
  $("metricActual").textContent = fixed(row?.state?.vx,3);
  $("metricSpeedFoot").textContent = simulation ? "生成的建议 / 模拟状态速度" : `历史帧 ${row?.frame ?? "—"} · 非当前机器人状态`;
  $("motionState").textContent = simulation ? "模拟 / 无真实步态" : `${row?.state?.mode ?? "—"} / ${row?.state?.gait_type ?? "未记录"}`;
  const labels = {ok:"通过计算检查", insufficient_front_depth:"前方深度不足", insufficient_depth:"有效深度不足", unreachable:"目标不可达", near_depth_stop:"近障拦截", stale_input:"输入过期", command_footprint_collision:"足迹碰撞拦截", simulated:"模拟生成"};
  $("reasonValue").textContent = simulation ? "未发送 · 模拟" : `${row?.sent ? "曾写管道" : "未发送"} · ${labels[row?.reason] || row?.reason || "—"}`;
  $("reasonValue").title = entry?.note || "";
  $("ageValue").textContent = simulation ? "—" : `${fixed(row?.capture_age_upper_ms,1)} ms`;
  $("chartMode").textContent = simulation ? "生成数据" : "历史记录";
  $("timeline").value = state.index;
  const total = entry?.rows?.at(-1)?.elapsed || entry?.duration || 0;
  $("replayTime").textContent = `${fixed(row?.elapsed || 0,1)} / ${fixed(total,1)} s`;
  if(state.mode==='live'){
    const real=state.status?.real,task=real?.task,actual=real?.state;
    $("metricSource").textContent='实机实时';$("metricSourceFoot").textContent=task?.outcome==='running'?'双层代价 FMM + MPPI':'相机 / 状态实时，规划尚未运行或已停止';
    $("metricPlan").textContent=fixed(row?.plan_ms,1);$("metricPlanFoot").textContent=task?.outcome==='running'?'当前帧耗时 · 不使用历史统计':'上次真实规划帧 · 非当前控制输出';
    $("metricProposed").textContent=task?.outcome==='running'?fixed(row?.proposed_vx,3):'—';
    $("metricActual").textContent=real?.connected?fixed(actual?.vx,3):'—';
    $("metricSpeedFoot").textContent=task?.kind==='execute'?'建议速度 / 当前实测速度':'影子建议 / 当前实测速度';
    $("motionState").textContent=real?.connected?`${actual.mode} / ${actual.gait_type}`:'状态未连接或已过期';
    $("reasonValue").textContent=(row?.sent?'已写速度桥 · ':'未发速度 · ')+(labels[row?.reason]||row?.reason||'等待规划');
    $("ageValue").textContent=`${fixed(row?.capture_age_upper_ms,1)} ms`;
    $("chartMode").textContent='真实遥测';$("replayTime").textContent=`${fixed(task?.duration,1)} s · 实时`;
  }
}

function setupCanvas(canvas) {
  const box = canvas.getBoundingClientRect();
  const width = Math.max(1,box.width), height=Math.max(1,box.height), dpr=window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(width*dpr) || canvas.height !== Math.round(height*dpr)) {
    canvas.width=Math.round(width*dpr); canvas.height=Math.round(height*dpr);
  }
  const ctx=canvas.getContext("2d"); ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,width,height);
  return {ctx,width,height};
}
function simulationScene() {
  const parameters=currentEntry()?.parameters || {forward:Number($("forward").value)||.5,lateral:Number($("lateral").value)||0};
  const occupied=[];
  for (const [r0,c0,dr,dc] of [[43,34,9,6],[65,78,8,10],[90,48,7,7]]) {
    for(let r=r0;r<r0+dr;r++) for(let c=c0;c<c0+dc;c++) occupied.push([r,c]);
  }
  const path=Array.from({length:20},(_,i)=>[parameters.forward*i/19,parameters.lateral*i/19]);
  return {occupied,path,mppi:[],soft_cost:[],shape:[120,120],resolution:.05};
}
function drawMap() {
  const bounds=$("bevCanvas").getBoundingClientRect();
  if(bounds.width<80 || bounds.height<60)return;
  const simulation=state.mode === "simulation";
  const scene=state.mode==='live'?state.status?.real?.scene:simulation ? simulationScene() : state.selected?.scene;
  const {ctx,width,height}=setupCanvas($("bevCanvas"));
  $("mapEmpty").hidden=Boolean(scene);
  $("mapMode").textContent=state.mode==='live'?(state.status?.real?.task?.outcome==='running'?'真实局部感知与规划':'实机上次规划帧 / 尚未规划'):simulation ? "生成场景 · 非算法规划输出" : "历史定格快照";
  document.querySelector(".map-legend").children[1].lastChild.textContent=simulation ? "模拟目标线" : "FMM 路径";
  document.querySelector(".map-legend").children[2].lastChild.textContent=simulation ? "无真实候选" : "MPPI 候选";
  if (!scene){$("mapEmpty").textContent=state.mode==='live'?'请先启动真实影子规划':'该实验没有留存地图快照';$("mapDetail").textContent='等待规划数据';return;}
  const scale=Math.min((height-46)/6,(width-70)/6), originX=width/2,originY=height-25;
  state.mapTransform={scale,originX,originY};
  const x = left => originX-left*scale, y=forward=>originY-forward*scale;
  ctx.fillStyle="#f7faff";ctx.fillRect(0,0,width,height);
  const grid=scale/2;
  ctx.lineWidth=.6;ctx.strokeStyle="#d5e0ed88";
  for(let px=originX%grid;px<width;px+=grid){ctx.beginPath();ctx.moveTo(px,0);ctx.lineTo(px,height);ctx.stroke();}
  for(let py=originY%grid;py<height;py+=grid){ctx.beginPath();ctx.moveTo(0,py);ctx.lineTo(width,py);ctx.stroke();}
  ctx.strokeStyle="#b9c9ddaa";ctx.setLineDash([3,5]);ctx.strokeRect(x(3),y(6),6*scale,6*scale);ctx.setLineDash([]);
  if(state.layer === "cost" && scene.soft_cost?.length){
    const cost=scene.soft_cost, cell=scene.resolution*2*scale;
    for(let r=0;r<cost.length;r++)for(let c=0;c<cost[r].length;c++){
      const v=cost[r][c];if(v<.01)continue;
      ctx.fillStyle=`rgba(236,${Math.round(192-35*v)},${Math.round(110-55*v)},${.10+.60*v})`;
      ctx.fillRect(x(c*.1-3)-cell,y(r*.1)-cell,cell+1,cell+1);
    }
  }
  ctx.fillStyle=simulation ? "#8b9fb1" : "#ac7c43";
  for(const [r,c] of scene.occupied){const cell=scene.resolution*scale;ctx.fillRect(x((c+.5)*scene.resolution-3)-cell/2,y((r+.5)*scene.resolution)-cell/2,Math.max(2,cell),Math.max(2,cell));}
  function line(points,color,widthLine,dashed=false){if(!points?.length)return;ctx.beginPath();points.forEach((p,i)=>{const px=x(p[1]),py=y(p[0]);if(i)ctx.lineTo(px,py);else ctx.moveTo(px,py);});ctx.lineWidth=widthLine;ctx.strokeStyle=color;ctx.setLineDash(dashed?[4,4]:[]);ctx.stroke();ctx.setLineDash([]);}
  line(scene.path,"#3477d5",2.2,simulation);line(scene.mppi,"#149b82",2.5);
  let row=currentRow();const rx=simulation?(row?.state?.x || 0):0,ry=simulation?(row?.state?.y || 0):0;
  ctx.save();ctx.translate(x(ry),y(rx));
  ctx.shadowBlur=12;ctx.shadowColor="#4789e733";ctx.fillStyle="#d8e8ff";ctx.strokeStyle="#397ac9";ctx.lineWidth=1.3;
  const bodyW=.42*scale,bodyH=.72*scale;ctx.fillRect(-bodyW/2,-bodyH/2,bodyW,bodyH);ctx.strokeRect(-bodyW/2,-bodyH/2,bodyW,bodyH);
  ctx.shadowBlur=0;ctx.fillStyle="#397ac9";ctx.beginPath();ctx.moveTo(0,-bodyH/2-8);ctx.lineTo(-4,-bodyH/2-2);ctx.lineTo(4,-bodyH/2-2);ctx.closePath();ctx.fill();ctx.restore();
  const goal=scene.path?.at(-1);
  if(goal){ctx.strokeStyle="#4879b9aa";ctx.lineWidth=1;ctx.beginPath();ctx.arc(x(goal[1]),y(goal[0]),5,0,Math.PI*2);ctx.stroke();ctx.fillStyle="#507096";ctx.font="9px sans-serif";ctx.fillText("目标",x(goal[1])+10,y(goal[0])+3);}
  if(state.mode==='live'&&state.status?.real?.task?.outcome!=='running'){
    const pending=[Number($("forward").value),Number($("lateral").value)];
    ctx.strokeStyle='#118a73';ctx.setLineDash([3,2]);ctx.strokeRect(x(pending[1])-5,y(pending[0])-5,10,10);ctx.setLineDash([]);
    ctx.fillStyle='#118a73';ctx.fillText('待启动目标',x(pending[1])+9,y(pending[0])-5);
  }
  ctx.fillStyle="#71839a";ctx.font="9px monospace";
  for(let i=1;i<=6;i++)ctx.fillText(i+"m",x(-3)+8,y(i)+3);
  $("mapDetail").textContent=`${scene.shape[0]} × ${scene.shape[1]} · ${scene.occupied.length} cells`;
}
$("bevCanvas").title='实机模式可点选下次相对目标；不会立即移动，运行中目标固定。';
$("bevCanvas").addEventListener('click',event=>{
  if(state.mode!=='live'||!state.mapTransform)return;
  if(state.status?.real?.task?.outcome==='running'){toast('运行中目标固定；请先停止，再设置新目标');return;}
  const box=$("bevCanvas").getBoundingClientRect(),t=state.mapTransform;
  const forward=Math.round((t.originY-(event.clientY-box.top))/t.scale*20)/20;
  const lateral=Math.round((t.originX-(event.clientX-box.left))/t.scale*20)/20;
  if(forward<.3||forward>5||Math.abs(lateral)>2){toast('请选择前方0.3–5m、左右±2m内的局部目标',true);return;}
  $("forward").value=forward;$("lateral").value=lateral;
  $("draftStatus").textContent='点选目标已填入；下次启动才生效';drawMap();toast(`待启动目标：前 ${forward}m，左 ${lateral}m；未发送指令`);
});
function drawSpeed() {
  const bounds=$("speedCanvas").getBoundingClientRect();
  if(bounds.width<40 || bounds.height<40)return;
  const {ctx,width,height}=setupCanvas($("speedCanvas"));
  const entry=currentEntry(), rows=entry?.rows || [];
  const left=31,right=width-9,top=12,bottom=height-19;
  const max=Math.max(.12,...rows.map(r=>Number(r.proposed_vx)||0))+.02,min=-.025;
  const total=Math.max(1,entry?.duration || rows.at(-1)?.elapsed || 1);
  const x=t=>left+(right-left)*t/total,y=v=>bottom-(v-min)/(max-min)*(bottom-top);
  ctx.font="8px monospace";ctx.fillStyle="#72849c";ctx.strokeStyle="#d8e2ef";ctx.lineWidth=.6;
  for(const v of [0,.05,.1,.2]){if(v>max)continue;ctx.beginPath();ctx.moveTo(left,y(v));ctx.lineTo(right,y(v));ctx.stroke();ctx.fillText(v.toFixed(2),0,y(v)+3);}
  ctx.fillText("0s",left,bottom+14);ctx.fillText(fixed(total,0)+"s",right-17,bottom+14);
  for(const [field,color] of [["proposed_vx","#3477d5"],["actual","#149b82"]]){
    ctx.beginPath();let begun=false;
    for(const row of rows){const v=field === "actual"?row.state?.vx:row[field];if(!Number.isFinite(v))continue;const px=x(row.elapsed),py=y(v);if(!begun){ctx.moveTo(px,py);begun=true;}else ctx.lineTo(px,py);}
    ctx.strokeStyle=color;ctx.lineWidth=1.6;ctx.stroke();
  }
  const row=currentRow();if(row){ctx.beginPath();ctx.moveTo(x(row.elapsed),top);ctx.lineTo(x(row.elapsed),bottom);ctx.setLineDash([2,3]);ctx.strokeStyle="#a6b9cf77";ctx.stroke();ctx.setLineDash([]);}
}
function drawAll(){drawMap();drawSpeed();}
document.querySelectorAll("[data-layer]").forEach(button=>button.addEventListener("click",()=>{state.layer=button.dataset.layer;document.querySelectorAll("[data-layer]").forEach(b=>b.classList.toggle("active",b===button));drawMap();}));
new ResizeObserver(drawAll).observe($("bevCanvas").parentElement);
window.addEventListener("resize",drawAll);
function animate(now){
  if(state.playing && state.mode === "history" && state.selected){
    state.replayElapsed+=(now-state.lastTick)/1000*state.rate;state.lastTick=now;
    const rows=state.selected.rows;let changed=false;
    while(state.index<rows.length-1 && rows[state.index+1].elapsed<=state.replayElapsed){state.index++;changed=true;}
    if(state.index===rows.length-1){state.playing=false;$("playReplay").textContent="▶";}
    if(changed){renderTelemetry();drawSpeed();}
  }
  requestAnimationFrame(animate);
}
requestAnimationFrame(animate);
function updateClock(){ $("clock").textContent = new Date().toLocaleTimeString("zh-CN",{hour12:false}); }
updateClock();setInterval(updateClock,1000);
async function poll(){try{
  if(state.lease){
    if(document.hidden||state.mode!=='live'||document.body.dataset.view!=='workspace')leaveExecution();
    else await api('/api/real/heartbeat','POST',{lease:state.lease});
  }
  await refreshStatus();
}catch(error){leaveExecution();$("modeNotice").textContent="本地服务连接中断";$("modeDescription").textContent="服务不可达，画面可能定格；执行心跳已停止。请现场接管。";$("runtimeBadge").textContent="服务失联";$("rgbSourceBadge").textContent="服务失联 · 定格";$("depthSourceBadge").textContent="服务失联 · 定格";document.querySelectorAll(".camera-panel").forEach(panel=>panel.classList.add("stream-stale"));}setTimeout(poll,200);}
async function init(){
  try {
    await refreshStatus();applyParams(state.status.draft);await refreshExperiments();
    const first=state.experiments.find(e=>e.id==="ours_ready_shadow_20261030") || state.experiments.find(e=>e.source==="historical");
    if(first)await loadExperiment(first.id);else setMode("simulation");
    if(state.status.real?.enabled){$("duration").value=5;setMode('live');updateReal();}
  }catch(error){toast("初始化失败："+error.message,true);}
  poll();
}
init();

$("captureCamera").addEventListener('click',async()=>{
  state.capturing=true;$("captureCamera").disabled=true;
  try{const result=await api('/api/camera/capture','POST',{});
    $("captureCamera").title='上次保存：'+result.directory;
    if(result.download){const link=document.createElement('a');link.href=result.download;link.download='GO2_RGBD.zip';link.click();}
    toast('RGB 和深度已保存：'+result.directory);
  }catch(error){toast(error.message,true);}
  finally{state.capturing=false;$("captureCamera").disabled=!state.status?.camera?.connected;}
});

// Independent, non-overlapping image requests: telemetry polling does not cap video FPS.
let previewEtag=null, displayedTimes=[];
async function refreshVideo(){
  const started=performance.now();
  try{
    if(!document.hidden && state.status?.camera?.connected){
      const response=await fetch('/api/camera/preview',{cache:'no-store',headers:previewEtag?{'If-None-Match':previewEtag}:{},signal:AbortSignal.timeout(1500)});
      if(response.status===200){
        const pair=await response.json();
        if(pair.images.rgb&&pair.images.depth){
          const decoded=await Promise.all(['rgb','depth'].map(async kind=>{const im=new Image();im.src=pair.images[kind];await im.decode();return im;}));
          if(state.status?.camera?.connected){
            ['rgb','depth'].forEach((kind,i)=>{$(kind+'Image').src=decoded[i].src;$(kind+'Image').classList.remove('no-image');});
            state.cameraPreview=pair.frame;previewEtag=response.headers.get('ETag');
            const now=performance.now();displayedTimes.push(now);displayedTimes=displayedTimes.filter(t=>now-t<2000);
            state.displayFps=displayedTimes.length>1?1000*(displayedTimes.length-1)/(now-displayedTimes[0]):0;
          }
        }
      }
    }else{previewEtag=null;displayedTimes=[];state.displayFps=0;}
  }catch(error){/* Connection state and stale labels are handled by telemetry polling. */}
  setTimeout(refreshVideo,Math.max(0,1000/15-(performance.now()-started)));
}
refreshVideo();
