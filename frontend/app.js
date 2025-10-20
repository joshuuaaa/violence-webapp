const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const startBtn = document.getElementById('startBtn');
const stopBtn  = document.getElementById('stopBtn');
const statusPill = document.getElementById('statusPill');
const fpsSlider = document.getElementById('fps');
const fpsLabel = document.getElementById('fpsLabel');
const violenceThreshold = document.getElementById('violenceThreshold');
const sequenceLength = document.getElementById('sequenceLength');
const fireThreshold = document.getElementById('fireThreshold');
const recordingEnabled = document.getElementById('recordingEnabled');
const saveConfigBtn = document.getElementById('saveConfig');
const configSaved = document.getElementById('configSaved');
const violenceScoreEl = document.getElementById('violenceScore');
const alertTypesEl = document.getElementById('alertTypes');
const beep = document.getElementById('beep');

let ws = null;
let streamInterval = null;
let pingInterval = null;
let reconnectTimer = null;
let fps = parseInt(fpsSlider.value, 10);
fpsLabel.textContent = fps;

function setStatus(text, theme = 'idle') {
  statusPill.textContent = text;
  if (theme === 'live') statusPill.style.background = '#dcfce7';
  else if (theme === 'alert') statusPill.style.background = '#fee2e2';
  else statusPill.style.background = '#f1f5f9';
}

async function loadConfig() {
  const res = await fetch('/api/config');
  const cfg = await res.json();
  fpsSlider.value = cfg.fps ?? 5;
  fpsLabel.textContent = cfg.fps ?? 5;
  violenceThreshold.value = cfg.violence_threshold ?? 0.6;
  sequenceLength.value = cfg.violence_sequence_length ?? 16;
  fireThreshold.value = cfg.fire_threshold ?? 0.4;
  recordingEnabled.checked = cfg.recording_enabled ?? true;
}
loadConfig().catch(console.error);

saveConfigBtn.onclick = async () => {
  const body = {
    fps: parseInt(fpsSlider.value, 10),
    enable_violence: true,
    enable_fire: true,
    violence_sequence_length: parseInt(sequenceLength.value, 10),
    violence_threshold: parseFloat(violenceThreshold.value),
    fire_threshold: parseFloat(fireThreshold.value),
    recording_enabled: recordingEnabled.checked,
    pre_roll_seconds: 3,
    post_roll_seconds: 5,
    recordings_dir: "recordings",
  };
  await fetch('/api/config', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  });
  configSaved.style.display = 'block';
  setTimeout(() => configSaved.style.display = 'none', 1500);
};

fpsSlider.oninput = () => {
  fps = parseInt(fpsSlider.value, 10);
  fpsLabel.textContent = fps;
};

async function start() {
  if (streamInterval) return;
  setStatus('Starting...', 'idle');

  try {
    const media = await navigator.mediaDevices.getUserMedia({ 
      video: { 
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: 'user'
      }, 
      audio: false 
    });
    video.srcObject = media;
    await video.play();
  } catch (err) {
    console.error('Camera access error:', err);
    alert('Camera access denied or not available: ' + err.message);
    setStatus('Camera Error', 'idle');
    return;
  }

  canvas.width = 640;
  canvas.height = 480;
  const ctx = canvas.getContext('2d');

  const wsProto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${wsProto}://${location.host}/ws/video`);
  
  ws.onopen = () => {
    console.log('WebSocket connected');
    setStatus('Live', 'live');
  };
  
  ws.onclose = () => {
    console.log('WebSocket closed');
    setStatus('Disconnected');
    if (pingInterval) { clearInterval(pingInterval); pingInterval = null; }
    if (!reconnectTimer) {
      reconnectTimer = setTimeout(() => { reconnectTimer = null; start(); }, 1500);
    }
  };
  
  ws.onerror = (err) => {
    console.error('WebSocket error:', err);
    setStatus('Connection Error', 'alert');
  };

  ws.onmessage = (evt) => {
    const data = JSON.parse(evt.data);
    // Draw server-rendered overlay preview if sent
    if (data.preview) {
      const img = new Image();
      img.onload = () => {
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      };
      // Append a dummy query to ensure no caching artefacts in some browsers
      img.src = data.preview + `#${Date.now()}`;
    }
    // Always also draw the live raw video as a safety to avoid stale previews
    try { ctx.drawImage(video, 0, 0, canvas.width, canvas.height); } catch {}

    const v = data.violence_score;
    violenceScoreEl.textContent = v === null || v === undefined ? '-' : v.toFixed(2);
    alertTypesEl.textContent = data.alert ? (data.alert_types || []).join(', ') : '-';

    // If model needs more frames to start accurate predictions, indicate that
    if ((data.violence_buffer ?? 0) < (data.violence_T ?? 16)) {
      statusPill.textContent = `Warming (${data.violence_buffer ?? 0}/${data.violence_T ?? 16})`;
    }

    if (data.alert) {
      setStatus('Alert', 'alert');
      playBeep();
    } else {
      setStatus('Live', 'live');
    }

    // Update dashboard periodically
    maybeRefreshDashboard();
  };

  // Send frames at configured FPS
  const sendFrame = () => {
    // Check if video is still playing
    if (video.readyState !== video.HAVE_ENOUGH_DATA) {
      console.warn('Video not ready yet');
      return;
    }
    
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      console.warn('WebSocket not ready, state:', ws ? ws.readyState : 'null');
      return;
    }
    
    try {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((blob) => {
        if (!blob) {
          console.error('Failed to create blob from canvas');
          return;
        }
        const reader = new FileReader();
        reader.onloadend = () => {
          try {
            ws.send(JSON.stringify({ frame: reader.result }));
          } catch (err) {
            console.error('Failed to send frame:', err);
          }
        };
        reader.onerror = (err) => console.error('FileReader error:', err);
        reader.readAsDataURL(blob);
      }, 'image/jpeg', 0.8);
    } catch (err) {
      console.error('Error in sendFrame:', err);
    }
  };
  
  streamInterval = setInterval(sendFrame, 1000 / fps);
  pingInterval = setInterval(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      try { ws.send(JSON.stringify({ type: 'ping', t: Date.now() })); } catch {}
    }
  }, 10000);
  console.log('Started sending frames at', fps, 'FPS');
}

function stop() {
  if (streamInterval) clearInterval(streamInterval);
  streamInterval = null;
  if (pingInterval) clearInterval(pingInterval);
  pingInterval = null;
  if (ws) ws.close();
  ws = null;
  if (video.srcObject) {
    video.srcObject.getTracks().forEach(t => t.stop());
    video.srcObject = null;
  }
  setStatus('Stopped');
}

startBtn.onclick = start;
stopBtn.onclick = stop;

function playBeep() {
  // Simple beep using Web Audio API
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const o = ctx.createOscillator();
  const g = ctx.createGain();
  o.type = 'sine';
  o.frequency.value = 880;
  o.connect(g);
  g.connect(ctx.destination);
  g.gain.setValueAtTime(0.1, ctx.currentTime);
  o.start();
  o.stop(ctx.currentTime + 0.2);
}

// Dashboard
let lastStatsRefresh = 0;
async function maybeRefreshDashboard() {
  const now = Date.now();
  if (now - lastStatsRefresh < 3000) return;
  lastStatsRefresh = now;

  const [eventsRes, statsRes] = await Promise.all([
    fetch('/api/events'),
    fetch('/api/stats'),
  ]);
  const [events, stats] = [await eventsRes.json(), await statsRes.json()];
  renderEvents(events);
  renderChart(stats);
}

function renderEvents(events) {
  const tbody = document.getElementById('eventsTbody');
  tbody.innerHTML = '';
  events.forEach(ev => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${new Date(ev.timestamp).toISOString().replace('T',' ').slice(0,19)}</td>
      <td>${ev.event_type}</td>
      <td>${ev.confidence.toFixed(2)}</td>
  <td>${ev.snapshot_path ? `<a href="/${ev.snapshot_path}" target="_blank">view</a>` : '-'}</td>
  <td>${ev.video_path ? `<a href="/${ev.video_path}" target="_blank">mp4</a>` : '-'}</td>
    `;
    tbody.appendChild(tr);
  });
}

let chartRef = null;
function renderChart(stats) {
  const ctx = document.getElementById('chart').getContext('2d');
  const labels = Object.keys(stats.by_day).sort();
  const data = labels.map(k => stats.by_day[k]);
  if (chartRef) {
    chartRef.data.labels = labels;
    chartRef.data.datasets[0].data = data;
    chartRef.update();
    return;
  }
  chartRef = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Incidents per day',
        data,
        borderColor: '#ef4444',
        backgroundColor: 'rgba(239,68,68,.2)',
        tension: .2
      }]
    },
    options: {
      responsive: true,
      scales: {
        y: { beginAtZero: true, ticks: { precision: 0 } }
      }
    }
  });
}