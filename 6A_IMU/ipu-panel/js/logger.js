// logger.js - central log pane and recording
export default function Logger() {
  const el = document.getElementById('log');
  const frames = []; // recorded JSON frames
  function log(s) {
    const t = new Date().toLocaleTimeString();
    el.textContent += `[${t}] ${s}\n`;
    el.scrollTop = el.scrollHeight;
  }
  function addFrame(obj) {
    frames.push(obj);
    // keep frames bounded
    if (frames.length > 20000) frames.shift();
  }
  function saveRecording(filename='imu-record.json') {
    const blob = new Blob([JSON.stringify(frames)], { type:'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename; a.click();
    URL.revokeObjectURL(url);
  }
  return { log, addFrame, saveRecording };
}
