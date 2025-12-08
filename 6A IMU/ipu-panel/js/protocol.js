// protocol.js - parse lines and build commands
export default function Protocol() {
  function parseLine(line) {
    try {
      const obj = JSON.parse(line);
      if (obj && Array.isArray(obj.q) && obj.q.length >= 4) {
        // ensure numbers
        const q = obj.q.map(Number);
        const t = obj.t || Date.now();
        return { q, t };
      }
    } catch (e) { return null; }
    return null;
  }

  function cmdCalGyro() { return 'CAL_GYRO'; }
  function cmdSetFreq(n) { return `SET_FREQ:${n}`; }
  function cmdSetBeta(b) { return `SET_BETA:${b}`; }
  function cmdInfo() { return 'INFO'; }
  function cmdSave() { return 'SAVE'; }
  function cmdLoad() { return 'LOAD'; }

  return { parseLine, cmdCalGyro, cmdSetFreq, cmdSetBeta, cmdInfo, cmdSave, cmdLoad };
}
