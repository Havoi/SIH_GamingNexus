// ui.js - DOM wiring & callbacks
export default function UI({ serial, proto, logger }) {
  const btnConnect = document.getElementById('btnConnect');
  const btnDisconnect = document.getElementById('btnDisconnect');
  const btnCal = document.getElementById('btnCal');
  const btnSave = document.getElementById('btnSave');
  const btnLoad = document.getElementById('btnLoad');
  const inpFreq = document.getElementById('inpFreq');
  const srVal = document.getElementById('srVal');
  const inpBeta = document.getElementById('inpBeta');
  const betaVal = document.getElementById('betaVal');
  const quatBox = document.getElementById('quatBox');
  const eulerBox = document.getElementById('eulerBox');
  const connectState = document.getElementById('connectState');

  let onConnectCb = ()=>{}, onDisconnectCb = ()=>{};

  btnConnect.addEventListener('click', () => onConnectCb());
  btnDisconnect.addEventListener('click', () => onDisconnectCb());
  btnCal.addEventListener('click', async () => {
    try { await serial.send(proto.cmdCalGyro()); logger.log('Sent CAL_GYRO'); } catch(e){ logger.log('send failed:'+e); }
  });
  btnSave.addEventListener('click', async () => { await serial.send(proto.cmdSave()); logger.log('SAVE sent'); });
  btnLoad.addEventListener('click', async () => { await serial.send(proto.cmdLoad()); logger.log('LOAD sent'); });

  inpFreq.addEventListener('input', ()=> { srVal.textContent = inpFreq.value; });
  inpBeta.addEventListener('input', ()=> { betaVal.textContent = Number(inpBeta.value).toFixed(3); });

  // helper update functions
  function updateTelemetry(q, t) {
    quatBox.textContent = `q: [${q.map(v=>v.toFixed(6)).join(', ')}]`;
    // compute Euler (ZYX)
    const x=q[0], y=q[1], z=q[2], w=q[3];
    // compute Euler angles (ZX'Y')
    // Using standard formulas
    const ysqr = y*y;
    const t0 = +2.0*(w*x + y*z);
    const t1 = +1.0 - 2.0*(x*x + ysqr);
    const roll = Math.atan2(t0, t1);
    const t2 = +2.0*(w*y - z*x);
    const t2c = Math.min(1.0, Math.max(-1.0, t2));
    const pitch = Math.asin(t2c);
    const t3 = +2.0*(w*z + x*y);
    const t4 = +1.0 - 2.0*(ysqr + z*z);
    const yaw = Math.atan2(t3, t4);
    eulerBox.textContent = `Yaw:${(yaw*180/Math.PI).toFixed(1)}° Pitch:${(pitch*180/Math.PI).toFixed(1)}° Roll:${(roll*180/Math.PI).toFixed(1)}°`;
  }

  function setConnected(v) {
    btnConnect.disabled = v;
    btnDisconnect.disabled = !v;
    connectState.textContent = v ? 'Connected' : 'Disconnected';
    connectState.style.color = v ? '#86efac' : '#fca5a5';
  }

  function getSlerpAlpha() {
    // optional link to UI; lower = smoother
    return 0.08;
  }

  function start() {
    // placeholder in case future init needed
  }

  return { start, updateTelemetry, setConnected, onConnect(cb){ onConnectCb = cb; }, onDisconnect(cb){ onDisconnectCb = cb; }, getSlerpAlpha };
}
