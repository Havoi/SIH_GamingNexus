// main.js - bootstrap and wiring
import * as THREE from 'https://unpkg.com/three@0.154.0/build/three.module.js';
import Serial from './serial.js';
import Protocol from './protocol.js';
import ThreeView from './threeView.js';
import UI from './ui.js';
import Logger from './logger.js';
import Filters from './filters.js';

(async function init() {
  // initialize modules
  const logger = Logger();
  const three = ThreeView({ mountId: 'visual', logger });
  const filters = Filters();
  const proto = Protocol();
  const serial = Serial({ onLine: handleLine, logger });
  const ui = UI({ serial, proto, logger });

  let lastQuat = null;

  // called on every incoming line from serial
  function handleLine(line) {
    const parsed = proto.parseLine(line);
    if (!parsed) {
      logger.log(line); // raw log
      return;
    }
    // parsed has { q: [x,y,z,w], t: ... }
    let q = parsed.q;
    // client-side smoothing (optional)
    q = filters.applySlerp(lastQuat, q, ui.getSlerpAlpha());
    lastQuat = q;
    // apply to scene
    three.setQuaternion(q);
    ui.updateTelemetry(q, parsed.t);
    logger.addFrame(parsed);
  }

  // hook UI to actions too
  ui.onConnect(async () => {
    try {
      await serial.connect();
      ui.setConnected(true);
    } catch (e) {
      logger.log('connect failed: ' + e);
      ui.setConnected(false);
    }
  });

  ui.onDisconnect(async () => {
    await serial.disconnect();
    ui.setConnected(false);
  });

  // start UI event wiring
  ui.start();

  // expose for debug
  window.__IMU = { serial, proto, three, ui, logger, filters };

})();
