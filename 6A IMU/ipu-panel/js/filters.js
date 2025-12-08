// filters.js - client-side smoothing helpers
import * as THREE from 'https://unpkg.com/three@0.154.0/build/three.module.js';
export default function Filters() {
  function applySlerp(prevQ, qArr, alpha=0.08) {
    if (!prevQ) return qArr;
    const qa = new THREE.Quaternion(prevQ[0], prevQ[1], prevQ[2], prevQ[3]);
    const qb = new THREE.Quaternion(qArr[0], qArr[1], qArr[2], qArr[3]);
    qa.slerp(qb, alpha);
    return [qa.x, qa.y, qa.z, qa.w];
  }
  return { applySlerp };
}
