// threeView.js - Three.js scene and helpers (updated for centered view & responsive canvas)
import * as THREE from 'https://unpkg.com/three@0.154.0/build/three.module.js';

export default function ThreeView({ mountId='visual', logger=console }) {
  const mount = document.getElementById(mountId);

  // Scene + camera
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x071427);

  const camera = new THREE.PerspectiveCamera(60, mount.clientWidth / mount.clientHeight, 0.1, 1000);
  // position camera so cube is centered and nicely visible
  camera.position.set(2.0, 1.6, 3.0);
  camera.lookAt(0, 0, 0);

  // Renderer
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  // We'll set size below after appending the canvas
  mount.appendChild(renderer.domElement);

  // Cube and helpers
  const geo = new THREE.BoxGeometry(1.0, 0.6, 1.6);
  const mat = new THREE.MeshStandardMaterial({ color: 0x78a1ff, metalness: 0.2, roughness: 0.35 });
  const cube = new THREE.Mesh(geo, mat);
  scene.add(cube);

  const axes = new THREE.AxesHelper(1.5);
  scene.add(axes);

  const hemi = new THREE.HemisphereLight(0xffffff, 0x444444, 0.9);
  scene.add(hemi);
  const dir = new THREE.DirectionalLight(0xffffff, 0.8);
  dir.position.set(5, 5, 5);
  scene.add(dir);

  // initial resize
  function setRendererSize() {
    const w = mount.clientWidth;
    const h = mount.clientHeight;
    // keep renderer size same as container
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  setRendererSize();

  // Recompute on window resize
  window.addEventListener('resize', () => {
    setRendererSize();
  });

  // Optional pointer orbit for quick inspection (lightweight)
  let isPointerDown = false, lastX = 0, lastY = 0, yaw = 0.0, pitch = -0.4;
  renderer.domElement.addEventListener('pointerdown', (e) => { isPointerDown = true; lastX = e.clientX; lastY = e.clientY; renderer.domElement.setPointerCapture(e.pointerId); });
  renderer.domElement.addEventListener('pointerup', (e) => { isPointerDown = false; try { renderer.domElement.releasePointerCapture(e.pointerId); } catch(e){} });
  renderer.domElement.addEventListener('pointermove', (e) => {
    if (!isPointerDown) return;
    const dx = (e.clientX - lastX) / 200;
    const dy = (e.clientY - lastY) / 200;
    yaw += dx; pitch += dy;
    lastX = e.clientX; lastY = e.clientY;
  });

  // Animation loop
  function animate() {
    requestAnimationFrame(animate);
    // gentle orbit applied to camera (keeps cube centered)
    const radius = 4.0;
    camera.position.x = Math.cos(yaw) * radius;
    camera.position.z = Math.sin(yaw) * radius;
    camera.position.y = 1.6 + Math.sin(pitch) * 0.4;
    camera.lookAt(0, 0, 0);
    renderer.render(scene, camera);
  }
  animate();

  // Public API: accept quaternion array [x,y,z,w]
  function setQuaternion(qArr) {
    if (!qArr || qArr.length < 4) return;
    cube.quaternion.set(qArr[0], qArr[1], qArr[2], qArr[3]);
  }

  return { setQuaternion };
}
