import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CityRenderer, COLORS, ZONE_NAMES } from './cityRenderer.js';
import { setupControls } from './controls.js';

const canvas = document.getElementById('canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0f1a);
scene.fog = new THREE.Fog(0x0b0f1a, 110, 380);

const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 800);

const orbit = new OrbitControls(camera, renderer.domElement);
orbit.enableDamping = true;
orbit.minDistance = 18;
orbit.maxDistance = 220;
orbit.maxPolarAngle = Math.PI * 0.49;

function frameScene(extent) {
  const r = extent * 1.8;
  camera.position.set(r * 0.75, r * 0.55, r * 0.9);
  orbit.target.set(0, extent * 0.08, 0);
  orbit.update();
}

frameScene(20);

const ambient = new THREE.AmbientLight(0xdfe6ff, 0.55);
scene.add(ambient);

const hemi = new THREE.HemisphereLight(0xe5eeff, 0x2a3b1a, 0.35);
scene.add(hemi);
const sun = new THREE.DirectionalLight(0xfff4e0, 1.35);
sun.position.set(55, 85, 35);
sun.castShadow = true;
sun.shadow.mapSize.width = 2048;
sun.shadow.mapSize.height = 2048;
sun.shadow.camera.near = 10;
sun.shadow.camera.far = 220;
sun.shadow.camera.left = -70;
sun.shadow.camera.right = 70;
sun.shadow.camera.top = 70;
sun.shadow.camera.bottom = -70;
scene.add(sun);

const cityRenderer = new CityRenderer(scene);

function updateStats(snapshot) {
  document.getElementById('stats').innerHTML = `
    <div>Step: ${snapshot.step}</div>
    <div>Population: ${snapshot.population}</div>
    <div>Pollution: ${snapshot.pollution.toFixed(2)}</div>
    <div>Traffic: ${snapshot.traffic.toFixed(2)}</div>
    <div>Energy Balance: ${snapshot.energy_balance.toFixed(2)}</div>
    <div>Reward: ${snapshot.reward.toFixed(3)}</div>
  `;
}

function showError(msg) {
  document.getElementById('stats').innerHTML = `<div style="color:#f88">${msg}</div>`;
}

function loadRollout(url) {
  return fetch(url)
    .then((r) => {
      if (!r.ok) throw new Error(`Failed to load ${url} (${r.status})`);
      return r.json();
    })
    .then((data) => {
      cityRenderer.setData(data);
      const history = cityRenderer.getHistory();
      setupControls(data, cityRenderer, updateStats);
      cityRenderer.renderAt(0);
      if (history.length) updateStats(history[0]);
      const slider = document.getElementById('step-slider');
      slider.max = Math.max(0, history.length - 1);
      frameScene(cityRenderer.worldExtent);
    });
}

function buildZoneLegend() {
  const el = document.getElementById('legend');
  if (!el) return;
  const order = [1, 2, 3, 4, 5, 6, 0];
  el.innerHTML = order
    .map((z) => {
      const hex = `#${COLORS[z].toString(16).padStart(6, '0')}`;
      return `<div class="lg-row"><span class="lg-swatch" style="background:${hex}"></span>${ZONE_NAMES[z]}</div>`;
    })
    .join('');
}
buildZoneLegend();

function populateRolloutPicker(index) {
  const picker = document.getElementById('rollout-picker');
  const select = document.getElementById('rollout-select');
  select.innerHTML = '';
  index.forEach((item) => {
    const opt = document.createElement('option');
    opt.value = `rollouts/${item.file}`;
    opt.textContent = `#${item.rank + 1}  reward=${item.total_reward.toFixed(2)}  seed=${item.seed}`;
    select.appendChild(opt);
  });
  picker.style.display = index.length > 1 ? 'block' : 'none';
  select.addEventListener('change', (e) => {
    loadRollout(e.target.value).catch((err) => showError(`Load error: ${err.message}`));
  });
}

fetch('rollouts/index.json')
  .then((r) => (r.ok ? r.json() : null))
  .then((index) => {
    if (Array.isArray(index) && index.length) {
      populateRolloutPicker(index);
      return loadRollout(`rollouts/${index[0].file}`);
    }
    document.getElementById('rollout-picker').style.display = 'none';
    return loadRollout('city_data.json');
  })
  .catch((err) => showError(`Load error: ${err.message}`));

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

function animate() {
  requestAnimationFrame(animate);
  orbit.update();
  renderer.render(scene, camera);
}
animate();
