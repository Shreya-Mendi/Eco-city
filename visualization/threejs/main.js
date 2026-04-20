import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CityRenderer } from './cityRenderer.js';
import { setupControls } from './controls.js';

const canvas = document.getElementById('canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x070714);
scene.fog = new THREE.Fog(0x070714, 80, 320);

const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 800);
camera.position.set(42, 38, 48);
camera.lookAt(0, 4, 0);

const orbit = new OrbitControls(camera, renderer.domElement);
orbit.enableDamping = true;

const ambient = new THREE.AmbientLight(0xffffff, 0.42);
scene.add(ambient);
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

fetch('city_data.json')
  .then((r) => {
    if (!r.ok) throw new Error(`Failed to load city_data.json (${r.status})`);
    return r.json();
  })
  .then((data) => {
    cityRenderer.setData(data);
    const history = cityRenderer.getHistory();
    setupControls(data, cityRenderer, updateStats);
    cityRenderer.renderAt(0);
    if (history.length) {
      updateStats(history[0]);
    }
    const slider = document.getElementById('step-slider');
    slider.max = Math.max(0, history.length - 1);
  })
  .catch((err) => {
    document.getElementById('stats').innerHTML = `<div style="color:#f88">Load error: ${err.message}</div>`;
  });

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
