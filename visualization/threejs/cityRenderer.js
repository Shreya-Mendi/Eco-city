import * as THREE from 'three';

const CELL_SIZE = 2;
const COLORS = {
  0: 0x222233,
  1: 0x1565c0,
  2: 0xf57f17,
  3: 0xb71c1c,
  4: 0x2e7d32,
  5: 0x424242,
  6: 0xf9a825,
};

const HEIGHTS = {
  0: 0.05,
  1: 2.0,
  2: 1.5,
  3: 2.5,
  4: 0.3,
  5: 0.05,
  6: 1.0,
};

export class CityRenderer {
  constructor(scene) {
    this.scene = scene;
    this.meshes = [];
  }

  render(snapshot) {
    this.meshes.forEach((m) => this.scene.remove(m));
    this.meshes = [];

    const grid = snapshot.grid;
    const N = grid.length;
    const offset = (N * CELL_SIZE) / 2;

    for (let row = 0; row < N; row++) {
      for (let col = 0; col < N; col++) {
        const zone = grid[row][col];
        const height = HEIGHTS[zone] ?? 0.05;
        const color = COLORS[zone] ?? 0x222233;

        const geo = new THREE.BoxGeometry(CELL_SIZE * 0.9, height, CELL_SIZE * 0.9);
        const mat = new THREE.MeshLambertMaterial({ color });
        const mesh = new THREE.Mesh(geo, mat);

        mesh.position.set(
          col * CELL_SIZE - offset + CELL_SIZE / 2,
          height / 2,
          row * CELL_SIZE - offset + CELL_SIZE / 2
        );
        mesh.castShadow = true;
        mesh.receiveShadow = true;

        this.scene.add(mesh);
        this.meshes.push(mesh);
      }
    }
  }
}
