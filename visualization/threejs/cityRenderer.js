import * as THREE from 'three';

const CELL_SIZE = 2;

export const COLORS = {
  0: 0x222233,
  1: 0x1565c0,
  2: 0xf57f17,
  3: 0xb71c1c,
  4: 0x2e7d32,
  5: 0x424242,
  6: 0xf9a825,
};

export const ZONE_NAMES = {
  0: 'Empty',
  1: 'Residential',
  2: 'Commercial',
  3: 'Industrial',
  4: 'Green',
  5: 'Road',
  6: 'Energy',
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

function sampleBilinear(heights, u, v) {
  const rows = heights.length;
  const cols = heights[0].length;
  const x = Math.max(0, Math.min(cols - 1, u * (cols - 1)));
  const y = Math.max(0, Math.min(rows - 1, v * (rows - 1)));
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(x0 + 1, cols - 1);
  const y1 = Math.min(y0 + 1, rows - 1);
  const tx = x - x0;
  const ty = y - y0;
  const h00 = heights[y0][x0];
  const h01 = heights[y0][x1];
  const h10 = heights[y1][x0];
  const h11 = heights[y1][x1];
  return (
    (1 - tx) * (1 - ty) * h00 +
    tx * (1 - ty) * h01 +
    (1 - tx) * ty * h10 +
    tx * ty * h11
  );
}

export class CityRenderer {
  constructor(scene) {
    this.scene = scene;
    this.dynamicMeshes = [];
    this.staticGroup = new THREE.Group();
    this.staticGroup.name = 'ecoStatic';
    this.scene.add(this.staticGroup);
    this._legacy = false;
    this._history = [];
    this._staticData = null;
    this._terrainMaxHeight = 12;
    this.worldExtent = 20;
    this.gridSize = 10;
    this.cellSize = CELL_SIZE;
  }

  setData(raw) {
    this.clearDynamic();
    if (Array.isArray(raw)) {
      this._legacy = true;
      this._history = raw;
      this._staticData = null;
      while (this.staticGroup.children.length) {
        this.staticGroup.remove(this.staticGroup.children[0]);
      }
      return;
    }
    this._legacy = false;
    this._history = raw.history || [];
    this._staticData = raw.static;
    this.buildStaticScene(raw.static);
  }

  getHistory() {
    return this._history;
  }

  clearDynamic() {
    this.dynamicMeshes.forEach((m) => {
      this.scene.remove(m);
      if (m.geometry) m.geometry.dispose();
      if (m.material) {
        if (Array.isArray(m.material)) m.material.forEach((mm) => mm.dispose());
        else m.material.dispose();
      }
    });
    this.dynamicMeshes = [];
  }

  buildStaticScene(staticData) {
    while (this.staticGroup.children.length) {
      const ch = this.staticGroup.children[0];
      this.staticGroup.remove(ch);
      if (ch.geometry) ch.geometry.dispose();
      if (ch.material) {
        if (Array.isArray(ch.material)) ch.material.forEach((mm) => mm.dispose());
        else ch.material.dispose();
      }
    }
    if (!staticData || !staticData.terrain) return;

    const cellSize = staticData.cellSize ?? CELL_SIZE;
    const gridSize = staticData.gridSize ?? 10;
    const worldExtent = staticData.worldExtent ?? gridSize * cellSize;
    const heights = staticData.terrain.heights;
    const res = staticData.terrain.resolution ?? heights.length;
    this.cellSize = cellSize;
    this.gridSize = gridSize;
    this.worldExtent = worldExtent;

    const geo = new THREE.PlaneGeometry(worldExtent, worldExtent, res - 1, res - 1);
    geo.rotateX(-Math.PI / 2);
    const pos = geo.attributes.position;
    const maxH = this._terrainMaxHeight;
    for (let i = 0; i < pos.count; i++) {
      const wx = pos.getX(i);
      const wz = pos.getZ(i);
      const u = (wx + worldExtent / 2) / worldExtent;
      const v = (wz + worldExtent / 2) / worldExtent;
      const h = sampleBilinear(heights, u, v);
      pos.setY(i, h * maxH);
    }
    pos.needsUpdate = true;
    geo.computeVertexNormals();

    const terrainMat = new THREE.MeshStandardMaterial({
      color: 0x3d5c3a,
      roughness: 0.85,
      metalness: 0.05,
      flatShading: false,
    });
    const terrain = new THREE.Mesh(geo, terrainMat);
    terrain.receiveShadow = true;
    terrain.castShadow = false;
    this.staticGroup.add(terrain);

    // Earth base under the terrain so it doesn't look like a floating sheet.
    const baseDepth = Math.max(6, maxH * 0.9);
    const baseGeo = new THREE.BoxGeometry(worldExtent, baseDepth, worldExtent);
    const baseMat = new THREE.MeshStandardMaterial({ color: 0x1f2a28, roughness: 1.0 });
    const base = new THREE.Mesh(baseGeo, baseMat);
    base.position.set(0, -baseDepth / 2 + 0.01, 0);
    base.receiveShadow = true;
    this.staticGroup.add(base);

    // Cache terrain-height sampler so buildings/roads can sit on the ground.
    this._heights = heights;
    this._heightsMaxH = maxH;
    this._heightAt = (x, z) => {
      const u = (x + worldExtent / 2) / worldExtent;
      const v = (z + worldExtent / 2) / worldExtent;
      return sampleBilinear(heights, u, v) * maxH;
    };

    const water = staticData.water;
    if (water && water.kind === 'circle') {
      const offset = (gridSize * cellSize) / 2;
      const cx = water.centerCol * cellSize - offset + cellSize / 2;
      const cz = water.centerRow * cellSize - offset + cellSize / 2;
      const radius = water.radiusCells * cellSize;
      const waterY = (water.level ?? 0.12) * maxH - 0.35;
      const wgeo = new THREE.CircleGeometry(radius, 48);
      wgeo.rotateX(-Math.PI / 2);
      const wmat = new THREE.MeshPhysicalMaterial({
        color: 0x1a6a8a,
        roughness: 0.15,
        metalness: 0.05,
        transmission: 0.35,
        thickness: 0.8,
        transparent: true,
        opacity: 0.92,
      });
      const wmesh = new THREE.Mesh(wgeo, wmat);
      wmesh.position.set(cx, waterY, cz);
      wmesh.receiveShadow = true;
      this.staticGroup.add(wmesh);
    }

    this.buildGridOverlay(gridSize, cellSize, worldExtent, heights, maxH);
    this.buildFramingBox(worldExtent, maxH);
  }

  buildGridOverlay(gridSize, cellSize, worldExtent, heights, maxH) {
    const half = worldExtent / 2;
    const lift = 0.08;
    // Drape the grid over the terrain by sampling height along each line.
    const sampleY = (x, z) => {
      const u = (x + half) / worldExtent;
      const v = (z + half) / worldExtent;
      return sampleBilinear(heights, u, v) * maxH + lift;
    };
    const positions = [];
    const subdivs = 24;
    for (let i = 0; i <= gridSize; i++) {
      const v = -half + i * cellSize;
      for (let s = 0; s < subdivs; s++) {
        const a = -half + (s / subdivs) * worldExtent;
        const b = -half + ((s + 1) / subdivs) * worldExtent;
        positions.push(a, sampleY(a, v), v, b, sampleY(b, v), v);
        positions.push(v, sampleY(v, a), a, v, sampleY(v, b), b);
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    const mat = new THREE.LineBasicMaterial({ color: 0xbfe8c8, transparent: true, opacity: 0.38 });
    const lines = new THREE.LineSegments(geo, mat);
    lines.name = 'gridOverlay';
    this.staticGroup.add(lines);
  }

  buildFramingBox(worldExtent, maxH) {
    const half = worldExtent / 2;
    const h = maxH * 1.4;
    const positions = [
      -half, 0, -half,  half, 0, -half,
       half, 0, -half,  half, 0,  half,
       half, 0,  half, -half, 0,  half,
      -half, 0,  half, -half, 0, -half,
      -half, h, -half,  half, h, -half,
       half, h, -half,  half, h,  half,
       half, h,  half, -half, h,  half,
      -half, h,  half, -half, h, -half,
      -half, 0, -half, -half, h, -half,
       half, 0, -half,  half, h, -half,
       half, 0,  half,  half, h,  half,
      -half, 0,  half, -half, h,  half,
    ];
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    const mat = new THREE.LineBasicMaterial({ color: 0x4c8cff, transparent: true, opacity: 0.18 });
    const box = new THREE.LineSegments(geo, mat);
    box.name = 'framingBox';
    this.staticGroup.add(box);
  }

  renderAt(index) {
    const frame = this._history[index];
    if (!frame) return;
    if (this._legacy) {
      this.renderLegacy(frame);
      return;
    }
    this.renderFrameV1(frame);
  }

  renderLegacy(snapshot) {
    this.clearDynamic();
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
        this.dynamicMeshes.push(mesh);
      }
    }
  }

  renderFrameV1(frame) {
    this.clearDynamic();

    const sampleHeight = this._heightAt ?? (() => 0);

    const roads = frame.roads || [];
    for (const r of roads) {
      const geo = new THREE.BoxGeometry(r.width, 0.12, r.depth);
      const mat = new THREE.MeshStandardMaterial({ color: 0x2a2a2e, roughness: 0.9 });
      const mesh = new THREE.Mesh(geo, mat);
      const groundY = sampleHeight(r.x, r.z);
      mesh.position.set(r.x, groundY + 0.06, r.z);
      mesh.receiveShadow = true;
      mesh.castShadow = true;
      this.scene.add(mesh);
      this.dynamicMeshes.push(mesh);
    }

    const buildings = frame.buildings || [];
    if (buildings.length === 0) return;

    const byZone = new Map();
    for (const b of buildings) {
      const z = b.zone;
      if (!byZone.has(z)) byZone.set(z, []);
      byZone.get(z).push(b);
    }

    for (const [zone, list] of byZone) {
      const color = COLORS[zone] ?? 0x888888;
      const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.65, metalness: 0.1 });
      const baseGeo = new THREE.BoxGeometry(1, 1, 1);
      const inst = new THREE.InstancedMesh(baseGeo, mat, list.length);
      const mat4 = new THREE.Matrix4();
      const pos = new THREE.Vector3();
      const quat = new THREE.Quaternion();
      const sc = new THREE.Vector3();

      list.forEach((b, i) => {
        const foot = b.footprint ?? CELL_SIZE * 0.88;
        const h = b.height ?? 2;
        const groundY = sampleHeight(b.x, b.z);
        pos.set(b.x, groundY + h / 2, b.z);
        quat.identity();
        sc.set(foot, h, foot);
        mat4.compose(pos, quat, sc);
        inst.setMatrixAt(i, mat4);
      });
      inst.instanceMatrix.needsUpdate = true;
      inst.castShadow = true;
      inst.receiveShadow = true;
      this.scene.add(inst);
      this.dynamicMeshes.push(inst);
    }
  }
}
