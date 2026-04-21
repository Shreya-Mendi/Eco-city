# Eco-city

## 3D viewer (Three.js)

Training can export a versioned JSON file consumed by the browser viewer:

- Run with `--export` from [`training/train.py`](training/train.py), or call [`visualization/exporter.export`](visualization/exporter.py) after a rollout.
- Output includes `formatVersion`, `static` (terrain heightmap, water, buildable grid), and `history` (per-step grid, metrics, roads, buildings).
- Serve the viewer over HTTP (browser `fetch` blocks `file://` in many setups):

```bash
cd visualization/threejs && python3 -m http.server 8080
```

Then open `http://localhost:8080` and ensure `city_data.json` is present in that folder.

Legacy exports (a plain JSON array of steps) still load; the viewer falls back to the original box-per-cell rendering.