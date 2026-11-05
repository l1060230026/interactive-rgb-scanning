import * as THREE from 'three';
import { Viewer, RenderMode, SceneRevealMode, SceneFormat } from '@mkkellogg/gaussian-splats-3d';

const $ = id => document.getElementById(id);
let accessKey = new URLSearchParams(location.hash.slice(1)).get('key') || '';
try { accessKey ||= sessionStorage.getItem('phone-test-key') || ''; } catch (_) {}
if (accessKey) { try { sessionStorage.setItem('phone-test-key', accessKey); } catch (_) {} }
if (location.hash) history.replaceState(null, '', location.pathname + location.search);
const headers = () => accessKey ? { Authorization: 'Bearer ' + accessKey } : {};
let viewer, scene, overlay, manifest, pathGroup, marker, ready = false, loadGeneration = 0;
const objectUrls = [];
const controls = ['appearance','points','overview','top','front','full','path','rotate','scale','download'];

function error(error) {
  ready = false;
  $('loading').hidden = false;
  $('loadingTitle').textContent = '模型暂时未能打开';
  $('loadingMessage').textContent = error.message || '请刷新后重试。';
  $('retry').hidden = false;
  $('loading').querySelector('.loader').hidden = true;
  controls.forEach(id => $(id).disabled = true);
  document.body.dataset.status = 'error';
}
async function request(path) {
  const response = await fetch(path, {headers: headers(), cache:'no-store'});
  if (response.status === 401) {
    $('auth').hidden = false; $('loading').hidden = true;
    $('authError').textContent = accessKey ? '访问码无效，请粘贴原来的完整链接。' : '';
    throw new Error('请使用原来的手机测试链接打开模型。');
  }
  if (!response.ok) throw new Error(await response.text());
  return response;
}
function redraw() { viewer?.forceRenderNextFrame(); }
function setView(position, target, name, up = [0,-1,0], fov = 50) {
  viewer.controls.autoRotate = false; $('rotate').checked = false;
  viewer.camera.up.fromArray(up).normalize();
  viewer.camera.fov = fov; viewer.camera.updateProjectionMatrix();
  viewer.camera.position.fromArray(position);
  viewer.controls.target.fromArray(target);
  viewer.controls.update();
  $('viewName').textContent = name;
  document.querySelectorAll('.viewbuttons button,.preset').forEach(b => b.classList.remove('selected'));
  redraw();
}
function fit(kind = 'overview') {
  if (!ready) return;
  const box = new THREE.Box3(new THREE.Vector3(...(kind === 'full' ? manifest.full_bounds : manifest.bounds)[0]),
                             new THREE.Vector3(...(kind === 'full' ? manifest.full_bounds : manifest.bounds)[1]));
  const center = box.getCenter(new THREE.Vector3());
  const radius = box.getSize(new THREE.Vector3()).length() / 2;
  const fov = Math.min(50, THREE.MathUtils.radToDeg(2 * Math.atan(Math.tan(THREE.MathUtils.degToRad(25)) * viewer.camera.aspect)));
  const distance = radius / Math.sin(THREE.MathUtils.degToRad(fov / 2)) * 1.08;
  const direction = new THREE.Vector3(...(kind === 'top' ? [0,-1,.001] : kind === 'front' ? [0,0,-1] : [-1,-.7,-1.1])).normalize();
  const position = center.clone().addScaledVector(direction, distance);
  setView(position.toArray(), center.toArray(), {overview:'整体视角',top:'俯视',front:'正面',full:'全范围 · 包含远处点'}[kind], kind === 'top' ? [0,0,1] : [0,-1,0]);
  $(kind).classList.add('selected');
  marker.visible = false;
}
function cameraView(camera, label, button) {
  const position = new THREE.Vector3(...camera.position);
  // Exported rotation is OpenCV camera-to-world: +z forward, +y down.
  const direction = new THREE.Vector3(...camera.rotation.map(row => row[2]));
  const up = camera.rotation.map(row => -row[1]);
  const span = new THREE.Vector3(...manifest.bounds[1]).sub(new THREE.Vector3(...manifest.bounds[0])).length();
  mode(false);
  setView(position.toArray(), position.clone().addScaledVector(direction, span * .3).toArray(), label, up,
    THREE.MathUtils.radToDeg(2 * Math.atan(camera.height / (2 * camera.fy))));
  button.classList.add('selected');
  marker.position.copy(position); marker.visible = false;
}
function addPath() {
  const points = manifest.cameras.map(c => new THREE.Vector3(...c.position));
  pathGroup = new THREE.Group();
  const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),
    new THREE.LineBasicMaterial({color:0x64dac5, transparent:true, opacity:.85, depthTest:false}));
  line.renderOrder = 100;
  pathGroup.add(line);
  const size = new THREE.Vector3(...manifest.bounds[1]).sub(new THREE.Vector3(...manifest.bounds[0])).length() * .008;
  marker = new THREE.Mesh(new THREE.SphereGeometry(size, 12, 8), new THREE.MeshBasicMaterial({color:0xf7c477,depthTest:false}));
  marker.renderOrder = 101; marker.visible = false; pathGroup.add(marker);
  for (const index of [0, manifest.cameras.length - 1]) {
    const dot = marker.clone(); dot.position.copy(points[index]); dot.visible = true;
    dot.material = new THREE.MeshBasicMaterial({color:index === 0 ? 0x6be4c2 : 0xffc578,depthTest:false});
    pathGroup.add(dot);
  }
  pathGroup.visible = $('path').checked;
  overlay.add(pathGroup);
}
async function load(id) {
  const generation = ++loadGeneration;
  ready = false; document.body.dataset.status = 'loading';
  controls.forEach(id => $(id).disabled = true);
  $('runs').disabled = true; $('loading').hidden = false; $('auth').hidden = true;
  $('retry').hidden = true; $('loading').querySelector('.loader').hidden = false;
  $('loadingTitle').textContent = '正在打开模型'; $('progress').value = 0;
  try {
    if (viewer) { await viewer.dispose(); $('viewport').replaceChildren(); }
    objectUrls.splice(0).forEach(url => URL.revokeObjectURL(url));
    const base = '/api/models/' + encodeURIComponent(id);
    manifest = await (await request(base + '/manifest')).json();
    $('count').textContent = (manifest.vertex_count / 10000).toFixed(1) + ' 万';
    $('cameras').textContent = manifest.cameras.length;
    $('fileSize').textContent = (manifest.model_bytes / 1e6).toFixed(1) + ' MB';
    $('modelNote').textContent = manifest.preview_note;
    scene = new THREE.Scene(); overlay = new THREE.Scene(); pathGroup = null;
    viewer = new Viewer({rootElement:$('viewport'), threeScene:scene,
      cameraUp:[0,-1,0], initialCameraPosition:[-2,-2,-3], initialCameraLookAt:[0,0,0],
      sharedMemoryForWorkers:false, gpuAcceleratedSort:false, integerBasedSort:false,
      sphericalHarmonicsDegree:2, ignoreDevicePixelRatio:true,
      renderMode:RenderMode.OnChange, sceneRevealMode:SceneRevealMode.Instant,
      optimizeSplatData:false, inMemoryCompressionLevel:0, freeIntermediateSplatData:true,
      enableOptionalEffects:false, halfPrecisionCovariancesOnGPU:false});
    viewer.renderer.setClearColor($('background').value, 1);
    // Photo-SLAM cuda_rasterizer/auxiliary.h rejects centers closer than 0.2.
    // This web shader expands the clip volume by 1.2: near * 1.1 matches that cutoff.
    viewer.camera.near = (manifest.native_near_clip || .2) * 1.1;
    viewer.camera.far = 1000; viewer.camera.updateProjectionMatrix();
    viewer.controls.maxDistance = 500; viewer.controls.minDistance = .003;
    viewer.controls.enableDamping = true;
    viewer.controls.autoRotateSpeed = .6;
    // Draw the trajectory after the transparent Gaussian pass so it stays legible.
    const render = viewer.render.bind(viewer);
    viewer.render = () => {
      render();
      if (pathGroup?.visible) {
        const autoClear = viewer.renderer.autoClear;
        viewer.renderer.autoClear = false;
        viewer.renderer.render(overlay, viewer.camera);
        viewer.renderer.autoClear = autoClear;
      }
    };
    viewer.start();
    await viewer.addSplatScene(base + '/model.ply', {format:SceneFormat.Ply, headers:headers(),
      showLoadingUI:false, splatAlphaRemovalThreshold:0,
      onProgress:(percent, _, stage) => {
        if (generation !== loadGeneration) return;
        $('progress').value = Number.isFinite(percent) ? percent : 0;
        $('loadingMessage').textContent = stage === 0 ? '读取原始三维模型 ' + Math.round(percent || 0) + '%' : '正在准备三维显示…';
      }});
    if (generation !== loadGeneration) return;
    addPath(); ready = true;
    $('scale').value = 1; $('scaleValue').textContent = '100%';
    mode(true); fit();
    $('presets').replaceChildren();
    for (const preset of manifest.previews) {
      const camera = manifest.cameras.find(c => c.id === preset.camera_id);
      const button = document.createElement('button'); button.className = 'preset';
      button.title = preset.label + ' · 模型渲染视角';
      const image = document.createElement('img'); image.alt = preset.label + '的模型渲染';
      const label = document.createElement('span'); label.textContent = preset.label;
      button.append(image, label); $('presets').append(button);
      button.onclick = () => cameraView(camera, preset.label, button);
      request(base + '/preview/' + preset.camera_id).then(r => r.blob()).then(blob => {
        if (generation !== loadGeneration) return;
        const url = URL.createObjectURL(blob); objectUrls.push(url); image.src = url;
      }).catch(() => { image.alt = '预览未能读取'; });
    }
    controls.forEach(id => $(id).disabled = false); $('runs').disabled = false;
    $('loading').hidden = true; document.body.dataset.status = 'ready';
    window.modelViewer = {viewer, manifest, fit, get ready(){return ready;}};
  } catch (e) { if ($('auth').hidden) error(e); }
}
function mode(points) {
  if (!ready) return;
  viewer.splatMesh.setPointCloudModeEnabled(points);
  $('appearance').classList.toggle('selected', !points); $('points').classList.toggle('selected', points);
  $('appearance').setAttribute('aria-pressed', String(!points)); $('points').setAttribute('aria-pressed', String(points));
  $('modeBadge').textContent = points ? 'POINT STRUCTURE' : '3D GAUSSIANS'; redraw();
}
async function boot() {
  try {
    const runs = (await (await request('/api/models')).json()).filter(run => run.backend !== 'mast3r-slam');
    if (!runs.length) throw new Error('还没有准备好的扫描模型。');
    $('runs').replaceChildren(...runs.map(run => {
      const option = document.createElement('option'); option.value = run.id;
      const stamp = run.id.match(/(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z/);
      const date = stamp ? new Date(`${stamp[1]}-${stamp[2]}-${stamp[3]}T${stamp[4]}:${stamp[5]}:${stamp[6]}Z`) : null;
      option.textContent = date ? date.toLocaleString('zh-CN',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit'}) + ' · ' + run.backend : run.title;
      return option;
    }));
    const requested = new URLSearchParams(location.search).get('run');
    $('runs').value = runs.some(r => r.id === requested) ? requested : runs[0].id;
    await load($('runs').value);
  } catch (e) { if ($('auth').hidden) error(e); }
}
$('runs').onchange = () => load($('runs').value);
for (const kind of ['overview','top','front','full']) $(kind).onclick = () => fit(kind);
$('appearance').onclick = () => mode(false); $('points').onclick = () => mode(true);
$('path').onchange = () => { if (pathGroup) pathGroup.visible = $('path').checked; redraw(); };
$('rotate').onchange = () => { if (ready) {viewer.controls.autoRotate = $('rotate').checked;redraw();} };
$('scale').oninput = () => { $('scaleValue').textContent = Math.round($('scale').value * 100) + '%'; if (ready) {viewer.splatMesh.setSplatScale(Number($('scale').value)); redraw();} };
$('background').onchange = () => { if (viewer) {viewer.renderer.setClearColor($('background').value,1);redraw();} };
$('fullscreen').onclick = async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else if ($('viewport').parentElement.requestFullscreen) await $('viewport').parentElement.requestFullscreen();
    else { $('loadingMessage').textContent = '当前浏览器不支持全屏，可横屏查看。'; }
  } catch (_) { /* Browser may decline fullscreen; keep the viewer usable. */ }
};
$('download').onclick = async () => {
  const button = $('download'); button.disabled = true;
  try {
    const blob = await (await request('/api/models/' + encodeURIComponent(manifest.id) + '/model.ply')).blob();
    const url = URL.createObjectURL(blob); const a = document.createElement('a');
    a.href = url; a.download = manifest.id + '.ply'; a.click(); setTimeout(() => URL.revokeObjectURL(url), 30000);
  } catch (e) { error(e); } finally {button.disabled = false;}
};
$('retry').onclick = () => location.reload();
$('auth').onsubmit = event => {
  event.preventDefault(); const value = $('access').value.trim();
  try {const url = new URL(value); accessKey = new URLSearchParams(url.hash.slice(1)).get('key') || decodeURIComponent(url.pathname.split('/join/')[1] || '');}
  catch (_) {accessKey = value;}
  if (!accessKey) { $('authError').textContent = '链接中没有找到访问码，请粘贴完整链接。'; return; }
  try {sessionStorage.setItem('phone-test-key', accessKey);} catch (_) {}
  $('auth').hidden = true; $('loading').hidden = false; void boot();
};
let resizeFrame;
new ResizeObserver(() => {
  cancelAnimationFrame(resizeFrame);
  resizeFrame = requestAnimationFrame(() => requestAnimationFrame(() => {
    if (!ready) return;
    const selected = document.querySelector('.viewbuttons .selected');
    if (selected) fit(selected.id);
  }));
}).observe($('viewport'));
void boot();
