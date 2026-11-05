import * as THREE from 'three';
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls.js';
import {PLYLoader} from 'three/examples/jsm/loaders/PLYLoader.js';
const $ = id => document.getElementById(id);
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
renderer.setClearColor(0x111b23); document.body.append(renderer.domElement);
const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(50,1,.005,10000);
camera.up.set(0,-1,0);
const controls = new OrbitControls(camera, renderer.domElement);
const render = () => renderer.render(scene,camera);
controls.addEventListener('change',render);
let bounds, fullBounds, cloud, trajectory;
function resize(){camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);render();}
addEventListener('resize',resize); resize();
function fit(kind='overview'){
  if (!bounds) return;
  const box=kind==='full'?fullBounds:bounds;
  const center=box.getCenter(new THREE.Vector3());
  const radius=box.getSize(new THREE.Vector3()).length()/2;
  const fov=Math.min(camera.fov,THREE.MathUtils.radToDeg(2*Math.atan(Math.tan(THREE.MathUtils.degToRad(camera.fov/2))*camera.aspect)));
  const direction=new THREE.Vector3(...(kind==='top'?[0,-1,.001]:[-1,-.7,-1.1])).normalize();
  camera.up.set(...(kind==='top'?[0,0,1]:[0,-1,0]));
  camera.position.copy(center).addScaledVector(direction,radius/Math.sin(THREE.MathUtils.degToRad(fov/2))*1.08);
  controls.target.copy(center);controls.update();render();
}
for(const kind of ['overview','top','full']) $(kind).onclick=()=>fit(kind);
$('path').onchange=()=>{if(trajectory) trajectory.visible=$('path').checked;render();};
$('size').oninput=()=>{if(cloud) cloud.material.size=Number($('size').value);render();};
async function boot(){try {
  const response=await fetch('../summary.json'); if(!response.ok) throw Error('无法读取测试记录');
  const summary=await response.json();
  const geometry=await new PLYLoader().loadAsync('../native/model.ply');
  geometry.computeBoundingBox();fullBounds=geometry.boundingBox.clone();
  cloud=new THREE.Points(geometry,new THREE.PointsMaterial({size:1,sizeAttenuation:false,vertexColors:true}));
  scene.add(cloud);
  const positions=geometry.attributes.position;
  // Bounds affect only framing; no point is removed or moved.
  const samples=[[],[],[]],stride=Math.max(1,Math.floor(positions.count/50000));
  for(let i=0;i<positions.count;i+=stride) for(let axis=0;axis<3;axis++) samples[axis].push(positions.array[i*3+axis]);
  for(const sample of samples) sample.sort((a,b)=>a-b);
  bounds=new THREE.Box3(new THREE.Vector3(...samples.map(s=>s[Math.floor(s.length*.01)])),new THREE.Vector3(...samples.map(s=>s[Math.floor(s.length*.99)])));
  const keyframes=await (await fetch('../native/keyframes.json')).json();
  const points=keyframes.map(k=>new THREE.Vector3(...k.sim3.flat().slice(0,3)));
  trajectory=new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),new THREE.LineBasicMaterial({color:0x74e0c4,depthTest:false}));
  trajectory.renderOrder=10;scene.add(trajectory);
  $('status').textContent=summary.replay
    ? `${summary.frames_processed}/${summary.replay.captures} 帧 · ${summary.replay.output_fps_during_capture.toFixed(2)} FPS 输出 · 帧龄 P95 ${summary.replay.capture_to_output_ms_p95.toFixed(0)} ms · ${(positions.count/10000).toFixed(1)} 万点`
    : `${summary.frames_processed} 帧 · ${summary.keyframes} 关键帧 · ${summary.mean_processing_fps.toFixed(2)} FPS · ${(positions.count/10000).toFixed(1)} 万点`;
  window.mast3rReview={scene,camera,controls,summary,pointCount:positions.count,fit};
  document.body.dataset.status='ready';fit();
} catch(error){$('status').textContent='打开失败：'+error.message;document.body.dataset.status='error';}}
void boot();
