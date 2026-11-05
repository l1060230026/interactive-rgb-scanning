import * as THREE from 'three';
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls.js';
import {PLYLoader} from 'three/examples/jsm/loaders/PLYLoader.js';
const $=id=>document.getElementById(id);
let key=new URLSearchParams(location.hash.slice(1)).get('key')||'';
try{key ||= sessionStorage.getItem('phone-test-key')||'';if(key)sessionStorage.setItem('phone-test-key',key);}catch(_){}
if(location.hash)history.replaceState(null,'',location.pathname+location.search);
const viewport=$('viewport');
const renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.setClearColor(0x172329);viewport.prepend(renderer.domElement);
const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(50,1,.005,10000);
camera.up.set(0,-1,0);
const controls=new OrbitControls(camera,renderer.domElement);
const render=()=>renderer.render(scene,camera);controls.addEventListener('change',render);
let cloud,trajectory,bounds,fullBounds,current,manifest,runs=[],loading=false;
for(const id of ['captureLink','brandCapture'])$(id).href='/'+(key?'#key='+encodeURIComponent(key):'');
function stamp(id){const m=id.match(/live-phone-(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z/);return m?new Date(Date.UTC(...m.slice(1).map((v,i)=>Number(v)-(i===1?1:0)))).toLocaleString('en-GB',{timeZone:'Asia/Shanghai',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false}):id;}
function renderRuns(){
 const query=$('search').value.trim().toLowerCase();$('runList').replaceChildren();
 for(const run of runs){
  const label=stamp(run.id);if(query&&!`${run.title} ${label}`.toLowerCase().includes(query))continue;
  const button=document.createElement('button');button.className='run-item';button.disabled=loading;button.setAttribute('aria-current',String(run.id===current));button.setAttribute('aria-label','Open scan '+label);
  const icon=document.createElement('span');icon.className='run-icon';icon.innerHTML='<svg><use href="#cube"/></svg>';
  const text=document.createElement('span'),title=document.createElement('strong'),sub=document.createElement('small');title.textContent='Scan · '+label;sub.textContent=(run.vertex_count/1e6).toFixed(2)+' M points · RGB';text.append(title,sub);button.append(icon,text);button.onclick=()=>{if(!loading){$('runs').value=run.id;void load(run.id);}};$('runList').append(button);
 }
}
$('search').oninput=renderRuns;
function state(title,hint){$('emptyState').hidden=false;$('emptyTitle').textContent=title;$('emptyHint').textContent=hint;}
function toast(text){$('modelToast').textContent=text;$('modelToast').hidden=false;setTimeout(()=>$('modelToast').hidden=true,2600);}
async function request(path){const r=await fetch(path,{headers:key?{Authorization:'Bearer '+key}:{},cache:'no-store'});if(!r.ok)throw Error(r.status===401?'Open the workspace using the original camera link.':await r.text());return r;}
function resize(){const {width,height}=viewport.getBoundingClientRect();camera.aspect=width/height;camera.updateProjectionMatrix();renderer.setSize(width,height);render();}
new ResizeObserver(resize).observe(viewport);resize();
function fit(kind='overview'){
 if(!bounds)return;const box=kind==='full'?fullBounds:bounds,center=box.getCenter(new THREE.Vector3()),radius=Math.max(.01,box.getSize(new THREE.Vector3()).length()/2);
 const fov=Math.min(camera.fov,THREE.MathUtils.radToDeg(2*Math.atan(Math.tan(THREE.MathUtils.degToRad(camera.fov/2))*camera.aspect)));
 const direction=new THREE.Vector3(...(kind==='top'?[0,-1,.001]:[-1,-.7,-1.1])).normalize();camera.up.set(...(kind==='top'?[0,0,1]:[0,-1,0]));
 camera.position.copy(center).addScaledVector(direction,radius/Math.sin(THREE.MathUtils.degToRad(fov/2))*1.08);controls.target.copy(center);controls.update();render();
 $('overview').setAttribute('aria-pressed',String(kind!=='top'));$('top').setAttribute('aria-pressed',String(kind==='top'));
}
for(const name of ['overview','top','full'])$(name).onclick=()=>fit(name);
$('path').onchange=()=>{if(trajectory)trajectory.visible=$('path').checked;render();};
$('size').oninput=()=>{if(cloud)cloud.material.size=Number($('size').value);$('pointSize').textContent=Number($('size').value).toFixed(2)+' px';render();};
async function load(id){
 loading=true;renderRuns();$('status').textContent='Loading point cloud…';document.body.dataset.status='loading';$('runs').disabled=true;state('Opening reconstruction','Loading saved point cloud');
 for(const name of ['download','saveView','overview','top','full'])$(name).disabled=true;
 try{
  const base='/api/models/'+encodeURIComponent(id),data=await(await request(base+'/manifest')).json();
  const geometry=new PLYLoader().parse(await(await request(base+'/model.ply')).arrayBuffer());
  geometry.computeBoundingBox();
  if(cloud){scene.remove(cloud);cloud.geometry.dispose();cloud.material.dispose();}
  if(trajectory){scene.remove(trajectory);trajectory.geometry.dispose();trajectory.material.dispose();}
  cloud=new THREE.Points(geometry,new THREE.PointsMaterial({size:Number($('size').value),sizeAttenuation:false,vertexColors:true}));scene.add(cloud);
  bounds=new THREE.Box3(new THREE.Vector3(...data.bounds[0]),new THREE.Vector3(...data.bounds[1]));fullBounds=geometry.boundingBox.clone();
  const points=data.cameras.map(k=>new THREE.Vector3(...k.position));trajectory=new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),new THREE.LineBasicMaterial({color:0x74e0c4,depthTest:false}));
  trajectory.renderOrder=10;trajectory.visible=$('path').checked;scene.add(trajectory);
  current=id;manifest=data;$('status').textContent='Scan loaded';
  $('modelTitle').textContent='Indoor reconstruction';$('modelSubtitle').textContent='Mobile scan · '+stamp(id);$('pointCount').textContent=data.vertex_count.toLocaleString('en-US');$('viewCount').textContent=data.cameras.length+'';$('fileSize').textContent=(data.model_bytes/1e6).toFixed(1)+' MB';$('savedState').hidden=false;$('emptyState').hidden=true;
  for(const name of ['download','saveView','overview','top','full'])$(name).disabled=false;
  window.mast3rReview={scene,camera,controls,manifest,pointCount:geometry.attributes.position.count,fit};document.body.dataset.status='ready';fit();
 }catch(error){$('status').textContent='Unable to open: '+error.message;state('Unable to open model',error.message);document.body.dataset.status='error';}
 finally{loading=false;$('runs').disabled=false;renderRuns();}
}
$('runs').onchange=()=>load($('runs').value);
$('download').onclick=async()=>{if(!current)return;try{const url=URL.createObjectURL(await(await request('/api/models/'+encodeURIComponent(current)+'/model.ply')).blob()),a=document.createElement('a');a.href=url;a.download=current+'.ply';a.click();setTimeout(()=>URL.revokeObjectURL(url),30000);}catch(e){$('status').textContent=e.message;}};
$('saveView').onclick=()=>{if(!current)return;render();renderer.domElement.toBlob(blob=>{if(!blob)return;const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=current+'-view.png';a.click();setTimeout(()=>URL.revokeObjectURL(url),30000);toast('View saved');});};
async function boot(){try{
 runs=(await(await request('/api/models')).json()).filter(r=>r.backend==='mast3r-slam');$('runs').replaceChildren();$('runCount').textContent=String(runs.length);renderRuns();
 if(!runs.length){$('runs').append(new Option('No saved scans',''));$('status').textContent='Saved models appear here after you finish a scan.';state('Ready for your first scan','Complete and save a scan on your phone, then explore the model here.');document.body.dataset.status='empty';return;}
 for(const run of runs)$('runs').append(new Option(stamp(run.id),run.id));
 const preferred=new URLSearchParams(location.search).get('run');if(runs.some(r=>r.id===preferred))$('runs').value=preferred;
 await load($('runs').value);
}catch(error){$('status').textContent=error.message;state('Unable to load scans',error.message);document.body.dataset.status='error';}}
void boot();
