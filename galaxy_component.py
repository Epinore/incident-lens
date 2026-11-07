"""Canvas 'galaxy' renderer: glowing points, floating theme labels, hover cards, and a
shooting-star landing animation for a new report. Pure HTML/JS (no CDN), embedded via
streamlit.components.v1.html.
"""
import json
import numpy as np


def galaxy_html(inc, themes, tcol, height=640, target=None, neighbors=None, sims=None, title_note=""):
    x = inc.x.to_numpy(); y = inc.y.to_numpy()
    cx, cy = (x.max() + x.min()) / 2, (y.max() + y.min()) / 2
    s = max(x.max() - x.min(), y.max() - y.min()) / 2
    nx, ny = (x - cx) / s, (y - cy) / s
    tidx = {t: i for i, t in enumerate(themes)}
    pts = [[round(float(a), 4), round(float(b), 4), tidx[t], f"{ty or 'Incident'} · {d[:4] or 'undated'}", w.lstrip('-• ')[:150]]
           for a, b, t, ty, d, w in zip(nx, ny, inc.theme, inc.type, inc.date, inc.what)]
    labels = []
    for t in themes:
        m = (inc.theme == t).to_numpy()
        labels.append([round(float(np.median(nx[m])), 4), round(float(np.median(ny[m])), 4), t, int(m.sum())])
    # nudge overlapping theme labels apart (screen aspect: x is stretched 1.25x)
    for _ in range(80):
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                dx = (labels[i][0] - labels[j][0]) * 1.25; dy = labels[i][1] - labels[j][1]
                if abs(dx) < 0.42 and abs(dy) < 0.11:
                    push = (0.11 - abs(dy)) / 2 + 0.005
                    sgn = 1 if dy >= 0 else -1
                    labels[i][1] += sgn * push; labels[j][1] -= sgn * push
    land = None
    if target is not None:
        tx, ty_ = (target[0] - cx) / s, (target[1] - cy) / s
        land = {"x": float(tx), "y": float(ty_), "nb": [int(i) for i in neighbors], "sim": [round(float(v) * 100) for v in sims]}
    data = json.dumps({"pts": pts, "labels": labels, "colors": [tcol[t] for t in themes], "land": land})
    return TEMPLATE.replace("__DATA__", data).replace("__H__", str(height)).replace("__NOTE__", title_note)


TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8">
<style>
html,body{margin:0;background:transparent;overflow:hidden;font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
#wrap{position:relative;width:100%;height:__H__px;border-radius:20px;overflow:hidden;
 background:radial-gradient(120% 90% at 50% 40%,#134A66 0%,#0A3049 50%,#05192A 100%);
 box-shadow:inset 0 0 0 1px rgba(94,234,212,.10)}
canvas{position:absolute;inset:0;width:100%;height:100%}
#tip{position:absolute;pointer-events:none;max-width:300px;padding:12px 14px;border-radius:12px;
 background:rgba(6,24,38,.94);backdrop-filter:blur(8px);border:1px solid rgba(94,234,212,.25);color:#D5EEEA;
 font-size:12.5px;line-height:1.45;opacity:0;transition:opacity .15s;box-shadow:0 10px 40px rgba(0,0,0,.5)}
#tip b{display:block;font-size:13px;color:#fff;margin-bottom:4px}
#tip .th{font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;margin-bottom:6px}
#note{position:absolute;left:18px;bottom:14px;color:rgba(127,167,181,.85);font-size:11.5px;letter-spacing:.02em}
</style></head><body><div id="wrap"><canvas id="c"></canvas><div id="tip"></div><div id="note">__NOTE__</div></div>
<script>
const D=__DATA__;
const wrap=document.getElementById('wrap'),cv=document.getElementById('c'),ctx=cv.getContext('2d'),tip=document.getElementById('tip');
let W=0,H=0,DPR=Math.min(2,window.devicePixelRatio||1);
function size(){W=wrap.clientWidth;H=wrap.clientHeight;cv.width=W*DPR;cv.height=H*DPR;ctx.setTransform(DPR,0,0,DPR,0,0)}
size();addEventListener('resize',size);
// glow sprites per color
const sprites=D.colors.map(c=>{const s=document.createElement('canvas');s.width=s.height=64;const g=s.getContext('2d');
 const r=g.createRadialGradient(32,32,0,32,32,32);r.addColorStop(0,'rgba(255,255,255,1)');r.addColorStop(.12,c);
 r.addColorStop(.35,c+'66');r.addColorStop(1,c+'00');g.fillStyle=r;g.fillRect(0,0,64,64);return s});
const starSprite=(()=>{const s=document.createElement('canvas');s.width=s.height=64;const g=s.getContext('2d');const r=g.createRadialGradient(32,32,0,32,32,32);r.addColorStop(0,'rgba(255,255,255,1)');r.addColorStop(.15,'#FF7A45');r.addColorStop(.4,'#FF7A4566');r.addColorStop(1,'#FF7A4500');g.fillStyle=r;g.fillRect(0,0,64,64);return s})();
const bg=Array.from({length:260},()=>({x:Math.random(),y:Math.random(),r:Math.random()*1.1+.2,p:Math.random()*6.28}));
const fr=v=>v-Math.floor(v);
const P=D.pts.map((p,i)=>({x:p[0],y:p[1],c:p[2],t:p[3],w:p[4],ph:Math.random()*6.28,d:Math.hypot(p[0],p[1]),
 z:.42*Math.sin(p[2]*1.9+.6)+.5*(fr(Math.sin(i*12.9898)*43758.5453)-.5)}));
let drag=0,dragV=0,down=null;
let mx=0,my=0,tmx=0,tmy=0,hover=-1,t0=performance.now()-(+((location.hash.match(/t=(\d+)/)||[])[1]||0));
const nbSet=new Set(D.land?D.land.nb:[]);
function proj(x,y,t,z=0){const rot=t*0.00002,ca=Math.cos(rot),sa=Math.sin(rot);
 const X=x*ca-y*sa,Y=x*sa+y*ca;const a=.38*Math.sin(t*0.00012)+drag,cb=Math.cos(a),sb=Math.sin(a);
 const X2=X*cb+z*sb,Z2=-X*sb+z*cb;const f=1/(1+Z2*.32);const sc=Math.min(W,H)*0.43;
 return [W/2+X2*sc*1.25*f+mx*14, H/2+Y*sc*f+my*14, f]}
wrap.style.cursor='grab';
wrap.addEventListener('pointerdown',e=>{down=e.clientX;dragV=0;wrap.style.cursor='grabbing'});
addEventListener('pointerup',()=>{down=null;wrap.style.cursor='grab'});
addEventListener('pointermove',e=>{if(down===null)return;const d=(e.clientX-down)*.006;down=e.clientX;drag+=d;dragV=d});
wrap.addEventListener('mousemove',e=>{const r=wrap.getBoundingClientRect();tmx=(e.clientX-r.left)/W-.5;tmy=(e.clientY-r.top)/H-.5;
 let best=-1,bd=144;const t=performance.now()-t0;
 for(let i=0;i<P.length;i++){const [px,py]=proj(P[i].x,P[i].y,t,P[i].z);const d=(px-e.clientX+r.left)**2+(py-e.clientY+r.top)**2;if(d<bd){bd=d;best=i}}
 hover=best;if(best>=0){const p=P[best];const col=D.colors[p.c];
  tip.innerHTML=`<div class="th" style="color:${col}">● ${D.labels[p.c][2]}</div><b>${p.t}</b>${p.w}…`;
  const [px,py]=proj(p.x,p.y,t,p.z);tip.style.left=Math.min(px+16,W-320)+'px';tip.style.top=Math.min(py+16,H-150)+'px';tip.style.opacity=1}
 else tip.style.opacity=0});
wrap.addEventListener('mouseleave',()=>{hover=-1;tip.style.opacity=0;tmx=tmy=0});
const ease=t=>1-Math.pow(1-Math.min(1,Math.max(0,t)),3);
function frame(now){const t=now-t0;if(down===null){drag+=dragV;dragV*=.94}mx+=(tmx-mx)*.05;my+=(tmy-my)*.05;ctx.clearRect(0,0,W,H);
 // background stars
 for(const s of bg){ctx.globalAlpha=.25+.25*Math.sin(t*.001+s.p);ctx.fillStyle='#A5F3FC';ctx.beginPath();ctx.arc(s.x*W+mx*4,s.y*H+my*4,s.r,0,6.28);ctx.fill()}
 const L=D.land, landT=L?(t-1400)/1700:0, landed=L&&landT>=1, dim=landed?Math.max(.22,1-ease((t-3100)/600)*.78):1;
 // points
 ctx.globalCompositeOperation='lighter';
 for(let i=0;i<P.length;i++){const p=P[i];const intro=ease((t-p.d*700)/900);if(intro<=0)continue;
  const [px,py,f]=proj(p.x,p.y,t,p.z);const br=.75+.25*Math.sin(t*.0016+p.ph);
  const isNb=nbSet.has(i);let a=intro*br*(isNb&&landed?1:dim)*Math.min(1,.35+.65*f*f);let sz=(10+ (i===hover?10:0))*(.6+.4*intro)*(isNb&&landed?1.8:1)*f*f;
  ctx.globalAlpha=Math.min(1,a);ctx.drawImage(sprites[p.c],px-sz,py-sz,sz*2,sz*2)}
 ctx.globalCompositeOperation='source-over';
 // theme labels
 const la=ease((t-900)/800)*(landed?.35:1);
 ctx.font='600 11px Inter,system-ui,sans-serif';ctx.textAlign='center';
 for(const [li,l] of D.labels.entries()){const [px,py]=proj(l[0],l[1],t);ctx.globalAlpha=la*.92;ctx.shadowColor='rgba(0,0,0,.9)';ctx.shadowBlur=8;
  ctx.fillStyle='#F0FDFA';ctx.fillText(l[2].toUpperCase(),px,py-6);ctx.globalAlpha=la*.6;ctx.font='500 10px Inter,system-ui,sans-serif';
  ctx.fillStyle=D.colors[D.labels.indexOf(l)];ctx.fillText(l[3]+' reports',px,py+8);ctx.font='600 11px Inter,system-ui,sans-serif'}
 ctx.shadowBlur=0;
 // hover ring
 if(hover>=0){const [px,py]=proj(P[hover].x,P[hover].y,t,P[hover].z);ctx.globalAlpha=.9;ctx.strokeStyle='#fff';ctx.lineWidth=1.2;ctx.beginPath();ctx.arc(px,py,9,0,6.28);ctx.stroke()}
 // shooting star landing
 if(L&&landT>0){const [tx,ty]=proj(L.x,L.y,t);const sx=-60,sy=-40,cx1=W*.25,cy1=-H*.15;
  const pos=u=>{const v=1-u;return [v*v*sx+2*v*u*cx1+u*u*tx, v*v*sy+2*v*u*cy1+u*u*ty]};
  const u=ease(landT);
  if(!landed){for(let k=0;k<26;k++){const uu=Math.max(0,u-k*.012);const [qx,qy]=pos(uu);ctx.globalAlpha=(1-k/26)*.9;
    ctx.fillStyle='#FFD3BF';ctx.beginPath();ctx.arc(qx,qy,Math.max(.5,4.2-k*.15),0,6.28);ctx.fill()}}
  const [hx,hy]=pos(u);
  if(landed){const k=(t-1400-1700);
   // shockwave
   for(const off of [0,350]){const r=ease((k-off)/1200)*Math.min(W,H)*.22;if(k>off&&k<off+1200){ctx.globalAlpha=1-(k-off)/1200;ctx.strokeStyle='#FF7A45';ctx.lineWidth=2;ctx.beginPath();ctx.arc(tx,ty,r,0,6.28);ctx.stroke()}}
   // links to neighbours
   L.nb.forEach((n,j)=>{const lp=ease((k-200-j*140)/600);if(lp<=0)return;const [nx_,ny_]=proj(P[n].x,P[n].y,t,P[n].z);
    ctx.globalAlpha=.75;ctx.strokeStyle='#5EEAD4';ctx.lineWidth=1.3;ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(tx,ty);
    ctx.lineTo(tx+(nx_-tx)*lp,ty+(ny_-ty)*lp);ctx.stroke();ctx.setLineDash([]);
    if(lp>=1){const pr=6+2*Math.sin(t*.006+j);ctx.globalAlpha=.95;ctx.strokeStyle='#CCFBF1';ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(nx_,ny_,pr+4,0,6.28);ctx.stroke();
     ctx.font='700 11px Inter,system-ui,sans-serif';ctx.fillStyle='#CCFBF1';ctx.textAlign='left';ctx.shadowColor='#000';ctx.shadowBlur=6;ctx.fillText(L.sim[j]+'%',nx_+12,ny_+4);ctx.shadowBlur=0}});
  }
  // the star itself
  const pulse=landed?1+.15*Math.sin(t*.005):1;ctx.globalCompositeOperation='lighter';ctx.globalAlpha=1;
  ctx.drawImage(starSprite,hx-26*pulse,hy-26*pulse,52*pulse,52*pulse);ctx.globalCompositeOperation='source-over';
  ctx.fillStyle='#fff';ctx.beginPath();for(let k=0;k<10;k++){const r=k%2?4:10*pulse,a=k*Math.PI/5-Math.PI/2;ctx.lineTo(hx+r*Math.cos(a),hy+r*Math.sin(a))}ctx.closePath();ctx.fill();
  if(landed){ctx.globalAlpha=ease((t-3300)/500);ctx.font='800 12px Inter,system-ui,sans-serif';ctx.textAlign='left';ctx.fillStyle='#fff';ctx.shadowColor='#000';ctx.shadowBlur=8;
   ctx.textAlign='center';ctx.fillText('NEW REPORT',hx,hy-26);ctx.shadowBlur=0}
 }
 ctx.globalAlpha=1;requestAnimationFrame(frame)}
requestAnimationFrame(frame);
</script></body></html>"""
