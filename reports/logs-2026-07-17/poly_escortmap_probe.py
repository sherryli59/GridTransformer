"""Early-efficacy paired probe of the escorted map (mid-training ckpt OK): identical
(state, pair, seed) through escorted_work with flow=None vs flow=ckpt.
Reports per-bin paired dW = W_map - W_plain and acceptance-proxy shift."""
import numpy as np, torch, sys, time
sys.path.insert(0,'/mnt/ssd/GridTransformer'); sys.path.insert(0,'/mnt/ssd/GridTransformer/reports/logs-2026-07-17')
from liquid_coupling_flow.poly.escorted_ncmc import escorted_work
from liquid_coupling_flow.poly.swap_flow import SwapBlockFlow
from liquid_coupling_flow.poly.model import seed_numba

NSTEPS=int(sys.argv[1]) if len(sys.argv)>1 else 8
CKPT=sys.argv[2] if len(sys.argv)>2 else 'liquid_coupling_flow/artifacts/poly_escortmap_T0.085_k8_best.pt'
OUT=sys.argv[3] if len(sys.argv)>3 else 'reports/logs-2026-07-17/poly_escortmap_probe.pt'
SWEEPS_TOTAL=200   # keep TOTAL relaxation constant: sweeps_per_step = 200//NSTEPS
dev='cuda' if torch.cuda.is_available() else 'cpu'
ck=torch.load(CKPT,map_location=dev,weights_only=False)
ar=ck['args']
flow=SwapBlockFlow(k_max=ar['k_max'],m_env=ar['m_env'],hidden_nf=ar['hidden_nf'],
                   n_layers=ar['n_layers'],n_sig_bins=ar.get('n_sig_bins',8),
                   base_w=ar.get('base_w',0.05)).to(dev)
flow.load_state_dict(ck['state_dict']); flow.eval()
print(f"NSTEPS={NSTEPS} ckpt step {ck.get('step')} best {ck.get('fm_loss', ck.get('best'))}", flush=True)

D=torch.load('reports/logs-2026-07-17/poly_bank_fixed_run3.pt',weights_only=False)
rec=D[(3,0.085)]; L=float(rec['L']); N=rec['x'].shape[1]; beta=1/0.085
BINS=[(0.1,0.2),(0.2,0.3),(0.3,0.45),(0.45,0.9)]
rng=np.random.default_rng(31)
res={b:{'plain':[],'map':[]} for b in BINS}
need={b:25 for b in BINS}
guard=0
while any(v>0 for v in need.values()) and guard<8000:
    guard+=1
    fr=guard%16
    i=int(rng.integers(N)); j=int(rng.integers(N-1)); j+=(j>=i)
    s0=rec['sigs'][fr].astype(np.float64)
    ds=abs(float(s0[i]-s0[j]))
    bb=next((b for b in BINS if b[0]<=ds<b[1]),None)
    if bb is None or need[bb]<=0: continue
    x0=rec['x'][fr].astype(np.float64)
    dv=x0[i]-x0[j]; dv-=L*np.round(dv/L)
    if float((dv**2).sum())**0.5>2.0: continue
    need[bb]-=1
    for arm,fl in (('plain',None),('map',flow)):
        x=x0.copy(); sg=s0.copy()
        seed_numba(70000+guard)
        W=escorted_work(x,sg,L,beta,i,j,fl,NSTEPS,SWEEPS_TOTAL//NSTEPS,3.0)
        W=W[0] if isinstance(W,tuple) else W
        res[bb][arm].append(float(W))
    torch.save(res,OUT)
print('--- paired dW (map - plain) per bin ---', flush=True)
for b in BINS:
    p=np.array(res[b]['plain']); m=np.array(res[b]['map'])
    if len(p)<3: continue
    d=m-p
    accp=np.minimum(1,np.exp(np.clip(-beta*p,-700,0))).mean()
    accm=np.minimum(1,np.exp(np.clip(-beta*m,-700,0))).mean()
    print(f'{b}: n={len(p)} Wmed {np.median(p):+.2f}->{np.median(m):+.2f} | dWmed {np.median(d):+.3f} frac_improved {np.mean(d<0):.2f} | acc {accp:.4f}->{accm:.4f}', flush=True)
print('PROBE DONE', flush=True)
