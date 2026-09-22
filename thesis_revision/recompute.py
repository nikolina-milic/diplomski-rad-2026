import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from spr.evaluation.experiments import novel_attack
import numpy as np
from scipy.stats import wilcoxon
out=Path(__file__).parent
runs={}
for seed in range(1,11):
    p=out/f'novel_{seed}.json'
    if p.exists(): r=json.loads(p.read_text())
    else:
        r=novel_attack(seed=seed)
        p.write_text(json.dumps(r,indent=2))
    runs[str(seed)]=r
    print('completed seed',seed,flush=True)
summary={}
for pattern in runs['1']:
    row={}
    for key in runs['1'][pattern]:
        a=np.array([r[pattern][key] for r in runs.values()])
        row[key]={'mean':float(a.mean()),'std':float(a.std(ddof=1)),'min':float(a.min()),'max':float(a.max())}
    a=[r[pattern]['ml'] for r in runs.values()]
    for key in ['rules','hybrid_stacking','hybrid_cascade']:
        b=[r[pattern][key] for r in runs.values()]
        row[key]['p_vs_ml']=float(wilcoxon(a,b).pvalue)
    summary[pattern]=row
(out/'novel_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2),flush=True)
