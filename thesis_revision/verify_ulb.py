import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import spr.realdata.ulb as u
out=Path(__file__).parent
frame=u.load_ulb('C:/Users/Korisnik/Downloads/creditcard.csv')
orig=u.compute_metrics
rows=[]
for seed in range(10):
    saved=[]
    def capture(y,s,threshold):
        saved.append(np.asarray(s)>=threshold)
        return orig(y,s,threshold)
    u.compute_metrics=capture
    r=u._evaluate_once(frame,seed,.2,100,3)
    row={'seed':seed,'different_predictions_vs_ml':{name:int(np.sum(saved[1]!=saved[i])) for i,name in enumerate(u.APPROACHES) if i>1},'approaches':r['approaches']}
    rows.append(row)
    (out/'ulb_prediction_check.json').write_text(json.dumps(rows,indent=2))
    print(row['seed'],row['different_predictions_vs_ml'],flush=True)
