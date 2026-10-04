#!/usr/bin/env python3
"""Evaluate selection-only changes with fixed tracking state evolution."""
import argparse
import concurrent.futures
import itertools
import json
import math
import pathlib
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary',type=pathlib.Path,required=True)
p.add_argument('--suite',type=pathlib.Path,required=True)
p.add_argument('--parameters',type=pathlib.Path,required=True)
p.add_argument('--output',type=pathlib.Path,required=True)
p.add_argument('--workers',type=int,default=16)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
suite=json.loads(a.suite.read_text());parameters=json.loads(a.parameters.read_text())
fields=['missing_seconds','close_range_missing_seconds','longest_missing_seconds','false_track_seconds','close_range_position_rmse_metres','motion_lag_rms_seconds','motion_lag_absolute_seconds']
def evaluate(name,config):
 path=a.output/(name+'.json');path.write_text(json.dumps(config,indent=2));destination=a.output/name
 command=['nice','-n','10',str(a.binary),'--train',*suite['train'],'--validation',*suite['validation'],'--parameters',str(a.parameters),'--evaluation-parameters',str(path),'--trials','1','--output',str(destination)]
 with (a.output/(name+'.log')).open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
 # A fixed candidate is the evaluation BASELINE. The one trial's optimized
 # values are intentionally ignored, as are all diagnostic validation results.
 return json.loads((destination/'report.json').read_text())
baseline=evaluate('baseline',parameters)
def violations(candidate,reference,per_clip):
 result={}
 for key in fields:
  c,b=candidate[key],reference[key]
  margin=(.01 if key=='close_range_position_rmse_metres' else .04 if key.startswith('motion_lag_') else 0.) if per_clip else 0.
  epsilon=1024.*2.220446049250313e-16*max(1.,reference['labelled_seconds'] if key.endswith('seconds') and not key.startswith('motion_lag_') else b or 0.)
  if b is not None and (c is None or not math.isfinite(c) or c>b+margin+epsilon):result[key]=[b,c,margin]
 return result
def run(item):
 index,(weight,cap)=item;config=dict(parameters);config.update(hypothesis_uncertainty_weight=weight,selection_confidence_cap=cap);name=f'candidate-{index:03}';report=evaluate(name,config);score=report['training']['baseline'];failed={}
 for path,c,b in zip(suite['train'],report['training_per_recording'],baseline['training_per_recording']):
  bad=violations(c['baseline'],b['baseline'],True)
  if bad:failed[path]=bad
  for key in fields[:4]:
   if abs(c['baseline'][key]-b['baseline'][key])>1e-9:raise AssertionError(f'Selection changed availability: {key} {path}')
 bad=violations(score,baseline['training']['baseline'],False)
 if bad:failed['aggregate']=bad
 result={'weight':weight,'cap':cap,'score':score,'failures':failed,'parameters':str(a.output/(name+'.json')),'report':str(a.output/name/'report.json')}
 print(name,weight,cap,'loss',round(score['loss'],6),'failures',len(failed),flush=True)
 return result
weights=[0.,.01,.03,.05,.1,.2,.3,.5,1.,2.,3.,5.,10.,20.]
caps=[0.,3.,4.,5.,6.,8.,10.,15.,20.,30.,50.,100.]
with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:results=list(pool.map(run,enumerate(itertools.product(weights,caps))))
results.sort(key=lambda x:x['score']['loss']);eligible=[r for r in results if not r['failures']]
(a.output/'summary.json').write_text(json.dumps({'baseline':baseline['training']['baseline'],'results':results,'selected':eligible[0] if eligible else None,'selection':'48-clip development metrics only; diagnostic validation ignored'},indent=2))
