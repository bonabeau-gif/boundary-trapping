"""Fast independent check for dimensional convergence: gradient time + physical-time quadrature.
No fitting or external data; reproduces dimension-4 landmarks and slopes.
"""
import json
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from pathlib import Path
P=Path(__file__).resolve().parent
u0,v0=0.2,0.7

def rhs(tau,y):
 u,v=y
 return [1.5-1.5*u*u-.5*v*v,-u*v]
sol=solve_ivp(rhs,(0,4.3),(u0,v0),dense_output=True,atol=5e-13,rtol=5e-13)
tau=np.linspace(0,4.3,14000);uv=sol.sol(tau)
u,v=uv
q=1-u*u-v*v
assert np.min(q)>0
physical=cumulative_trapezoid(9/q**2,tau,initial=0)
fitness=.5*(3-u*u-v*v)*u
landmarks={}
for tt in [1e2,1e4,1e7,1e10]:
 assert tt<=physical[-1],(tt,physical[-1])
 landmarks[str(tt)]={'v':float(np.interp(tt,physical,v)),'fitness_gap':float(np.interp(tt,physical,1-fitness))}
print('physical t last=',physical[-1]);print(landmarks)
(P/'dim4_fast_results.json').write_text(json.dumps(landmarks,indent=2))
