"""Analytical controls: d=4 global direction recovery, exponential tilting, Watson repair.
All values are evaluated independently of the earlier Paper_A_project code.
"""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import ive
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
F=R/'figures';O=R/'results';F.mkdir(exist_ok=True);O.mkdir(exist_ok=True)

def fun_d4(t,uv):
 u,v=uv;r2=u*u+v*v
 return [1.5-.5*r2-u*u,-u*v]

def watson_stats(k):
 z0=ive(0,k);m=ive(1,k)/z0;m4=ive(2,k)/z0
 return m, .5*(1+m4)-m*m

def watson_equilibrium(scheme,rate):
 decay=4*rate if scheme=='diffusion' else rate
 k=brentq(lambda k: .5-decay*watson_stats(k)[0]/watson_stats(k)[1],1e-8,100)
 m,_=watson_stats(k)
 return {'kappa':float(k),'m2':float(m)}

def exact_reset(rate):return np.sqrt(1+4*rate*rate)-2*rate

def exact_diff(rate):
 from scipy.linalg import eigh
 K=62;kk=np.arange(-K,K+1,2)
 L=np.diag(.5-rate*kk*kk)+np.diag(np.full(len(kk)-1,.25),1)+np.diag(np.full(len(kk)-1,.25),-1)
 vals,vecs=eigh(L,subset_by_index=(len(kk)-1,len(kk)-1))
 q=vecs[:,0]
 return float(q[np.where(kk==2)[0][0]]/q[np.where(kk==0)[0][0]])

def run():
 fig,axes=plt.subplots(1,2,figsize=(9.1,4))
 tested=[]
 for u,v in [(0,.7),(-.6,.6),(.2,.7),(.7,.7)]:
  sol=solve_ivp(fun_d4,(0,35),(u,v),rtol=1e-11,atol=1e-12,dense_output=True,max_step=.05)
  t=np.linspace(0,35,500)
  us,vs=sol.sol(t)
  meanf=(3-(us*us+vs*vs))/2*us
  axes[0].plot(t,meanf,label=f'({u:g}, {v:g})')
  axes[1].plot(t,vs,label=f'({u:g}, {v:g})')
  tested.append({'a0':[u,v],'tau_end':35,'u_end':float(us[-1]),'v_end':float(vs[-1]),'fitness_end':float(meanf[-1])})
 axes[0].set_xlabel('Rescaled gradient time τ');axes[0].set_ylabel('Mean directional fitness')
 axes[1].set_xlabel('Rescaled gradient time τ');axes[1].set_ylabel('Transverse parameter v')
 axes[0].legend(frameon=False,fontsize=8)
 fig.tight_layout();fig.savefig(F/'fig5_dim4_comparison.png',dpi=185);plt.close(fig)
 compared=[]
 for scheme in ['diffusion','reset']:
  for rate in [.1,.3,.6]:
   cont=exact_diff(rate) if scheme=='diffusion' else exact_reset(rate)
   wat=watson_equilibrium(scheme,rate)
   # wrapped Cauchy positive stable projected equilibrium
   if scheme=='diffusion':
    rho=brentq(lambda y: (1-y)**3-2*rate*(1+y),0,1) if rate<.5 else 0.
   else:rho=max(0,1-2*rate)
   compared.append({'scheme':scheme,'rate':rate,'true_stationary_m2':float(cont),
      'cauchy_m2':float(rho),'cauchy_m1':float(np.sqrt(rho)),
      'watson_m2':wat['m2'],'watson_kappa':wat['kappa'],
      'cauchy_error_m2':float(rho-cont),'watson_error_m2':float(wat['m2']-cont)})
 (O/'model_controls.json').write_text(json.dumps({'d4':tested,'watson':compared},indent=2))
 print('Dimension 4 after rescaled τ=35:')
 for d in tested:print(d)
 print('Watson repair:')
 for q in compared: print(q)
if __name__=='__main__':run()
