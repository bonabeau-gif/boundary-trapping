"""Independent exact/numerical verification of Poisson-boundary trapping for wrapped Cauchy.
Python >=3.10; numpy scipy matplotlib. No stochastic simulation is needed for these claims.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from scipy.integrate import solve_ivp, trapezoid
from scipy.special import i0e
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
FIG=R/'figures'; RESULT=R/'results'
FIG.mkdir(exist_ok=True); RESULT.mkdir(exist_ok=True)
TH=np.linspace(-np.pi,np.pi,8192,endpoint=False)
DX=2*np.pi/len(TH)

def wrapped_cauchy(theta,u,v):
 r2=u*u+v*v
 return (1-r2)/(2*np.pi*(1+r2-2*(u*np.cos(theta)+v*np.sin(theta))))

def full_density(t,u0,v0, fitness=None):
 if fitness is None:fitness=np.cos(TH)
 pp=wrapped_cauchy(TH,u0,v0)
 logw=np.log(pp)+t*fitness
 zz=np.exp(logw-logw.max())
 return zz/(DX*zz.sum())

def evolve_directional(u0,v0,T=300.0):
 c2=1-v0*v0
 def rhs(t,y):return [0.5*(c2-y[0]**2)**2]
 sol=solve_ivp(rhs,[0,T],[u0],rtol=3e-12,atol=1e-13,dense_output=True,max_step=5)
 return sol

def J_fourier(u,v,cos_coefs=(),sin_coefs=()):
 z=u+1j*v
 zpow=1+0j
 val=0.
 for k in range(1,max(len(cos_coefs),len(sin_coefs))+1):
  zpow*=z
  if k<=len(cos_coefs):val+=cos_coefs[k-1]*zpow.real
  if k<=len(sin_coefs):val+=sin_coefs[k-1]*zpow.imag
 return val

def grad_J(u,v,cc=(),ss=()):
 z=u+1j*v; dz=0j
 for k in range(1,max(len(cc),len(ss))+1):
  c=cc[k-1] if k<=len(cc) else 0
  s=ss[k-1] if k<=len(ss) else 0
  dz += k*(c-1j*s)*z**(k-1)
 return dz.real, -dz.imag # derivative in y is -Im complex analytic derivative

def generic_flow(u0,v0,cc,ss=(),rstop=1-1e-9, tau_max=50):
 def rhs(t,y): return grad_J(y[0],y[1],cc,ss)
 def hit(t,y):return np.linalg.norm(y)-rstop
 hit.terminal=True;hit.direction=1
 sol=solve_ivp(rhs,[0,tau_max],[u0,v0],events=[hit],rtol=2e-11,atol=1e-12,max_step=.03)
 if len(sol.t_events[0]):
  a=sol.y_events[0][0]; angle=float(np.arctan2(a[1],a[0]))
  return {'hit':True,'angle':angle,'fitness':float(fboundary(angle,cc,ss)),'tau':float(sol.t_events[0][0]),'initial':[u0,v0]},sol
 return {'hit':False,'initial':[u0,v0], 'final':sol.y[:,-1].tolist()},sol

def fboundary(theta,cc,ss=()):
 return sum(c*np.cos((k+1)*theta) for k,c in enumerate(cc))+sum(s*np.sin((k+1)*theta) for k,s in enumerate(ss))

def projected_defect(u,v):
 r2=u*u+v*v
 return {'total_variance':.5*(1-r2),'represented_variance':.5*(1-r2)**2,'squared_defect':.5*r2*(1-r2),'represented_fraction':1-r2,'omitted_fraction':r2}

def exact_one_parameter_tilt(t,theta,u0,v0):
 pp=wrapped_cauchy(theta,u0,v0); w=pp*np.exp(t*(np.cos(theta)-1))
 return w/(DX*w.sum())

def run():
 plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
 outcomes=[]
 ts=np.array([0,1,5,10,30,100,300.],float)
 u0=0.;v0=.7
 sol=evolve_directional(u0,v0,T=300)
 for t in ts:
  uu=float(sol.sol(t)[0]); p=full_density(t,u0,v0)
  full_m=float(DX*np.sum(p*np.cos(TH))); full_var=float(DX*np.sum(p*np.cos(TH)**2)-full_m**2)
  outcomes.append({'t':float(t),'projected_mean':uu,'full_mean':full_m,'full_var':full_var,**projected_defect(uu,v0)})
 # Figure 1: density comparison
 fig,axes=plt.subplots(2,2,figsize=(10,6.6),sharex=True)
 for ax,t in zip(axes.flat,[0,5,30,100]):
  uu=float(sol.sol(t)[0]); pf=full_density(t,u0,v0); pc=wrapped_cauchy(TH,uu,v0)
  ax.plot(TH,pf,label='Full replicator / exact tilt',linewidth=1.8)
  ax.plot(TH,pc,label='Fisher-projected wrapped Cauchy',linewidth=1.8,ls='--')
  ax.axvline(np.arcsin(v0),lw=.8,alpha=.7,color='gray')
  ax.text(.025,.93,f't = {t}',transform=ax.transAxes,va='top',fontsize=9)
  ax.set_xlim(-.4,2.2 if t>=5 else np.pi)
  ax.set_ylabel('Density')
 axes[1,0].set_xlabel('Trait angle θ');axes[1,1].set_xlabel('Trait angle θ')
 axes[0,0].legend(fontsize=8,frameon=False)
 fig.tight_layout();fig.savefig(FIG/'fig1_unimodal_same_initial.png',dpi=180);plt.close(fig)
 # Figure 2: time series exact from same initial vs projected for v0 values
 fig,ax=plt.subplots(figsize=(8,4.3))
 for v0 in [0.,.3,.7,.9]:
  s=evolve_directional(0,v0,300)
  vals=np.maximum(0,s.sol(np.linspace(0,300,700))[0])
  ax.plot(np.linspace(0,300,700),vals,label=f'Projected, v₀={v0:.1f}; limit={np.sqrt(1-v0**2):.3f}')
  # exact density for moderate times (p_t no longer WC)
  tt=np.array([0,1,2,5,10,20,50,100,300])
  fm=[float((full_density(t,0,v0)*np.cos(TH)).sum()*DX) for t in tt]
  ax.plot(tt,fm,ls='none',marker='.',markersize=3,alpha=.7)
 ax.set_ylim(-.04,1.045);ax.set_xlim(0,300)
 ax.set_xlabel('Selection time');ax.set_ylabel('Mean fitness: E[cos θ]')
 ax.legend(loc='lower right',frameon=False,fontsize=8)
 fig.tight_layout();fig.savefig(FIG/'fig2_fitness_gap.png',dpi=200);plt.close(fig)
 # Figure 3: flow field in parameter disk directional and generic analytic perturbation
 fig,axes=plt.subplots(1,2,figsize=(10,4.6))
 configs=[('Unique optimum: f(θ) = cos θ',(1,),()),('Perturbed fitness: cos θ + 0.2 cos 2θ',(1,.2),())]
 for ax,(title,cc,ss) in zip(axes,configs):
  ph=np.linspace(0,2*np.pi,500);ax.plot(np.cos(ph),np.sin(ph),c='black',lw=1.2)
  U,V=np.meshgrid(np.linspace(-.9,.9,24),np.linspace(-.9,.9,24));GU=np.zeros_like(U);GV=np.zeros_like(V)
  for idx in np.ndindex(U.shape):GU[idx],GV[idx]=grad_J(U[idx],V[idx],cc,ss)
  mask=(U*U+V*V)<.92**2
  denom=np.hypot(GU,GV)+1e-9
  ax.quiver(U[mask],V[mask],GU[mask]/denom[mask],GV[mask]/denom[mask],width=.003,alpha=.4,color='steelblue')
  for u_init in [-.6,0,.5]:
   for v_init in [-.7,-.3,.3,.7]:
    if u_init*u_init+v_init*v_init>=.95**2:continue
    rec,s=generic_flow(u_init,v_init,cc,ss,rstop=1-1e-6)
    ax.plot(s.y[0],s.y[1],lw=1.4)
    ax.plot([u_init],[v_init],marker='.',color='black',markersize=2)
  ax.plot([1],[0],marker='*',color='red',markersize=11)
  ax.text(.04,.96,'(a)' if len(cc)==1 else '(b)',transform=ax.transAxes,va='top',fontsize=10);ax.set_xlabel('u');ax.set_ylabel('v')
  ax.set_aspect('equal');ax.set_xlim(-1.1,1.1);ax.set_ylim(-1.1,1.1)
 fig.tight_layout();fig.savefig(FIG/'fig3_generic_boundary_trap.png',dpi=200);plt.close(fig)
 # Figure 4: missed selection proportion vs absolute defect in unique-peak example
 uvals=np.linspace(0,.71414284,350); v=.7;r2=uvals**2+v**2
 fig,ax=plt.subplots(figsize=(7,4))
 ax.plot(uvals,.5*r2*(1-r2),label='Absolute squared invariance defect')
 ax.plot(uvals,r2,label='Fraction of selection signal missed',ls='--')
 ax.plot(uvals,.5*(1-r2)**2,label='Fisher-represented selection variance',ls=':')
 ax.set_xlabel('Projected u; transverse component v = 0.7 fixed')
 ax.set_ylabel('Value (variance scale or dimensionless fraction)')
 ax.legend(frameon=False,fontsize=9)
 fig.tight_layout();fig.savefig(FIG/'fig4_false_certification.png',dpi=200);plt.close(fig)
 # Numeric generic under analytic perturbations
 generic=[]
 for cc,ss in [((1,),()), ((1,.2),()), ((1,.2,-.08),()), ((1,.1,.05,-.03),()),
              ((1,.2),(.12,-.08)), ((.85,-.1,.07),(.2,.06,-.01))]:
  fitness_max=max(fboundary(np.linspace(-np.pi,np.pi,10000),cc,ss))
  for uu,vv in [(-.2,.7),(0,.7),(.2,.7),(-.3,-.4)]:
   out,_=generic_flow(uu,vv,cc,ss,rstop=1-1e-9)
   out.update({'cos_coefficients':cc,'sin_coefficients':ss,'global_fitness_max':float(fitness_max),'fitness_shortfall':float(fitness_max-out.get('fitness',np.nan))})
   generic.append(out)
 # numerical derivative and identity consistency tests
 errors={}
 for v in [.1,.3,.7,.9]:
  for u in [-np.sqrt(1-v*v)*.9, 0.,np.sqrt(1-v*v)*.8]:
   th=TH;p=wrapped_cauchy(th,u,v)
   mean=float(DX*(p*np.cos(th)).sum())
   variance=float(DX*(p*np.cos(th)**2).sum()-mean**2)
   errors[f'u={u:.4f},v={v:.4f}']={'mean_err':abs(mean-u),'variance_err':abs(variance-(1-u*u-v*v)/2)}
 max_err=max(max(x.values()) for x in errors.values())
 # explicit exact-selection exponential tilt consistency
 for t in [.5,2.,5.,10.]:
  diff=np.max(np.abs(full_density(t,0,.7)-exact_one_parameter_tilt(t,TH,0,.7)))
  assert diff<1e-12,(t,diff)
 assert max_err<2e-10,max_err
 assert abs(np.sqrt(1-.7**2)-.714142842854285)<1e-12
 assert all(z['hit'] and z['fitness_shortfall']>0.001 for z in generic)
 record={'example_t':outcomes,'generic_perturbations':generic,'wrapped_moment_max_numeric_err':max_err,
        'closed_form_limit_u':float(np.sqrt(1-.7**2)),
        'full_limit_mean_fitness':1.,
        'projected_long_run_fitness_loss':float(1-np.sqrt(1-.7**2)),
        'notes':'Exact one-dimensional tilt reproduces full replicator from exactly the same initial wrapped-Cauchy distribution. Cauchy Fisher projection generally does not.'}
 (RESULT/'independent_numeric_checks.json').write_text(json.dumps(record,indent=2))
 print('Derived / tested unimodal gap:',record['projected_long_run_fitness_loss'])
 print('Max quadrature identity error:',max_err)
 print('Generic analytic experiments:',len(generic))
 print('Range of checked suboptimal losses:',min(x['fitness_shortfall'] for x in generic),max(x['fitness_shortfall'] for x in generic))
 for x in outcomes: print('t={t:5g} exact={full_mean:.6f} projected={projected_mean:.6f} missed_fraction={omitted_fraction:.6f}'.format(**x))

if __name__=='__main__':run()
