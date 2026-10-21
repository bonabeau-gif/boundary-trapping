"""Independent check of false conserved transverse moment and mutation effects.

Full selection is exact normalized exponential tilting. Two closures use
analytical Fisher projection and moment closure, respectively.
For reset mutation the full model is solved by Strang splitting with exact
selection and reset substeps; step convergence is checked in the driver.
"""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import logsumexp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parent
FIG=BASE/'figures'; RES=BASE/'results'
FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
N=4096
TH=np.linspace(-np.pi,np.pi,N,endpoint=False)
COS=np.cos(TH); SIN=np.sin(TH)
U=np.ones(N)/N

def wc_mass(a):
 u,v=a.real,a.imag
 return (1-abs(a)**2)/((1+abs(a)**2-2*(u*COS+v*SIN))*N)

def full_select_moments(times,a0):
 init=wc_mass(a0)
 ans=[]
 for t in times:
  z=np.log(init)+t*COS
  p=np.exp(z-logsumexp(z))
  ans.append(np.sum(p*np.exp(1j*TH)))
 return np.array(ans)

def fisher(t,y,gamma=0,operator='reset'):
 u,v=y;r2=u*u+v*v
 m=(1-r2)
 out=np.array([0.5*m*m,0.])
 if gamma>0:
  if operator=='reset':out-=gamma*m*np.array([u,v])
  if operator=='diffusion':out-=gamma*(1+r2)/m*np.array([u,v])
 return out

def moments(t,y,gamma=0):
 u,v=y;r2=u*u+v*v
 return [0.5*(1-r2)-gamma*u,-gamma*v]

def solve_model(rhs,a0,times,rate=0,op='reset'):
 if rhs==fisher:
  s=solve_ivp(lambda t,y:rhs(t,y,rate,op),(float(times[0]),float(times[-1])),[a0.real,a0.imag],
              t_eval=times,rtol=2e-11,atol=3e-13,max_step=1)
 else:
  s=solve_ivp(lambda t,y:rhs(t,y,rate),(float(times[0]),float(times[-1])),[a0.real,a0.imag],
              t_eval=times,rtol=2e-11,atol=3e-13,max_step=1)
 if not s.success:raise RuntimeError(s.message)
 return s.y[0]+1j*s.y[1]

def split_reset(times,a0,gamma,dt):
 init=wc_mass(a0)
 mass=init.copy();now=0.
 means=[]
 for tgt in times:
  while now<tgt-1e-13:
   h=min(dt,tgt-now)
   # exact selection half step + exact uniform reset full step + selection half
   for half in [True,False]:
    weights=mass*np.exp((h/2)*COS)
    mass=weights/weights.sum()
    if half:mass=np.exp(-gamma*h)*mass+(1-np.exp(-gamma*h))*U
   now+=h
  means.append((mass*np.exp(1j*TH)).sum())
 return np.array(means)

def check_projected_mutation_formula():
    # Independent score inner product, evaluated by circular quadrature.
    a=.25+.50j
    u,v=a.real,a.imag
    r2=u*u+v*v
    den=1+r2-2*(u*COS+v*SIN)
    p=(1-r2)/(N*den)
    scores=np.column_stack((
        -2*u/(1-r2)+2*(COS-u)/den,
        -2*v/(1-r2)+2*(SIN-v)/den))
    fisher_inv=(1-r2)**2/2
    mut_reset=fisher_inv*np.sum(scores*(U-p)[:,None],axis=0)
    pred_reset=-(1-r2)*np.array([u,v])
    assert np.max(np.abs(mut_reset-pred_reset))<1e-11
    kk=np.fft.fftfreq(N,d=1/N)
    d2p=np.fft.ifft(-(kk**2)*np.fft.fft(p)).real
    mut_diff=fisher_inv*np.sum(scores*d2p[:,None],axis=0)
    pred_diff=-(1+r2)/(1-r2)*np.array([u,v])
    assert np.max(np.abs(mut_diff-pred_diff))<1e-10
    return {'reset_error':float(np.max(abs(mut_reset-pred_reset))),
            'diffusion_error':float(np.max(abs(mut_diff-pred_diff)))}

def main():
 projection_errors=check_projected_mutation_formula()
 a0=0+0.7j
 ts=np.linspace(0,35,281)
 full=full_select_moments(ts,a0)
 fish=solve_model(fisher,a0,ts)
 mom=solve_model(moments,a0,ts)
 # finite difference / quadratic Taylor validation
 h=2e-3
 z=full_select_moments(np.array([0,h,2*h]),a0).imag
 d2=(z[2]-2*z[1]+z[0])/h**2
 expected=-a0.imag*(1-abs(a0)**2)/4
 assert abs(d2-expected)<1e-3,(d2,expected)
 assert np.max(abs(mom.imag-a0.imag))<2e-11
 assert np.max(abs(fish.imag-a0.imag))<2e-11
 assert full[-1].imag<0.05

 fig,ax=plt.subplots(figsize=(7.1,3.5))
 ax.plot(ts,full.imag,label='Unrestricted selection (exact)',lw=2.3)
 ax.plot(ts,mom.imag,'--',label='Circular-moment closure',lw=1.8)
 ax.plot(ts,fish.imag,':',label='Fisher projection',lw=2.2)
 ax.set(xlabel='Selection time',ylabel=r'Transverse moment $E[\sin\theta]$',ylim=(-.03,.75))
 ax.legend(fontsize=9);ax.grid(alpha=.15)
 fig.tight_layout();fig.savefig(FIG/'fig6_false_invariant.png',dpi=210);plt.close(fig)

 # Mutation: choose weak reset rate to show singular no-mutation limit.
 gamma=.05
 tt=np.linspace(0,180,361)
 ffish=solve_model(fisher,a0,tt,gamma)
 mmom=solve_model(moments,a0,tt,gamma)
 x_full=split_reset(tt,a0,gamma,dt=.1)
 x_fine=split_reset(tt,a0,gamma,dt=.05)
 x_half=split_reset(tt,a0,gamma,dt=.025)
 err_coarse=max(abs(x_full-x_half));err_fine=max(abs(x_fine-x_half))
 print('Strang dt errors',err_coarse,err_fine)
 assert err_fine<0.35*err_coarse,(err_coarse,err_fine)
 assert abs(mmom[-1].imag)<0.001
 fisher_tail=solve_model(fisher,a0,np.array([0.,1200.]),gamma)[-1]
 assert abs(fisher_tail.imag)<0.005
 eq=-gamma+np.sqrt(1+gamma*gamma)
 # stationary reset for both closures: both u→same eq.
 assert abs(fisher_tail.real-eq)<0.01
 assert abs(mmom[-1].real-eq)<0.025
 fig,axs=plt.subplots(1,2,figsize=(9.0,3.25))
 for ax,get,lab in [(axs[0],np.imag,'Transverse moment'),(axs[1],np.real,'Axial moment')]:
  ax.plot(tt,get(x_fine),label='Full selection + reset',lw=2.25)
  ax.plot(tt,get(ffish),'--',label='Fisher + reset',lw=1.9)
  ax.plot(tt,get(mmom),':',label='Moment closure + reset',lw=2.15)
  ax.set(xlabel='Time',ylabel=lab);ax.grid(alpha=.16)
 axs[1].legend(loc='lower right',fontsize=7.6)
 fig.tight_layout();fig.savefig(FIG/'fig7_mutation_restores_motion.png',dpi=210);plt.close(fig)
 data={
 'initial_parameter':[a0.real,a0.imag],
 'exact_initial_second_derivative_transverse':expected,
 'numeric_second_derivative_transverse':float(d2),
 'full_selection_transverse_at_t35':float(full[-1].imag),
 'fisher_selection_transverse_at_t35':float(fish[-1].imag),
 'moment_closure_transverse_at_t35':float(mom[-1].imag),
 'mutation_operator':'reset', 'reset_rate':gamma,
 'independent_score_projection_checks':projection_errors,
 'full_reset_transverse_at_t180':float(x_fine[-1].imag),
 'fisher_reset_transverse_at_t180':float(ffish[-1].imag),
 'moment_reset_transverse_at_t180':float(mmom[-1].imag),
 'full_reset_axial_at_t180':float(x_fine[-1].real),
 'fisher_reset_axial_at_t180':float(ffish[-1].real),
 'fisher_reset_at_t1200': [float(fisher_tail.real),float(fisher_tail.imag)],
 'fisher_transverse_local_decay_rate':float(gamma*(1-eq*eq)),
 'weak_mutation_transverse_rate_prediction':float(2*gamma*gamma),
 'moment_reset_axial_at_t180':float(mmom[-1].real),
 'closure_expected_axial_equilibrium':float(eq),
 'strang_step_0.1_error_vs_step_0.025':float(err_coarse),
 'strang_step_0.05_error_vs_step_0.025':float(err_fine),
 'notes':'Full reset equation is simulated by Strang splitting; no finite-population biological simulation is claimed.'}
 (RES/'false_invariant_mutation.json').write_text(json.dumps(data,indent=2))
 print(json.dumps(data,indent=2))

if __name__=='__main__':main()
