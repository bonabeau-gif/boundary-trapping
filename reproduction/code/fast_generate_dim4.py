"""Generate dimension-4 physical-time comparison (Figure 7) using stable gradient-time integration.
Avoids integrating enormous physical time directly in the ODE solver.
"""
import numpy as np
from scipy.integrate import solve_ivp,cumulative_trapezoid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
ROOT=Path(__file__).resolve().parent
FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
J4=lambda u,v: .5*(3-u*u-v*v)*u

def rhs(tau,y):
 u,v=y
 return [1.5-1.5*u*u-.5*v*v,-u*v]

def data_dim4(u0,v0,tau_end=5.0):
 sol=solve_ivp(rhs,(0,tau_end),(u0,v0),rtol=1e-12,atol=1e-13,dense_output=True)
 tau=np.linspace(0,tau_end,18000)
 u,v=sol.sol(tau)
 q=np.maximum(1-u*u-v*v,1e-15)
 physical=.001+cumulative_trapezoid(9/q**2,tau,initial=0)
 return np.log10(physical), J4(u,v),v

inits=[(0.0,0.7),(-0.6,0.6),(0.2,0.7),(0.7,0.7)]
S=np.linspace(-3*np.log(10),12*np.log(10),900)
fig,ax=plt.subplots(1,2,figsize=(10,4))
colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
for (u0,v0),col in zip(inits,colors):
 tx,fitness,v=data_dim4(u0,v0)
 ax[0].plot(tx,fitness,color=col,label=f'({u0}, {v0})')
 ax[1].plot(tx,v,color=col)
 c=np.sqrt(1-v0*v0)
 sol2=solve_ivp(lambda s,y:[np.exp(s)*.5*(c*c-y[0]*y[0])**2],(S[0],S[-1]),[u0],method='LSODA',rtol=1e-10,atol=1e-13,dense_output=True)
 ax[0].plot(S/np.log(10),sol2.sol(S)[0],color=col,linestyle='--',linewidth=1)
 ax[1].plot(S/np.log(10),np.full_like(S,v0),color=col,linestyle='--',linewidth=1)
tref=np.logspace(6,12,50)
ax[1].plot(np.log10(tref),.09*(tref/1e6)**(-1/6),color='k',linestyle=':',linewidth=1)
ax[1].text(9.3,.07,r'$\propto t^{-1/6}$',fontsize=9)
ax[0].set_xlabel('log$_{10}$ physical time $t$');ax[0].set_ylabel('Projected mean fitness $J$')
ax[1].set_xlabel('log$_{10}$ physical time $t$');ax[1].set_ylabel('Transverse parameter $v$')
ax[0].legend(frameon=False,title='$(u_0,v_0)$',fontsize=9)
ax[0].set_xlim(-3,12);ax[1].set_xlim(-3,12);ax[1].set_yscale('log');ax[1].set_ylim(1e-2,1)
ax[1].legend([Line2D([],[],color='k'),Line2D([],[],color='k',ls='--')],['$d=4$','$d=2$'],frameon=False,fontsize=9,loc='lower left')
fig.tight_layout()
fig.savefig(FIG/'fig7_dim4_physical_time_fast.png',dpi=200)
plt.close(fig)
print('Generated',FIG/'fig7_dim4_physical_time_fast.png')
