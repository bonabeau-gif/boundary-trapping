import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import json
from pathlib import Path
base=Path(__file__).resolve().parent
rng=np.random.default_rng(20261019)
n=160000
rho=np.sqrt(rng.random(n)); theta=2*np.pi*rng.random(n)
u=rho*np.cos(theta); v=rho*np.sin(theta)
loss=1-np.sqrt(1-v*v)
mean=1-8/(3*np.pi)
xs=np.linspace(0,1,450)
V=np.sqrt(np.maximum(0,2*xs-xs*xs))
cdf=(2/np.pi)*(np.arcsin(V)+(1-xs)*V)
emp=np.searchsorted(np.sort(loss),xs,side='right')/n
print('theoretical mean',mean, 'MC mean', loss.mean(),'difference',loss.mean()-mean)
print('CDF max diff',np.max(np.abs(cdf-emp)))
for t in [0.05,0.1,0.2,0.3,0.5]:
 Vt=np.sqrt(2*t-t*t); Ft=(2/np.pi)*(np.arcsin(Vt)+(1-t)*Vt)
 print('P loss >',t, 'theory',1-Ft, 'MC',np.mean(loss>t))
fig,ax=plt.subplots(figsize=(8,4.3))
ax.plot(xs,cdf,label='Exact distribution',linewidth=2)
ax.plot(xs,emp,'--',label='Uniform-disk numerical sample',linewidth=1.6,alpha=.8)
ax.set_xlabel('Asymptotic fitness loss  $L=1-f(\\theta_*)$')
ax.set_ylabel('Probability  $\\Pr(L\\leq \\ell)$')
ax.set_xlim(0,1);ax.set_ylim(0,1)
ax.grid(alpha=.20)
ax.legend(frameon=False)
ax.text(.53,.38,f'Mean loss: {mean:.4f}',fontsize=10)
fig.tight_layout()
fig.savefig(base/'figures/fig5_quantified_loss.png',dpi=180)
plt.close(fig)
# Cross-check harmonic conjugate integrals for randomized polynomial fitness and both closed fields
z=(rng.uniform(-.8,.8,300)+1j*rng.uniform(-.8,.8,300))
z=z[np.abs(z)<.95]
coeff=np.array([0.4+0.7j,-.5+.15j,.2-.3j,.9+.35j])
# F(z)=sum coeff z^k, tildeF(z)=sum coeff/k z^k
F_prime=sum(k*c*z**(k-1) for k,c in enumerate(coeff,1))
T_prime=sum(c*z**(k-1) for k,c in enumerate(coeff,1))
aF=((1-abs(z)**2)**2/2)*np.conj(F_prime)
aM=((1-abs(z)**2)/2)*np.conj(T_prime)
errF=max(abs(np.imag(F_prime*aF)))
errM=max(abs(np.imag(T_prime*aM)))
print('first integral rate error',errF,errM)
result={'n_uniform_disk_samples':n,'theoretical_mean_loss':mean,'mc_mean_loss':float(loss.mean()),'cdf_sup_error':float(np.max(np.abs(cdf-emp))),'conjugate_F_rate_max':float(errF),'conjugate_M_rate_max':float(errM)}
(base/'results.json').write_text(json.dumps(result,indent=2))
