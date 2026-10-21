#!/usr/bin/env python3
"""Deterministic core figures; not a claim to regenerate all embedded paper originals."""
import argparse,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import logsumexp
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from boundary_trapping.circle import wrapped_cauchy,exact_replicator,loss_cdf,loss_mean,integrate_fisher,fisher_velocity
from boundary_trapping.cauchy import lorentzian_endpoint,lorentzian_expected_fitness
from boundary_trapping.mutation import reset_fisher,reset_moment

def save(fig,out,name):
    fig.tight_layout();fig.savefig(out/name,dpi=170);plt.close(fig);print('Wrote',name)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'figures/generated');args=ap.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    th=np.linspace(-np.pi,np.pi,8192,endpoint=False);z=.1+.7j;p0=wrapped_cauchy(th,z)
    fig,ax=plt.subplots(1,3,figsize=(11,3.2));
    for i,t in enumerate((0,5,30)):
        r=integrate_fisher(z,[1.],t_end=max(t,1e-5)).y[:,-1]
        ax[i].plot(th,exact_replicator(th,t,p0,np.cos(th)),label='Unrestricted')
        ax[i].plot(th,wrapped_cauchy(th,complex(*r)),label='Fisher-projected',ls='--')
        ax[i].set_xlim(-1,2);ax[i].set_title(f't={t}')
    ax[0].legend(fontsize=7);save(fig,out,'unimodal_densities.png')
    fig,ax=plt.subplots(figsize=(5,3.5));v=np.linspace(0,.95,250)
    ax.plot(v,1-np.sqrt(1-v*v));ax.set(xlabel='Initial transverse moment |v|',ylabel='Asymptotic fitness loss');save(fig,out,'fitness_loss.png')
    fig,ax=plt.subplots(figsize=(5,3.5));ells=np.linspace(0,1,400)
    ax.plot(ells,loss_cdf(ells),label='Analytical CDF');rng=np.random.default_rng(20261020)
    xx=[]
    for _ in range(3):
        n=60000;rr=np.sqrt(rng.random(n));ang=rng.uniform(-np.pi,np.pi,n);xx.append(1-np.sqrt(1-(rr*np.sin(ang))**2))
    vals=np.sort(np.concatenate(xx));ax.step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',alpha=.5,label='MC 180,000')
    ax.set(xlabel='Limiting loss',ylabel='CDF');ax.legend();save(fig,out,'loss_cdf.png')
    fig,ax=plt.subplots(figsize=(5,5));angles=np.linspace(0,2*np.pi,300);ax.plot(np.cos(angles),np.sin(angles),color='black')
    for yy in np.linspace(-.85,.85,9):
        zz0=0+1j*yy;s=integrate_fisher(zz0,[1.],t_end=300,rtol=1e-9)
        ax.plot(s.y[0],s.y[1],lw=1.2)
    ax.set(aspect='equal',xlabel='u',ylabel='v',title='Cosine Fisher projection');save(fig,out,'harmonic_paths.png')
    fig,ax=plt.subplots(figsize=(6,4));tt=np.geomspace(1e-3,300,600)
    for gamma in (.05,.15):
        for kind,style in ((reset_fisher,'-'),(reset_moment,'--')):
            def rhs(t,y):
                z=complex(*y);dr=kind(z,gamma);return [dr.real,dr.imag]
            sol=solve_ivp(rhs,(0,300),[.1,.7],t_eval=np.r_[0,tt],rtol=1e-8,atol=1e-10)
            ax.plot(sol.t,sol.y[1],style,label=f'{kind.__name__} gamma={gamma}')
    ax.set(xlabel='Time',ylabel='Transverse moment');ax.legend(fontsize=7);save(fig,out,'reset_transient.png')
    fig,ax=plt.subplots(figsize=(6,3.5));mm=np.linspace(-4,4,500)
    for sig in (.5,1,2):
        x=np.array([lorentzian_endpoint(m,sig) for m in mm]);ax.plot(mm,x,label=f'sigma={sig}')
    ax.plot(mm,np.zeros_like(mm),ls=':',c='black',label='Optimum');ax.set(xlabel='Initial location',ylabel='Limiting location');ax.legend();save(fig,out,'cauchy_wrong_location.png')
    print('Core deterministic figure reproduction completed')
if __name__=='__main__':main()
