import numpy as np
from scipy.integrate import quad
from boundary_trapping.circle import (wrapped_cauchy,harmonic_extension,complex_gradient,
    harmonic_conjugate,fisher_velocity,moment_velocity,exact_replicator,loss_cdf,loss_mean,
    transverse_second_derivative,integrate_fisher,projected_cosine_endpoint)
from boundary_trapping.cauchy import cauchy_density,lorentzian_expected_fitness,lorentzian_endpoint
from boundary_trapping.spherical import d4_linear_gradient,d4_linear_objective,d4_h1,d4_h2,d4_fisher_velocity
from boundary_trapping.mutation import reset_fisher,reset_moment,fisher_equilibrium

def test_normalization_fourier():
    th=np.linspace(-np.pi,np.pi,32768,endpoint=False);z=.32+.37j
    p=wrapped_cauchy(th,z); h=2*np.pi/len(th)
    assert abs(sum(p)*h-1)<1e-12
    for k in range(1,7): assert abs(np.sum(p*np.exp(1j*k*th))*h-z**k)<1e-12

def test_harmonic_gradient():
    z=.22+.31j;c=[1+.2j,-.3+.4j,.12-.1j];eps=1e-6
    grad=complex_gradient(z,c)
    du=(harmonic_extension(z+eps,c)-harmonic_extension(z-eps,c))/(2*eps)
    dv=(harmonic_extension(z+1j*eps,c)-harmonic_extension(z-1j*eps,c))/(2*eps)
    assert abs(grad-(du+1j*dv))<1e-9

def test_fisher_conjugate_conserved():
    z=.23+.41j;c=[1+.2j,.3-.1j]
    grad=complex_gradient(z,c);dz=fisher_velocity(z,c)
    # d Im H = Im H'(z) dz; H'(z)=conj(grad)
    assert abs(np.imag(np.conj(grad)*dz))<1e-14

def test_moment_cosine():
    for z in [.1+.2j,-.5+.3j,.6-.2j]:
        assert abs(moment_velocity(z,[1.])-(1-abs(z)**2)/2)<1e-13

def test_constant_fitness_stationary():
    assert fisher_velocity(.2+.3j,[])==0
    assert moment_velocity(.2+.3j,[])==0

def test_cosine_endpoint():
    z=.1+.7j
    sol=integrate_fisher(z,[1.],t_end=100000.)
    endpoint=projected_cosine_endpoint(z)
    assert abs(sol.y[1,-1]-.7)<1e-10
    assert abs(endpoint.real-np.sqrt(.51))<1e-13
    assert abs(complex(sol.y[0,-1],sol.y[1,-1])-endpoint)<.007

def test_replicator_mean():
    th=np.linspace(-np.pi,np.pi,4096,endpoint=False);p=wrapped_cauchy(th,.2+.45j)
    out=exact_replicator(th,20,p,np.cos(th))
    assert np.isclose(out.sum()*2*np.pi/len(th),1)
    assert (out*np.cos(th)).sum()*2*np.pi/len(th)>.97

def test_transverse_acceleration():
    z=.1+.4j
    # exact first and second derivatives are obtained from moments at t=0
    th=np.linspace(-np.pi,np.pi,32768,endpoint=False);p=wrapped_cauchy(th,z);h=2*np.pi/len(th)
    Y=np.sin(th);X=np.cos(th); mx=(p*X).sum()*h
    dd=(p*Y*(X-mx)**2).sum()*h-(p*Y).sum()*h*((p*(X-mx)**2).sum()*h)
    assert abs(dd-transverse_second_derivative(z))<1e-12

def test_loss_distribution():
    assert abs(loss_mean()-(1-8/(3*np.pi)))<1e-14
    assert loss_cdf(-1)==0 and loss_cdf(2)==1
    assert abs(1-loss_cdf(.2)-.28476)<.0001

def test_cauchy_integral():
    mu,sig=.8,.7
    v=quad(lambda x:cauchy_density(x,mu,sig)/(1+x*x),-np.inf,np.inf,epsabs=1e-10)[0]
    assert abs(v-lorentzian_expected_fitness(mu,sig))<1e-10
    assert abs(lorentzian_endpoint(0,sig))<1e-12
    assert abs(lorentzian_endpoint(mu,sig))>1e-3

def test_d4_harmonics_and_derivative():
    assert abs(d4_h1(1)-1)<1e-13 and abs(d4_h2(1)-1)<1e-13
    a=np.array([.23,.32,.13,.08]);g=d4_linear_gradient(a)
    for k in range(4):
        delta=np.zeros(4);delta[k]=1e-6
        derivative=(d4_linear_objective(a+delta)-d4_linear_objective(a-delta))/2e-6
        assert abs(derivative-g[k])<1e-9
    assert np.dot(d4_fisher_velocity(a),g)>0

def test_reset_equilibrium():
    for rate in (.01,.1,.3,1.):
        z=fisher_equilibrium(rate)
        assert abs(reset_fisher(z,rate))<1e-12
        assert abs(reset_moment(z,rate))<1e-12

def test_watson_symmetric_closure():
    from boundary_trapping.watson import order_parameter,variance_cos2,fisher_watson_velocity
    assert abs(order_parameter(0))<1e-14
    assert abs(variance_cos2(0)-.5)<1e-14
    for k in (.1,.9,3.):
        eps=1e-6
        deriv=(order_parameter(k+eps)-order_parameter(k-eps))/(2*eps)
        assert abs(deriv-variance_cos2(k))<1e-9
    assert fisher_watson_velocity(0,0.3,'reset')==.5
