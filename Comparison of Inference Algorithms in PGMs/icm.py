import numpy as np   
import scipy.stats

K = 100 #num pixels
T = 50 #number of training examples
J = 20 #number of images in library

#hidden parameters
m = np.random.randint(2,size=K) # binary r.v.s
b = np.random.randint(J,size=T)
f = np.random.randint(J,size=T)

#theta
alpha = np.random.random(size=(T,K))
mu = np.random.random(size=(J,K))*255
psi = np.random.random(size=(J,K))*30
pi = np.random.random(size=J); pi = pi / sum()

#calling normal density
scipy.stats.norm(np.array([[100,100],[99,99]]), [[12,12],[10,10]]).pdf([98,99]) #returns 2x2 matrix

z = np.zeros((T,K))

def icm(m, b, f, alpha, mu, psi, pi, z ):
    for t in np.range(T):
        f[t] = np.argmax(pi * np.prod((alpha**m)*((1-alpha)**(1-m))*(scipy.stats.norm(mu, psi).pdf(z[t,:])**m), axis=1))
        
        mu_f = mu[f[t],:]
        psi_f = psi[f[t],:]
        alpha_f = alpha[f[t],:]
        
        m = np.argmax(np.array([
            (1-alpha_f)*scipy.stats.norm(mu[b[t],:],psi[b[t],:]).pdf(z[t,:]),
            alpha_f*scipy.stats.norm(mu_f, psi_f).pdf(z[t,:])
        ]), axis=1)

        b[t] = np.argmax(pi * np.prod(( scipy.stats.norm(mu, psi).pdf(z[t,:]) ** (1-m) ),axis=1) )

    for j in range(J):
        pi
