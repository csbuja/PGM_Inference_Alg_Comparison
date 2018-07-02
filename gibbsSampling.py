import numpy as np   
import scipy.stats

K = 100 #num pixels
T = 50 #number of training examples
J = 20 #number of images in library

#hidden parameters
M = np.random.randint(2,size=(T,K)) # binary r.v.s
b = np.random.randint(J,size=T)
f = np.random.randint(J,size=T)

#theta
alpha = np.random.random(size=(J,K))
mu = np.random.random(size=(J,K))*255
psi = np.random.random(size=(J,K))*30
pi = np.random.random(size=J); pi = pi / sum()

#calling normal density
scipy.stats.norm(np.array([[100,100],[99,99]]), [[12,12],[10,10]]).pdf([98,99]) #returns 2x2 matrix

z = np.zeros((T,K))


#In this gibbs sampler the E-step is updated stochastically using sampling, but we use ICM-updates for the model parameters.
def gibbs(M, b, f, alpha, mu, psi, pi, z ):

    for t in np.range(T):
        m = M[t,:]
        f[t] = np.random.choice(np.arange(J),p=(pi * np.prod((alpha**m)*((1-alpha)**(1-m))*(scipy.stats.norm(mu, psi).pdf(z[t,:])**m), axis=1))) 
        mu_f = mu[f[t],:]
        psi_f = psi[f[t],:]
        alpha_f = alpha[f[t],:]
    
        M_dist = np.array([
            (1-alpha_f)*scipy.stats.norm(mu[b[t],:],psi[b[t],:]).pdf(z[t,:]),
            alpha_f*scipy.stats.norm(mu_f, psi_f).pdf(z[t,:])
        ]) # K by 2

        for i in range(K):
            M[t,i] = np.random.choice([0,1],p=M_dist[i,:])
        b[t] = np.random.choice(np.arange(J),p= pi * np.prod(( scipy.stats.norm(mu, psi).pdf(z[t,:]) ** (1-M[t,:]) ),axis=1))

    for j in range(J):
        pi[j] = (1.0/(2*T) )*np.sum((f==j).astype(int) + (b==j).astype(int))

    for j in range(J):
        for i in range(K):
            alpha[j,i] = np.sum((f == j).astype(float) * M[:,i])/np.sum((f==j).astype(float))
            mu[j,i] =   np.sum((f == j).astype(float) * (b == j).astype(float) * z[:,i]) / np.sum((f == j).astype(float) * (b == j).astype(float) )
            psi[j,i] = np.sum((f == j).astype(float) * (b == j).astype(float) * (z[:,i] - mu[j,i])**2) / np.sum((f == j).astype(float) * (b == j).astype(float) )

    