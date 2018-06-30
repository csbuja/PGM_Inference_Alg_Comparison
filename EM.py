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
scipy.stats.norm(np.array([[100,100,100],[99,99,99]]), [[12,12,12],[10,10,10]]).pdf([98,99,100]) #returns 2x3 matrix

z = np.zeros((T,K))

def EM(M, b, f, alpha, mu, psi, pi, z ):
    Q = np.zeros((T,J,J))
    for t in range(T):
        Q[t,:,:] = np.outer(pi,pi)
        normal = (scipy.stats.norm(mu, psi).pdf(z[t,:]))  #J x K

        


        

        np.outer