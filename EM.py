import numpy as np   
import scipy.stats

K = 100 #num pixels
T = 50 #number of training examples
J = 20 #number of images in library

#theta
alpha = np.random.random(size=(J,K))
mu = np.random.random(size=(J,K))*255
psi = np.random.random(size=(J,K))*30
pi = np.random.random(size=J); pi = pi / sum()

#calling normal density
scipy.stats.norm(np.array([[100,100,100],[99,99,99]]), [[12,12,12],[10,10,10]]).pdf([98,99,100]) #returns 2x3 matrix

z = np.zeros((T,K))

def EM(M, b, f, alpha, mu, psi, pi, z ):
    #Set Q to be the exact posterior of that hidden variable given its markov blanket

    Q = np.zeros((T,J,J)) # T, B, F
    for t in range(T):
        for b in range(J):
            for f in range(J):
                Q[t,b,f] = pi[b]*pi[f]*np.prod( alpha[f,:] * scipy.stats.norm(mu[f,:],psi[f,:]).pdf(z[t,:]) + (1.0-alpha[f,:])*( scipy.stats.norm(mu[b,:],psi[b,:]).pdf(z[t,:]))        ,axis=0)
   
    #normalize Q over JJ for each example
    sQ = np.sum(np.sum(Q,axis=1),axis=1) #T values
    Q = np.transpose( np.transpose(Q)/ sQ) 

    Q_b = np.sum(Q,axis=2)
    Q_f = np.sum(Q,axis=1)

    Q_m_givenbf = np.zeros((2,T,J,J,K))   # if m is 0 , index is 0 different than in paper
    #renormalize m dis

    for t in range(T):
        for b in range(J):
            for f in range(J):
                a = (1.0-alpha[f,:])* scipy.stats.norm(mu[b,:],psi[b,:]).pdf(z[t,:])
                b = (alpha[f,:])* scipy.stats.norm(mu[f,:],psi[f,:]).pdf(z[t,:])
                Q_m_givenbf[0,t,b,f,:] = a/(a+b)
                Q_m_givenbf[1,t,b,f,:] = b/(a+b)

    #Q_bf (called Q) is computed above in T, J, J
    Q_mbf= np.zeros((2,T,J,J,K))
    #Q_m_givenbf = np.zeros((2,T,J,J,K)) #
    for t in range(T):
        for i in range(K):
            Q_mbf[ 0, t, :, :, i] = Q_m_givenbf[ 0, t, :, :, i] * Q[t,:,:]
            Q_mbf[ 1, t, :, :, i] = Q_m_givenbf[ 1, t, :, :, i] * Q[t,:,:]

    Q_mb = np.sum(Q_mbf,axis=3) #2, T, J , K
    Q_mf = np.sum(Q_mbf,axis=2) #2, T, J, K



    #Update parameters - AKA M step
    for j in range(J):
        pi[j] = 1.0/(2*T) * np.sum(Q_f[:,j] + Q_b[:,j])
        alpha[j,:] = np.sum(Q_mf[1,:,j,:],axis=0) /np.sum(Q_f[:,j])


    #update mu and psi parameters of pixel intensity model
    for j in range(J):
        for i in range(K):
            denom =   np.sum(Q_mf[1,:,j,i] + Q_mb[0,:,j,i]) 
            mu_numer = np.sum((Q_mf[1,:,j,i] + Q_mb[0,:,j,i])*z[:,i]) #expectation over the pixel intensity
            mu[j,i] = mu_numer / denom

            psi_numer = np.sum( (Q_mf[1,:,j,i] + Q_mb[0,:,j,i]) * (z[:,i] - mu[j,i]) ** 2 )
            psi[j,i] = psi_numer/denom
            
