# This script fits a matrix variate regression model given input data X and output matrices Y
# In this script, A refers to the row-coefficients (known in Boyle et al as beta_1)
# B refers to the column-coefficients (known in Boyle et al as beta_2)

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import wishart
from scipy.linalg import inv



# get data dimensions
n = 60 # number of matrix observations
p = 4 # number of rows of each matrix
r = 4 # number of columns of each matrix
q1 = 8 # number of rows of covariate matrix (X matrix)
q2 = 4 # number of columns of covariate matrix (X matrix)

# read in data
Y_dat = pd.read_csv('example.csv') # observation matrix, given as an n*p by r dataframe.
Covariates = pd.read_csv('covariates.csv', index_col=(0)) # covariate matrix, given as an n*q1 by q2 dataframe

# Convert to Tensors
Y = tf.convert_to_tensor(Y_dat)
Y = tf.reshape(Y, [n, p, r]) # This should reflext your dataset's n, p, and r. 


X = tf.convert_to_tensor(Covariates)
X = tf.reshape(X, [n, q1, q2]) #This should reflect your covariates


# Standardize
X = tf.math.truediv(X - tf.reduce_mean(X,0),tf.math.reduce_std(X,0))


# Get common tensors used in model fitting
Y_prime = tf.transpose(Y,perm = [0,2,1])
X_prime = tf.transpose(X, perm = [0,2,1])
Y_diffs_prime = tf.transpose(Y-tf.reduce_mean(Y,0),perm = [0,2,1])





################################### Standard Model Fitting

np.random.seed(5)
loglikes = []
As = []
Bs = []
sigmas = [] 
psis = []
for k in range(10):
    
    # initialize convergence hyperparameters
    epsilon = .000001
    max_iter = 1000
    criteria_met = False
    i = 0
    
    # set hyperparamters used in random paramter initialization 
    hyper1= 4 #lower bound for wishart degrees of freedom
    hyper2 = 10 #upper bound for wishart degrees of freedom
    hyper3 = 10 #scale for mean of B initialization
    hyper4 = .5 #shift for mean of B initialization
    hyper5 = 3 #variance for B initialization
    
    # Randomly initialize parameters
    df = np.random.randint(hyper1,hyper2) #get random wishart degrees of freedom
    m = hyper3*(np.random.random()-hyper4) #get random mean of B
    sigma = tf.constant((wishart.rvs(df,np.identity(p)))) #initialize sigma
    psi = tf.constant((wishart.rvs(df,np.identity(r)))) #initialize psi
    B = tf.constant(np.random.normal(m, hyper5, size=(q2, r))) #initialize B

    # Fit inital A based on randomization
    A = tf.reduce_sum((Y - tf.reduce_mean(Y,0)) @ inv(psi.numpy()) @ B.numpy() @ X_prime, 0) @ inv(tf.reduce_sum(
            X @ B.numpy().transpose() @ inv(psi.numpy()) @ B.numpy() @ X_prime, 0).numpy())
    
    # Get initial loglikelihood    
    R = Y - tf.reduce_mean(Y,0) - A.numpy() @ X @ B.numpy().transpose()
    R_prime = tf.transpose(R,perm = [0,2,1])
    
    first_terms = -p*r*np.log(2*np.pi)/2 - n*p*np.log(np.linalg.det(psi.numpy()))/2 - n*r*np.log(np.linalg.det(sigma.numpy()))/2
    second_term = -tf.reduce_sum( tf.linalg.trace( inv(sigma.numpy()) @ R @ inv(psi.numpy()) @ R_prime ),0)/2

    loglike= first_terms+second_term
    
    while (not criteria_met and i < max_iter):
        A_last = A
        B_last = B
        psi_last = psi
        sigma_last = sigma
        ll_last = loglike
    
            
        A = tf.reduce_sum((Y - tf.reduce_mean(Y,0)) @ inv(psi.numpy()) @ B.numpy() @ X_prime, 0) @ inv(tf.reduce_sum(
                X @ B.numpy().transpose() @ inv(psi.numpy()) @ B.numpy() @ X_prime, 0).numpy())
        
        R = Y - tf.reduce_mean(Y,0) - A.numpy() @ X @ B.numpy().transpose()
        R_prime = tf.transpose(R,perm = [0,2,1])
        
        sigma = tf.reduce_sum(R @ inv(psi.numpy()) @ R_prime, 0)/(n*r)
        
        
        B = tf.reduce_sum(Y_diffs_prime @ inv(sigma.numpy()) @ A.numpy() @ X,0) @ inv(tf.reduce_sum(
            X_prime @ A.numpy().transpose() @ inv(sigma.numpy()) @ A.numpy() @ X, 0).numpy())
        
        R = Y - tf.reduce_mean(Y,0) - A.numpy() @ X @ B.numpy().transpose()
        R_prime = tf.transpose(R,perm = [0,2,1])
        
        psi = tf.reduce_sum(R_prime @ inv(sigma.numpy()) @ R, 0)/(n*p)
        
        #normalize so that frobenius norm is 1
        A = (1/np.linalg.norm(A.numpy(), 'fro'))*A
        B = (np.linalg.norm(A.numpy(), 'fro'))*B
        
        norm2 = np.linalg.norm(sigma.numpy(), 'fro')
        sigma = (1/norm2)*sigma
        psi = (norm2)*psi
    
        # Caclulate LL 
        R = Y - tf.reduce_mean(Y,0) - A.numpy() @ X @ B.numpy().transpose()
        R_prime = tf.transpose(R,perm = [0,2,1])

        first_terms = -p*r*np.log(2*np.pi)/2 - n*p*np.log(np.linalg.det(psi.numpy()))/2 - n*r*np.log(np.linalg.det(sigma.numpy()))/2
        second_term = -tf.reduce_sum( tf.linalg.trace( inv(sigma.numpy()) @ R @ inv(psi.numpy()) @ R_prime ),0)/2

        loglike= first_terms+second_term

        # Stoping rule: based on euclidean distance between estimates
        i += 1
        criteria_met = ((abs((float(loglike) -float(ll_last))) < epsilon) and
                        (abs(tf.norm(A - A_last).numpy()) < epsilon) and 
                        (abs(tf.norm(B - B_last).numpy()) < epsilon) and 
                        (abs(tf.norm(psi - psi_last).numpy()) < epsilon) and 
                        (abs(tf.norm(sigma - sigma_last).numpy()) < epsilon))

    As += [A]
    Bs += [B]
    sigmas += [sigma]
    psis += [psi]
    loglikes += [loglike]
    
    print("iteration:", k)
    print("converged in: "+str(i))
    print("loglike: ", loglike)
            







####################################If psi is the identity matrix

np.random.seed(5)
loglikes = []
As = []
Bs = []
sigmas = [] 
psis = []
for k in range(10):
    
    # initialize convergence hyperparameters
    epsilon = .000001
    max_iter = 1000
    criteria_met = False
    i = 0
    
    # set hyperparamters used in random paramter initialization 
    hyper1= 4 #lower bound for wishart degrees of freedom
    hyper2 = 10 #upper bound for wishart degrees of freedom
    hyper3 = 10 #scale for mean of B initialization
    hyper4 = .5 #shift for mean of B initialization
    hyper5 = 3 #variance for B initialization
    
    # Randomly initialize parameters
    df = np.random.randint(hyper1,hyper2) #get random wishart degrees of freedom
    m = hyper3*(np.random.random()-hyper4) #get random mean of B
    sigma = tf.constant((wishart.rvs(df,np.identity(p)))) #initialize sigma
    psi = tf.eye(r,r,batch_shape=None,dtype=tf.dtypes.float32, name=None) #initialize psi
    B = tf.constant(np.random.normal(m, hyper5, size=(q2, r))) #initialize B

    # Fit inital A based on randomization
    A = tf.reduce_sum((Y - tf.reduce_mean(Y,0)) @ inv(psi.numpy()) @ B.numpy() @ X_prime, 0) @ inv(tf.reduce_sum(
            X @ B.numpy().transpose() @ inv(psi.numpy()) @ B.numpy() @ X_prime, 0).numpy())
    
    # Get initial loglikelihood    
    R = Y - tf.reduce_mean(Y,0) - A.numpy() @ X @ B.numpy().transpose()
    R_prime = tf.transpose(R,perm = [0,2,1])
    
    first_terms = -p*r*np.log(2*np.pi)/2 - n*p*np.log(np.linalg.det(psi.numpy()))/2 - n*r*np.log(np.linalg.det(sigma.numpy()))/2
    second_term = -tf.reduce_sum( tf.linalg.trace( inv(sigma.numpy()) @ R @ inv(psi.numpy()) @ R_prime ),0)/2

    loglike= first_terms+second_term
    
    while (not criteria_met and i < max_iter):
        A_last = A
        B_last = B
        psi_last = psi
        sigma_last = sigma
        ll_last = loglike
    
            
        A = tf.reduce_sum((Y - tf.reduce_mean(Y,0)) @ inv(psi.numpy()) @ B.numpy() @ X_prime, 0) @ inv(tf.reduce_sum(
                X @ B.numpy().transpose() @ inv(psi.numpy()) @ B.numpy() @ X_prime, 0).numpy())
        
        R = Y - tf.reduce_mean(Y,0) - A.numpy() @ X @ B.numpy().transpose()
        R_prime = tf.transpose(R,perm = [0,2,1])
        
        sigma = tf.reduce_sum(R @ inv(psi.numpy()) @ R_prime, 0)/(n*r)
        
        
        B = tf.reduce_sum(Y_diffs_prime @ inv(sigma.numpy()) @ A.numpy() @ X,0) @ inv(tf.reduce_sum(
            X_prime @ A.numpy().transpose() @ inv(sigma.numpy()) @ A.numpy() @ X, 0).numpy())
        
        
        #normalize so that frobenius norm is 1
        A = (1/np.linalg.norm(A.numpy(), 'fro'))*A
        B = (np.linalg.norm(A.numpy(), 'fro'))*B
        
       
        # Caclulate LL 
        R = Y - tf.reduce_mean(Y,0) - A.numpy() @ X @ B.numpy().transpose()
        R_prime = tf.transpose(R,perm = [0,2,1])

        first_terms = -p*r*np.log(2*np.pi)/2 - n*p*np.log(np.linalg.det(psi.numpy()))/2 - n*r*np.log(np.linalg.det(sigma.numpy()))/2
        second_term = -tf.reduce_sum( tf.linalg.trace( inv(sigma.numpy()) @ R @ inv(psi.numpy()) @ R_prime ),0)/2

        loglike= first_terms+second_term

        # Stoping rule: based on euclidean distance between estimates
        i += 1
        criteria_met = ((abs((float(loglike) -float(ll_last))) < epsilon) and
                        (abs(tf.norm(A - A_last).numpy()) < epsilon) and 
                        (abs(tf.norm(B - B_last).numpy()) < epsilon) and 
                        (abs(tf.norm(psi - psi_last).numpy()) < epsilon) and 
                        (abs(tf.norm(sigma - sigma_last).numpy()) < epsilon))

    As += [A]
    Bs += [B]
    sigmas += [sigma]
    psis += [psi]
    loglikes += [loglike]
    
    print("iteration:", k)
    print("converged in: "+str(i))
    print("loglike: ", loglike)

            



################################### If sigma is the identity matrix

np.random.seed(5)
loglikes = []
As = []
Bs = []
sigmas = [] 
psis = []
for k in range(10):
    
    # initialize convergence hyperparameters
    epsilon = .000001
    max_iter = 1000
    criteria_met = False
    i = 0
    
    # set hyperparamters used in random paramter initialization 
    hyper1= 4 #lower bound for wishart degrees of freedom
    hyper2 = 10 #upper bound for wishart degrees of freedom
    hyper3 = 10 #scale for mean of B initialization
    hyper4 = .5 #shift for mean of B initialization
    hyper5 = 3 #variance for B initialization
    
    # Randomly initialize parameters
    df = np.random.randint(hyper1,hyper2) #get random wishart degrees of freedom
    m = hyper3*(np.random.random()-hyper4) #get random mean of B
    sigma = tf.eye(p,p,batch_shape=None,dtype=tf.dtypes.float32, name=None) #initialize sigma
    psi = tf.constant((wishart.rvs(df,np.identity(r)))) #initialize psi
    B = tf.constant(np.random.normal(m, hyper5, size=(q2, r))) #initialize B

    # Fit inital A based on randomization
    A = tf.reduce_sum((Y - tf.reduce_mean(Y,0)) @ inv(psi.numpy()) @ B.numpy() @ X_prime, 0) @ inv(tf.reduce_sum(
            X @ B.numpy().transpose() @ inv(psi.numpy()) @ B.numpy() @ X_prime, 0).numpy())
    
    # Get initial loglikelihood    
    R = Y - tf.reduce_mean(Y,0) - A.numpy() @ X @ B.numpy().transpose()
    R_prime = tf.transpose(R,perm = [0,2,1])
    
    first_terms = -p*r*np.log(2*np.pi)/2 - n*p*np.log(np.linalg.det(psi.numpy()))/2 - n*r*np.log(np.linalg.det(sigma.numpy()))/2
    second_term = -tf.reduce_sum( tf.linalg.trace( inv(sigma.numpy()) @ R @ inv(psi.numpy()) @ R_prime ),0)/2

    loglike= first_terms+second_term
    
    while (not criteria_met and i < max_iter):
        A_last = A
        B_last = B
        psi_last = psi
        sigma_last = sigma
        ll_last = loglike
    
            
        A = tf.reduce_sum((Y - tf.reduce_mean(Y,0)) @ inv(psi.numpy()) @ B.numpy() @ X_prime, 0) @ inv(tf.reduce_sum(
                X @ B.numpy().transpose() @ inv(psi.numpy()) @ B.numpy() @ X_prime, 0).numpy())
        
        
        B = tf.reduce_sum(Y_diffs_prime @ inv(sigma.numpy()) @ A.numpy() @ X,0) @ inv(tf.reduce_sum(
            X_prime @ A.numpy().transpose() @ inv(sigma.numpy()) @ A.numpy() @ X, 0).numpy())
        
        R = Y - tf.reduce_mean(Y,0) - A.numpy() @ X @ B.numpy().transpose()
        R_prime = tf.transpose(R,perm = [0,2,1])
        
        psi = tf.reduce_sum(R_prime @ inv(sigma.numpy()) @ R, 0)/(n*p)
        
        #normalize so that frobenius norm is 1
        A = (1/np.linalg.norm(A.numpy(), 'fro'))*A
        B = (np.linalg.norm(A.numpy(), 'fro'))*B
        
    
        # Caclulate LL 
        R = Y - tf.reduce_mean(Y,0) - A.numpy() @ X @ B.numpy().transpose()
        R_prime = tf.transpose(R,perm = [0,2,1])

        first_terms = -p*r*np.log(2*np.pi)/2 - n*p*np.log(np.linalg.det(psi.numpy()))/2 - n*r*np.log(np.linalg.det(sigma.numpy()))/2
        second_term = -tf.reduce_sum( tf.linalg.trace( inv(sigma.numpy()) @ R @ inv(psi.numpy()) @ R_prime ),0)/2

        loglike= first_terms+second_term

        # Stoping rule: based on euclidean distance between estimates
        i += 1
        criteria_met = ((abs((float(loglike) -float(ll_last))) < epsilon) and
                        (abs(tf.norm(A - A_last).numpy()) < epsilon) and 
                        (abs(tf.norm(B - B_last).numpy()) < epsilon) and 
                        (abs(tf.norm(psi - psi_last).numpy()) < epsilon) and 
                        (abs(tf.norm(sigma - sigma_last).numpy()) < epsilon))

    As += [A]
    Bs += [B]
    sigmas += [sigma]
    psis += [psi]
    loglikes += [loglike]
    
    print("iteration:", k)
    print("converged in: "+str(i))
    print("loglike: ", loglike)   
            



################################### If sigma and psi are both identity matrices
# Instead, a single constant variance parameter is fit


np.random.seed(5)
loglikes = []
As = []
Bs = []
sigmas = [] 
psis = []
for k in range(10):
    
    # initialize convergence hyperparameters
    epsilon = .000001
    max_iter = 1000
    criteria_met = False
    i = 0
    
    # set hyperparamters used in random paramter initialization 
    hyper1= 4 #lower bound for wishart degrees of freedom
    hyper2 = 10 #upper bound for wishart degrees of freedom
    hyper3 = 10 #scale for mean of B initialization
    hyper4 = .5 #shift for mean of B initialization
    hyper5 = 3 #variance for B initialization
    var_constant = np.random.normal(1) # variance parameter
    
    
    # Randomly initialize parameters
    df = np.random.randint(hyper1,hyper2) #get random wishart degrees of freedom
    m = hyper3*(np.random.random()-hyper4) #get random mean of B
    sigma = tf.eye(p,p,batch_shape=None,dtype=tf.dtypes.float32, name=None) #initialize sigma
    sigma = var_constant*sigma
    psi = tf.eye(r,r,batch_shape=None,dtype=tf.dtypes.float32, name=None) #initialize psi
    B = tf.constant(np.random.normal(m, hyper5, size=(q2, r))) #initialize B


    # Fit inital A based on randomization
    A = tf.reduce_sum((Y - tf.reduce_mean(Y,0)) @ inv(psi.numpy()) @ B.numpy() @ X_prime, 0) @ inv(tf.reduce_sum(
            X @ B.numpy().transpose() @ inv(psi.numpy()) @ B.numpy() @ X_prime, 0).numpy())
    
    # Get initial loglikelihood    
    R = Y - tf.reduce_mean(Y,0) - A.numpy() @ X @ B.numpy().transpose()
    R_prime = tf.transpose(R,perm = [0,2,1])
    
    first_terms = -p*r*np.log(2*np.pi)/2 - n*p*np.log(np.linalg.det(psi.numpy()))/2 - n*r*np.log(np.linalg.det(sigma.numpy()))/2
    second_term = -tf.reduce_sum( tf.linalg.trace( inv(sigma.numpy()) @ R @ inv(psi.numpy()) @ R_prime ),0)/2

    loglike= first_terms+second_term
    
    while (not criteria_met and i < max_iter):
        A_last = A
        B_last = B
        psi_last = psi
        sigma_last = sigma
        ll_last = loglike
    
            
        A = tf.reduce_sum((Y - tf.reduce_mean(Y,0)) @ inv(psi.numpy()) @ B.numpy() @ X_prime, 0) @ inv(tf.reduce_sum(
                X @ B.numpy().transpose() @ inv(psi.numpy()) @ B.numpy() @ X_prime, 0).numpy())
        

        B = tf.reduce_sum(Y_diffs_prime @ inv(sigma.numpy()) @ A.numpy() @ X,0) @ inv(tf.reduce_sum(
            X_prime @ A.numpy().transpose() @ inv(sigma.numpy()) @ A.numpy() @ X, 0).numpy())
        
        R = Y - tf.reduce_mean(Y,0) - A.numpy() @ X @ B.numpy().transpose()
        R_prime = tf.transpose(R,perm = [0,2,1])
        
        var_constant = np.sum(tf.linalg.trace(R_prime @ inv(psi.numpy()) @ R))/4*60
        sigma =var_constant*tf.eye(4,4,batch_shape=None,dtype=tf.dtypes.float32, name=None)
    
        
        #normalize so that frobenius norm is 1
        A = (1/np.linalg.norm(A.numpy(), 'fro'))*A
        B = (np.linalg.norm(A.numpy(), 'fro'))*B
        
    
        # Caclulate LL 
        R = Y - tf.reduce_mean(Y,0) - A.numpy() @ X @ B.numpy().transpose()
        R_prime = tf.transpose(R,perm = [0,2,1])

        first_terms = -p*r*np.log(2*np.pi)/2 - n*p*np.log(np.linalg.det(psi.numpy()))/2 - n*r*np.log(np.linalg.det(sigma.numpy()))/2
        second_term = -tf.reduce_sum( tf.linalg.trace( inv(sigma.numpy()) @ R @ inv(psi.numpy()) @ R_prime ),0)/2

        loglike= first_terms+second_term

        # Stoping rule: based on euclidean distance between estimates
        i += 1
        criteria_met = ((abs((float(loglike) -float(ll_last))) < epsilon) and
                        (abs(tf.norm(A - A_last).numpy()) < epsilon) and 
                        (abs(tf.norm(B - B_last).numpy()) < epsilon) and 
                        (abs(tf.norm(psi - psi_last).numpy()) < epsilon) and 
                        (abs(tf.norm(sigma - sigma_last).numpy()) < epsilon))

    As += [A]
    Bs += [B]
    sigmas += [sigma]
    psis += [psi]
    loglikes += [loglike]
    
    print("iteration:", k)
    print("converged in: "+str(i))
    print("loglike: ", loglike)
    

