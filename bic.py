import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("data_to_fit.txt")

x_obs = data[:,0]
y_obs = data[:,1]
sigma_y = data[:,2]

def model_A(x,params):
    y = 0.0
    y = params[0] + x*params[1] + params[2]*x**2
    return y

def model_B(x,params):
    y = 0.0
    y = params[0]*(np.exp(-0.5*(x-params[1])**2/params[2]**2))   
    return y

def model_C(x,params):
    y = params[0]*(np.exp(-0.5*(x-params[1])**2/params[2]**2))
     
    y += params[0]*(np.exp(-0.5*(x-params[3])**2/params[4]**2)) 
    
    return y


def log_verosimil_A(y_obs,x_obs,params):
    n_obs = len(y_obs)
    l=0.0
    for i in range(n_obs):
        l += -0.5*(y_obs[i]-model_A(x_obs[i],params))**2/sigma_y[i]**2
    return l

def log_verosimil_B(y_obs,x_obs,params):
    n_obs = len(y_obs)
    l=0.0
    for i in range(n_obs):
        l += -0.5*(y_obs[i]-model_B(x_obs[i],params))**2/sigma_y[i]**2
    return l


def log_verosimil_C(y_obs,x_obs,params):
    n_obs = len(y_obs)
    l=0.0
    for i in range(n_obs):
        l += -0.5*(y_obs[i]-model_C(x_obs[i],params))**2/sigma_y[i]**2
    return l



def run_mcmc(x_obs, y_obs, sigma_y):
    
    n_iterations = 100000
    params_A = np.zeros([n_iterations,3])
    params_B = np.zeros([n_iterations,3])
    params_C = np.zeros([n_iterations,5])
    
    for i in range(1,n_iterations):
        
        actual_params_A = params_A[i-1,:]
        propuesta_params_A = params_A[i-1,:] + np.random.normal(scale=0.01,size=3)
        
        actual_params_B = params_B[i-1,:]
        propuesta_params_B = params_B[i-1,:] + np.random.normal(scale=0.01,size=3)
        
        actual_params_C = params_C[i-1,:]
        propuesta_params_C = params_C[i-1,:] + np.random.normal(scale=0.01,size=5)
        
        log_vero_actual_A = log_verosimil_A(y_obs,x_obs,actual_params_A)
        log_vero_propuesta_A = log_verosimil_A(y_obs,x_obs,propuesta_params_A)
        
        
        log_vero_actual_B = log_verosimil_B(y_obs,x_obs,actual_params_B)
        log_vero_propuesta_B = log_verosimil_B(y_obs,x_obs,propuesta_params_B)
        
        log_vero_actual_C = log_verosimil_C(y_obs,x_obs,actual_params_C)
        log_vero_propuesta_C = log_verosimil_C(y_obs,x_obs,propuesta_params_C)
        
        r_A = np.min([np.exp(log_vero_propuesta_A - log_vero_actual_A), 1.0])
        r_B = np.min([np.exp(log_vero_propuesta_B - log_vero_actual_B), 1.0])
        r_C = np.min([np.exp(log_vero_propuesta_C - log_vero_actual_C), 1.0])
        
        alpha_A = np.random.random()
        alpha_B = np.random.random()
        alpha_C = np.random.random()
        
        if alpha_A < r_A:
            params_A[i,:] = propuesta_params_A
                              
        else:
            params_A[i,:] = actual_params_A
            
        if alpha_B < r_B:
            params_B[i,:] = propuesta_params_B
        else:
            params_B[i,:] = actual_params_B    
            
            
        if alpha_C < r_C:
            params_C[i,:] = propuesta_params_C
        else:
            params_C[i,:] = actual_params_C
            
            
    params_A = params_A[n_iterations//3:,:]
    params_B = params_B[n_iterations//3:,:]
    params_C = params_C[n_iterations//3:,:]
    
    return {'params_A':params_A, 'params_B':params_B, 'params_C':params_C}




results = run_mcmc(x_obs,y_obs,sigma_y) 

params_A = results['params_A']
params_B = results['params_B']
params_C = results['params_C']

params_A_mean = np.zeros(3)
params_B_mean = np.zeros(3)
params_C_mean = np.zeros(5)


for i in range(3):
    params_A_mean[i] = np.mean(params_A[:,i])
    params_B_mean[i] = np.mean(params_B[:,i])
    
for i in range(5):
    params_C_mean[i] = np.mean(params_C[:,i])
    
y_A = np.zeros(len(y_obs))
y_B = np.zeros(len(y_obs))
y_C = np.zeros(len(y_obs))
   
for i in range(len(y_obs)):
    y_A[i] = params_A_mean[0] + x_obs[i]*params_A_mean[1] + params_A_mean[2]*x_obs[i]**2
    y_B[i] = params_B_mean[0]*(np.exp(-0.5*(x_obs[i]-params_B_mean[1])**2/params_B_mean[2]**2)) 
    y_C[i] = y = params_C_mean[0]*(np.exp(-0.5*(x_obs[i]-params_C_mean[1])**2/params_C_mean[2]**2)) + params_C_mean[0]*(np.exp(-0.5*(x_obs[i]-params_C_mean[3])**2/params_C_mean[4]**2))
     


x_obs_B = x_obs     

x_obs_A = x_obs

x_obs_C = x_obs


x_obs_C, y_B = zip(*sorted(zip(x_obs_C, y_C)))
x_obs_B, y_B = zip(*sorted(zip(x_obs_B, y_B)))
x_obs_A, y_A = zip(*sorted(zip(x_obs_A, y_A)))



BIC_A = log_verosimil_A(y_obs,x_obs,params_A_mean)

BIC_B = log_verosimil_B(y_obs,x_obs,params_B_mean)

BIC_C = log_verosimil_C(y_obs,x_obs,params_C_mean)

print(BIC_A)


plt.figure(figsize=(10,7))
for i in range(0,3):
    plt.subplot(2,2,i+1)
    plt.hist(params_A[:,i],bins=15, density=True)
    plt.title(r"$p_{}= {:.2f}\pm {:.2f}$".format(i,np.mean(params_A[:,i]), np.std(params_A[:,i])))
    plt.xlabel(r"$p_{}$".format(i))
plt.subplot(2,2,4)
plt.plot(x_obs_A,y_A)
plt.errorbar(x_obs, y_obs, yerr=sigma_y, fmt='o')
plt.title(r"$BIC= {:.2f}$".format(BIC_A))
plt.tight_layout()
plt.savefig("modelo_A.png",  bbox_inches='tight')


plt.figure(figsize=(10,7))
for i in range(0,3):
    plt.subplot(2,2,i+1)
    plt.hist(params_B[:,i],bins=15, density=True)
    plt.title(r"$p_{}= {:.2f}\pm {:.2f}$".format(i,np.mean(params_B[:,i]), np.std(params_B[:,i])))
    plt.xlabel(r"$p_{}$".format(i))
plt.subplot(2,2,4)
plt.plot(x_obs_B,y_B)
plt.errorbar(x_obs, y_obs, yerr=sigma_y, fmt='o')
plt.title(r"$BIC= {:.2f}$".format(BIC_B))
plt.tight_layout()
plt.savefig("modelo_B.png",  bbox_inches='tight')



plt.figure(figsize=(10,7))
for i in range(0,5):
    plt.subplot(2,3,i+1)
    plt.hist(params_C[:,i],bins=15, density=True)
    plt.title(r"$p_{}= {:.2f}\pm {:.2f}$".format(i,np.mean(params_C[:,i]), np.std(params_C[:,i])))
    plt.xlabel(r"$p_{}$".format(i))
plt.subplot(2,3,6)
plt.plot(x_obs_C,y_C)
plt.errorbar(x_obs, y_obs, yerr=sigma_y, fmt='o')
plt.title(r"$BIC= {:.2f}$".format(BIC_C))
plt.tight_layout()
plt.savefig("modelo_C.png",  bbox_inches='tight')











