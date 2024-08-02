import numpy as np
import matplotlib.pyplot as plt

#constants
I_0=1
h =1
def gen_matrix(sigma, r, r_0, R, n):
    N = len(sigma)
    size = 2 * len(sigma) + 1
    matrix = np.zeros((size, size))
    matrix[0, 0:3] = [1, -1, -r[0]**(-2 * n)]
    matrix[1, 0:3] = [sigma[0], -sigma[1], sigma[1]*r[0]**(-2 * n)]
    delta = False

    for i in range(1, N):
        idx = 2 * i 
        if delta:
            matrix[idx, idx-1:idx+3] = [1, r[i-1]**(-2 * n), -1, -r[i-1]**(-2 * n)]
            matrix[idx + 1, idx-1:idx+3] = [sigma[i-1], -sigma[i-1] * r[i-1]**(-2 * n), -sigma[i], sigma[i] * r[i-1]**(-2 * n)]
        elif r[i] > r_0 and not delta:
            delta = True
            matrix[idx, idx-1:idx+3] = [1, r_0**(-2 * n), -1, -r_0**(-2 * n)]
            matrix[idx + 1, idx-1:idx+3] = [-1, r_0**(-2 * n), 1, -r_0**(-2 * n)]
        else:
            matrix[idx, idx-1:idx+3] = [1, r[i]**(-2 * n), -1, -r[i]**(-2 * n)]
            matrix[idx + 1, idx-1:idx+3] = [sigma[i], -sigma[i] * r[i]**(-2 * n), -sigma[i+1], sigma[i+1] * r[i]**(-2 * n)]

    matrix[-1, -1] = R**(-2*n)
    matrix[-1, -2] = -1
    return matrix

def vector(r, r_0, n):
    vec = np.zeros(2*len(r) + 3)
    r = np.array(r)
    
    idx = np.where(r >= r_0)[0]
    if idx.size > 0:
        first_idx = 2*idx[0] + 1 
        vec[first_idx] = -I_0 / (h*sigma[idx[0]]*np.pi *r_0**n * n)
    
    return vec


def solution(r, theta, r_position, sigma, r_0, theta_0, R, N):
    coefficients = np.array([np.dot(np.linalg.inv(gen_matrix(sigma, r_position, r_0, R, i)), vector(r_position, r_0, i)) for i in range(1, N)])
    C = False
    n = np.arange(1, N)
    n_expanded = n[:, None, None]
    final = 0
    idx = np.where(r_position >= r_0)[0][0]
    
    for i in range(len(sigma)):
        if i == 0:
            mask = r < r_position[i]

        elif i < len(r_position):
            mask = (r >= r_position[i-1]) & (r < r_position[i])
        else:
            mask = r >= r_position[i-1]
        theta_masked = theta
        matrix_theta = theta
        r_masked = r[mask]
        matrix_r_masked = r_masked
        
        result_r_n = np.power(matrix_r_masked, n_expanded)
        result_r_neg_n = np.power(matrix_r_masked, -n_expanded)
        angles = np.cos(n_expanded * (matrix_theta[:] - theta_0)) 
        angles_transposed = angles.T
        angles_reshaped = angles_transposed.reshape(angles_transposed.shape[0], angles_transposed.shape[1] * angles_transposed.shape[2])
        result_reshaped = result_r_n.reshape(result_r_n.shape[0] * result_r_n.shape[1], result_r_n.shape[2])
        result_reshaped_neg = result_r_neg_n.reshape(result_r_neg_n.shape[0] * result_r_neg_n.shape[1], result_r_neg_n.shape[2])
        
        #print(r_masked,angles_reshaped @ (result_reshaped * coefficients[:,0].reshape(-1,1)))
        #print(i, -I_0 * np.log(r_0) / (2 * np.pi * sigma[idx] * h) + angles_reshaped @ (result_reshaped * coefficients[:,2*i].reshape(-1,1)))
        
        if i < idx:
            if i == 0:
                final = -I_0 * np.log(r_0) / (2 * np.pi * sigma[idx] * h) + angles_reshaped @ (result_reshaped * coefficients[:,2*i].reshape(-1,1))

            else:
                final = np.concatenate([final,(-I_0 * np.log(r_0) / (2 * np.pi * sigma[idx] * h) + angles_reshaped @ (result_reshaped * coefficients[:,2*i-1].reshape(-1,1)) + angles_reshaped @ (result_reshaped_neg * coefficients[:,2*i].reshape(-1,1)))],axis = 1)

        elif i == idx:
            mask =  (r >= r_position[i-1]) & (r < r_0)
            theta_masked = theta
            matrix_theta = theta
            r_masked = r[mask]
            matrix_r_masked = r_masked
            result_r_n = np.power(matrix_r_masked, n_expanded)
            result_r_neg_n = np.power(matrix_r_masked, -n_expanded)
            angles = np.cos(n_expanded * (matrix_theta[:] - theta_0))
            angles_transposed = angles.T
            angles_reshaped = angles_transposed.reshape(angles_transposed.shape[0], angles_transposed.shape[1] * angles_transposed.shape[2])
            result_reshaped = result_r_n.reshape(result_r_n.shape[0] * result_r_n.shape[1], result_r_n.shape[2])
            result_reshaped_neg = result_r_neg_n.reshape(result_r_neg_n.shape[0] * result_r_neg_n.shape[1], result_r_neg_n.shape[2])

            final = np.concatenate([final,(-I_0 * np.log(r_0) / (2 * np.pi * sigma[idx] * h) + angles_reshaped @ (result_reshaped * coefficients[:,2*i-1].reshape(-1,1)) + angles_reshaped @ (result_reshaped_neg * coefficients[:,2*i].reshape(-1,1)))],axis = 1)
            
            mask =  (r > r_0) & (r < r_position[i])
            theta_masked = theta
            matrix_theta = theta
            r_masked = r[mask]
            matrix_r_masked = r_masked
            result_r_n = np.power(matrix_r_masked, n_expanded)
            result_r_neg_n = np.power(matrix_r_masked, -n_expanded)
            angles = np.cos(n_expanded * (matrix_theta[:] - theta_0))
            angles_transposed = angles.T
            angles_reshaped = angles_transposed.reshape(angles_transposed.shape[0], angles_transposed.shape[1] * angles_transposed.shape[2])
            result_reshaped = result_r_n.reshape(result_r_n.shape[0] * result_r_n.shape[1], result_r_n.shape[2])
            result_reshaped_neg = result_r_neg_n.reshape(result_r_neg_n.shape[0] * result_r_neg_n.shape[1], result_r_neg_n.shape[2])

            final = np.concatenate([final,(-I_0 * np.log(r_masked) / (2 * np.pi * sigma[idx] * h) + angles_reshaped @ (result_reshaped * coefficients[:,2*i+1].reshape(-1,1)) + angles_reshaped @ (result_reshaped_neg * coefficients[:,2*i+2].reshape(-1,1)))],axis = 1)

            


        else:
            if C == False:
                C = -I_0 * np.log(r_position[i-1])/ (2 * np.pi * sigma[i-1] * h) + I_0 * np.log(r_position[i-1])/ (2 * np.pi * sigma[i] * h) 


            else:
                C = C-I_0 * np.log(r_position[i-1])/ (2 * np.pi * sigma[i-1] * h) + I_0 * np.log(r_position[i-1])/ (2 * np.pi * sigma[i] * h) 

            final = np.concatenate([final,(C - I_0 * np.log(r_masked)/ (2 * np.pi * sigma[i] * h) + angles_reshaped @ (result_reshaped * coefficients[:,2*i+1].reshape(-1,1)) + angles_reshaped @ (result_reshaped_neg * coefficients[:,2*i+2].reshape(-1,1)))],axis = 1)
 
    return final
r_position = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.5,1.51,1.55,1.6,1.7,2,2.5,2.8,2.9,3,3.1,10])+0.
sigma = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
#r_position = np.linspace(0.1,10,100)
r_0 = 2.15 
theta_0 = 0
R = 10
new_r = np.append(r_position,R)
r = np.linspace(0.01,10,100)
#sigma = 3+np.random.normal(loc=0, scale=1, size=new_r.shape)
sigma = 2 + np.cos(2*np.pi*new_r/2.5)
#prefactor = 1 / (0.3 * np.sqrt(1 * np.pi))
#exponent = -((new_r - 1)**2) / (2 * 0.3**2)
#sigma = 1 + prefactor * np.exp(exponent)
theta = np.linspace(0,2 *np.pi,100)
Z = solution(r,theta,r_position,sigma,r_0,theta_0,R,100)
R, Theta = np.meshgrid(r,theta)
X = R * np.cos(Theta)
Y = R * np.sin(Theta)
# Create contour plot in terms of X and Y
#plt.contourf(X, Y, Z, levels=100, cmap='tab20b')
#plt.colorbar()  # Add a colorbar to a plot
#plt.xlabel('X')
#plt.ylabel('Y')
#plt.title('Contour Plot of Z in Cartesian Coordinates')
#plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Contour plot
contour = ax1.contourf(X, Y, Z, levels=100, cmap='tab20b')
cbar = fig.colorbar(contour, ax=ax1)
cbar.set_label('$\\phi$', rotation=270, labelpad=20)

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Graf potenciala $\\phi$ pri radialni odvisnosti prevodnosti $\\sigma(r)$')
ax1.axis('equal')

ax1.axis('equal')  # Ensure aspect ratio that X and Y are equally scaled
# Line plot
ax2.plot(new_r[:len(new_r)-1], sigma[:len(new_r)-1], label='$\\sigma(r)$',color='black', linestyle='--',lw = 3)
ax2.set_title('Odvisnost prevodnosti od radialne razdalje')
ax2.set_xlabel('r')
ax2.set_ylabel('$\\sigma [\\frac{{1}}{\\Omega \\mathrm{m}}]$')
ax2.legend()
ax2.grid(True)

plt.show()
