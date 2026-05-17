import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.linalg as sla


np.random.seed(1)

#Functions

#algorithm for creating matrices
def create_matrix(c, N, W):
    rrg = nx.random_regular_graph(c, N)         #create graph
    H = nx.to_numpy_array(rrg) * (-1)           #construct matrix from graph
    E = []
    for i in range(N):
        E.append(np.random.uniform(-W/2, W/2))
        H[i, i] += E[i]
    evalues, evectors = sla.eig(H)      	#derive eigenvalues
    evalues = np.real(evalues)
    return H, E, evalues, evectors

#search neighbours in RRG
def search_neighbors(H):
    neighbors = []
    for i in range(N):
        node_neighbors = [j for j in range(N) if H[i, j] != 0 and i != j]
        neighbors.append(node_neighbors)
    return neighbors


# derive cavity precisions
def calc_cavity_precisions(H, neighbors, Lambda, epsilon):
    omega1_values = np.ones((N, N), dtype=np.complex128)  # Initialize omega1_values as complex numbers
    
    for t in range(T):                   # iteration steps
        max_difference = 0
        
        for i in range(N):                  # iterate over all nodes
            for j in neighbors[i]:          # iterate over neighbours of nodes
                sum_omega = 0
                for k in neighbors[i]:      # iterate again over neighbours and exclude one node
                    if k != j:   
                        sum_omega += 1 / omega1_values[k, i]
                        
                new_omega1_value = (1 - alpha) * (Lambda + epsilon - (E[i] * 1j) + sum_omega) + alpha * omega1_values[i, j]
                old_omega1_value = omega1_values[i, j]
                omega1_values[i, j] = new_omega1_value

        		# Check for equilibrium for all cavity precisions          
                difference = np.abs(new_omega1_value - old_omega1_value)
                if difference > max_difference:
                    max_difference = difference
                
        if np.abs(max_difference) < epsilon:
            print(f'iteration ends after {t} cycles.')
            break
                
            
    return omega1_values


#derive marginal precisions (omega2_values)
def calc_marginal_precisions(H, neighbors, omega1_values, Lambda, epsilon):
    omega2_values = np.ones((N), dtype=np.complex128)
    
    for i in range(N):
        sum_omega = 0
        for j in neighbors[i]:
            sum_omega += 1 / omega1_values[j, i]
        E = np.random.uniform(-W/2, W/2)
        
        omega2_values[i] = Lambda + epsilon - (E * 1j) + sum_omega

    return omega2_values

#------------------------------------------------------------------------------------------

c = 3                           # connectivity
N = 2**11                       # number of nodes / system size
W = 0.3                         # max energy depth
T = 200                         # max iteration cycles
alpha = 0.2                     # parameter for numerical stability
epsilon = 10**-6                # parameter for equilibrium
Lambda_values = np.linspace(-np.pi * 1j, np.pi * 1j, 100)           # real part of cavity shift / range of eigenvalues
epsilon_values = [0.001]                                            # imaginary part of cavity shift

H, E, evalues, _ = create_matrix(c, N, W)                           # create RRG
neighbors = search_neighbors(H)                                     # determine neighbors from RRG

fig = plt.figure(figsize=(8,5))
ax1 = fig.add_subplot(1,1,1)

#derive spectral density for various epsilon values
for epsilon in epsilon_values:
    # Calculate cavity precisions and marginal precisions for each Lambda value
    rho_values = []
    for Lambda in Lambda_values:
        omega1_values = calc_cavity_precisions(H, neighbors, Lambda, epsilon)
        omega2_values = calc_marginal_precisions(H, neighbors, omega1_values, Lambda, epsilon)
        
        # Calculate spectral density based on omega2_values
        G_values = 1j / omega2_values
        rho = sum(np.imag(G_values)) / (N * np.pi)
        rho_values.append(rho)

    # Plot spectral density
    ax1.plot(np.imag(Lambda_values), rho_values, label='Cavity Method')
    

# Plot histogram of eigenvalues normalized by N
ax1.hist(evalues, bins=100, range = (-np.pi, np.pi), alpha=0.6, color='red', label='Diagonalization', density=True)
ax1.set_xlabel(r'$\lambda$')
ax1.set_ylabel(r'$\rho(\lambda)$')
ax1.set_title('Spectral Density')
ax1.grid(True)
ax1.legend()
plt.show()
