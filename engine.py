import numpy as np
import numba as nb


def uniform_lattice(N):
    return np.ones((N,N))


def random_lattice(N):
    return 2*np.random.randint(0, 2, (N,N)) - 1


@nb.jit(nopython=True)
def single_step(lattice, T):
    """Sweeps through lattice once, updating the spins in place using Metropolis-Hastings"""
    N = lattice.shape[0]
    for i in range(N):
        for j in range(N):
            spin = lattice[i][j]
            iplus1 = 0 if i == N-1 else i+1
            jplus1 = 0 if j == N-1 else j+1
            iminus1 = N-1 if i == 0 else i-1
            jminus1 = N-1 if j == 0 else j-1
            neighbour_1 = lattice[iplus1][j]
            neighbour_2 = lattice[iminus1][j]
            neighbour_3 = lattice[i][jplus1]
            neighbour_4 = lattice[i][jminus1]
            neighbours_product_sum = spin * (neighbour_1 + neighbour_2 + neighbour_3 + neighbour_4)
            E_0 = -1 * neighbours_product_sum
            delta_E = -2*E_0
            if delta_E < 0:
                lattice[i][j] *= -1
            else:
                p = np.random.rand()
                if np.exp(-1*delta_E/T) > p:
                    lattice[i][j] *= -1


@nb.jit(nopython=True)
def energy(lattice):
    """Calculate energy of given lattice"""
    N = lattice.shape[0]
    # Calculate sum of s_i * s_j
    sisj_sum = 0
    for i in range(N):
        for j in range(N):
            spin = lattice[i][j]
            iplus1 = 0 if i == N-1 else i+1
            jplus1 = 0 if j == N-1 else j+1
            iminus1 = N-1 if i == 0 else i-1
            jminus1 = N-1 if j == 0 else j-1
            neighbour_1 = lattice[iplus1][j]
            neighbour_2 = lattice[iminus1][j]
            neighbour_3 = lattice[i][jplus1]
            neighbour_4 = lattice[i][jminus1]
            neighbours_sisj_sum = spin * (neighbour_1 + neighbour_2 + neighbour_3 + neighbour_4)
            sisj_sum += neighbours_sisj_sum
    energy = -1 * sisj_sum
    return energy


def magnetisation(lattice):
    """Returns magnetisation of lattice"""
    return np.sum(lattice)


class Ising:
    def __init__(self, T, initial_lattice, N=None):
        if type(initial_lattice) == str:
            if initial_lattice == 'uniform':
                self.initial_lattice = uniform_lattice(N)
            elif initial_lattice == 'random':
                self.initial_lattice = random_lattice(N)
        else:
            self.initial_lattice = initial_lattice
        
        self.T = T

        if type(initial_lattice) == str:
            self.N = N
        else:
            self.N = initial_lattice.shape[0]
    

    def evolve(self, steps):
        """Evolve lattice for specific number of steps, returns arrays of (magnetisations, energies)"""
        lattice = self.initial_lattice
        magnetisations = []
        energies = []

        for i in range(steps):
            single_step(lattice, self.T)
            magnetisations.append(magnetisation(lattice))
            energies.append(energy(lattice))
        
        return lattice, np.array(magnetisations), np.array(energies)
    

    def equilibrium(self, max_coeff = 0.2):
        lattice = self.initial_lattice
        T = self.T
        
        equilibrium_window = 1000
        magnetisations = []
        equilibrium = False
        time = 0
        
        while not equilibrium:
            time += 1
            if time == 20*equilibrium_window:
                equilibrium_window *= 2
            time_window = np.arange(equilibrium_window)
            
            single_step(lattice, T)
            M = magnetisation(lattice)
            if (time > equilibrium_window) and (time % int(equilibrium_window/2) == 0):
                M_window = magnetisations[-equilibrium_window:]
                a, b = np.polyfit(time_window, M_window, 1)
                if abs(a) < max_coeff:
                    half = int(equilibrium_window/2)
                    a1, b1 = np.polyfit(time_window[:half], M_window[:half], 1)
                    a2, b2 = np.polyfit(time_window[half:], M_window[half:], 1)
                    if abs(a1) < max_coeff and abs(a2) < max_coeff:
                        equilibrium = True
            magnetisations.append(M)
            if time == 200000:
                break
        return lattice, np.array(magnetisations), time


