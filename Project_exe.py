#Imports
from preproc import *
from proc import *
from postproc import *
from datastructures import *

#define FEM
nelx, nely = 20,10
fixeddofs = np.arange(0,2*(nely+1),1)
loaddofs = [(nely+1)*(2*(nelx+1) - 1)]
print('fixeddofs are: ', fixeddofs)
print('loaddofs are: ', loaddofs)
fem = FEM(nelx, nely, fixeddofs, loaddofs)

# x = np.zeros((fem.nely,fem.nelx))
# x[4:6,:] = 1
# x[:,13:] = 1
# plt.matshow(x,cmap='binary')
# ax = plt.gca()
# ax.set_xticks(np.arange(0, nelx-1, 1))
# ax.set_yticks(np.arange(0, nely-1, 1))
# ax.set_xticklabels([])
# ax.set_yticklabels([])
# ax.set_xticks(np.arange(-.5, nelx, 1), minor=True)
# ax.set_yticks(np.arange(-.5, nely, 1), minor=True)
# ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
# plt.savefig('start_random.png',format='png')
# plt.show()

# U = fem.FEA(x,3)
# c = fem.compliance(U,x,3)
# print(c)
# plt.matshow(ninenine,cmap='binary')

#problem parameters
f = 0.4 #mass fraction
m = 140 #population size
pe = 0.05 #percentage of elites kept
pc = 0.9 #probability of crossover
pm = 0.003 #probability of mutation

# starting conditions
pop = population_select('solid', m, fem)

# solver
sol, obj_sol = GA(pop, fem, f, m, pc, pm, pe, i_max=2000, plot=False)

#relocate results file to pertinent folder
test_num = 47
wdir = os.getcwd()
stor_dir = '\\ProjectResults\\'+str(test_num)
if not os.path.exists(wdir+stor_dir):
    os.makedirs(wdir+stor_dir)
os.rename(wdir+'\\Results.txt', wdir+stor_dir+'\\Results.txt')
os.rename(wdir+'\\Results_topo.png', wdir+stor_dir+'\\Results_topo.png')
os.rename(wdir+'\\Results_sol.txt', wdir+stor_dir+'\\Results_sol.txt')

#then, can call read_n_plot() from terminal to view results
#read_n_plot(2)