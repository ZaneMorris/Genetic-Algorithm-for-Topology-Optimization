# Imports
import os
import ast
import numpy as np
import matplotlib.pyplot as plt

# Subroutines
def read_n_plot(test_num):
    import os
    import ast
    elite_F, uniq_pop, uniq_mpool, uniq_elite, feas_count = [],[],[],[],[]
    avg_tot, avg_cpen, avg_spen, avg_vfpen = [], [], [], []
    c_por, s_por, vf_por = [], [], []
    elite_vf = []
    file = open(os.getcwd()+'\\ProjectResults\\'+str(test_num)+'\\Results.txt')

    for line in file:
        data = line.split(' ')
        elite_F.append(float(data[0]))
        uniq_pop.append(float(data[1]))
        uniq_mpool.append(float(data[2]))
        uniq_elite.append(float(data[3]))
        feas_count.append(float(data[4]))
        avg_tot.append(float(data[5]))
        avg_cpen.append(float(data[6]))
        avg_spen.append(float(data[7]))
        avg_vfpen.append(float(data[8]))
        c_por.append(float(data[9]))
        s_por.append(float(data[10]))
        vf_por.append(float(data[11]))
        elite_vf.append(float(data[12]))
    
    file.close()
    with open(os.getcwd()+'\\ProjectResults\\'+str(test_num)+'\\Results_sol.txt','r') as file:
        sol = file.read().replace(" ", ",")
    


    # PLOT 1 #######################
    data_len = len(elite_F)
    t = [i for i in range(data_len)]
    fig1, ax = plt.subplots(1,3)
    fig1.set_figheight(4)
    fig1.set_figwidth(14)
    ax[0].plot(t, uniq_pop, color='m', label='Population')
    ax[0].set(xlabel='Iterations [-]', ylabel='#/Unique Individuals [-]')
    ax[0].legend()
    ax[1].plot(t, uniq_mpool, color='m', label='Mating Pool')
    ax[1].set(xlabel='Iterations [-]', ylabel='#/Unique Individuals [-]')
    ax[1].legend()
    ax[2].plot(t, uniq_elite, color='m', label='Elites')
    ax[2].set(xlabel='Iterations [-]', ylabel='#/Unique Individuals [-]')
    ax[2].legend()

    # PLOT 2 #########################
    plt.figure()
    plt.xlabel('Iteration [-]')
    plt.ylabel('Compliance [J]')
    plt.plot(t[88:], elite_F[88:], color='m')

    # PLOT 3 #########################
    plt.figure()
    plt.plot(avg_tot, color='k', label='Tot Avg Penalty')
    plt.plot(avg_cpen, color='r', label='Conn. Penalty')
    plt.plot(avg_spen, color='g', label='Stuct. Conn. Penalty')
    plt.plot(avg_vfpen, color='b', label='VF Penalty')
    plt.legend()

    # PLOT 4 #########################
    plt.figure()
    plt.plot(c_por, color='r', label='Conn. Infeas %')
    plt.plot(s_por, color='g', label='Struct. Infeas %')
    plt.plot(vf_por, color='b', label='VF. Infeas %')
    plt.plot([1-(c_por[i]+s_por[i]+vf_por[i]) for i in range(data_len)], color='k', label='Feasible %')
    plt.xlabel('Iterations [-]')
    plt.ylabel('% of Population')
    plt.legend()

    # PLOT 5 ############################
    plt.figure()
    plt.xlabel('Iterations [-]')
    plt.ylabel('Volume Fraction of Elite Individual [-]')
    plt.plot(elite_vf, color='m')

    # PLOT 6
    read_sol(test_num) #only works for the tests that have ints for results_sol  
    plt.show()

def read_sol(test_num):
    lis = []
    with open(os.getcwd()+'\\ProjectResults\\'+str(test_num)+'\\Results_sol.txt','r') as file:
        for line in file:
            if line[-1] == '\n':
                line = line[0:-1]
            if line[-2:] == ']]':
                line = line[0:-1]
            
            line = line[1:] #remove weither first bracket or space
            row = line.replace(" ",",")
            lis.append(ast.literal_eval(row))
    arr = np.array(lis)
    plt.matshow(arr, cmap='binary')
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([]) 
    plt.show()