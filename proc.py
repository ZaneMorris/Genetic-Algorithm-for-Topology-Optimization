import os
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
#from Project_SupportingRoutines import unique

def GA(pop, fem, f, m, pc, pm, pe, i_max=100, consec_lim=300, plot=False):
    
    # set initial f_tilde (arbitrary)
    f_tilde = 5e3 #CHANGE ACCORIDNG TO PAPER
    
    # logging variables
    elite_log = []
    iter, consec = 0, 0
    switch = False

    # creating log files and new directory
    file = open('Results.txt','w') #creates new file
    
    while iter <= i_max and consec <= consec_lim:

        status, areas, nc = find_catagor_bodies(fem, pop)

        # evaluate fitness of each population member m
        Fs, f_tilde_opt, feas_count, pens, por = obj(pop, fem, f, f_tilde, status, areas, nc)

        # update f_tilde
        if bool(f_tilde_opt): #if f_tilde_opt isnt empty, meaning feas solns exist
            f_tilde, switch = update_f_tilde(f_tilde, f_tilde_opt, switch)

        # save the elite member, and pop it off of the population and Fs lists
        pop, elites, elite, elite_F = elitism(pop, Fs, pe)

        ### GENERATION EVALUATION
        
        # selection
        mpool = selection(pop, Fs)
        # temp2, uniq_mpool_ind = np.unique(np.array(mpool), return_index=True, axis=0)
        # print('Number of Unique mpool membs is: ', len(uniq_mpool_ind))

        # crossover and mutation
        mpool = crossover(fem, mpool, pc, pm)

        # update population
        pop = np.array(mpool + elites)

        elite_log.append(elite)
        consec = conv_check(iter, consec, elite_log)
        iter += 1
        print('iter: ', iter)
        print('Best: ', elite_F)

        # POST-PROCESSING
        elite_vf = np.sum(np.array(elite))/fem.dv
        temp1, uniq_pop_ind = np.unique(pop, return_index=True, axis=0)
        temp2, uniq_mpool_ind = np.unique(np.array(mpool), return_index=True, axis=0)
        temp3, uniq_elite_ind = np.unique(np.array(elites), return_index=True, axis=0)
        unq_pop, unq_mpool, unq_elite = len(uniq_pop_ind), len(uniq_mpool_ind), len(uniq_elite_ind)

        print(unq_pop, unq_mpool, unq_elite, feas_count)
        file.write(str(elite_F)+' '+ str(unq_pop)+' '+ str(unq_mpool)+' '+ str(unq_elite)+' '
                    + str(feas_count)+' '+str(pens[0])+' '+str(pens[1])+' '+str(pens[2])+' '
                    + str(pens[3])+' '+str(por[0])+' '+str(por[1])+' '+str(por[2])+' '+str(elite_vf)+'\n')

        if plot:
            plt.imshow(elite,cmap='binary')
            plt.title(str(iter))
            plt.pause(1e-6)
    
    file.close()

    file = open('Results_sol.txt','w')
    file.write(str(elite))
    file.close()

    print('Solution Found')
    print(elite)

    #plt.show()
    plt.savefig('Results_topo.png',format='png')
    return elite, elite_F

def update_f_tilde(f_tilde, f_tilde_opt, switch):
    if switch: #post switch
        f_tilde = max(max(f_tilde_opt), f_tilde)
    else: #switch hasnt happened yet, but now switch is activated
        f_tilde = min(min(f_tilde_opt), f_tilde)
        switch = True
    return f_tilde, switch

def elitism(pop, Fs, pe):
    ne = int(np.ceil(len(pop)*pe))
    elites, elite_Fs = [], []
    for i in range(ne):
        elite_ind = min(range(len(Fs)), key=Fs.__getitem__)
        elites.append(pop[elite_ind])
        elite_Fs.append(Fs.pop(elite_ind))
        if i == 0:
            elite = elites[-1]
            elite_F = elite_Fs[-1]
        pop = np.delete(pop,elite_ind,axis=0)

    return pop, elites, elite, elite_F

def selection(pop, Fs):
    # remaps probability for selection based on rank
    # selects mating pool members using stochastic universal selection

    # Remapping
    # sorting is for worst performing -> best performing (where its max or min problem)
    inds = [] #indices that associate a sorted Fitness array with population array
    fmax = min(Fs)-1
    for ind, f in enumerate(Fs):
        if f > fmax:
            fmax = f
            inds.insert(0,ind)
        elif f == fmax:
            inds.insert(1,ind)
        else:
            i = 0
            while f < Fs[inds[i]]:
                i += 1
                if i > len(inds)-1:
                    break
            inds.insert(i,ind)

    rank = []
    rank_sum = 0
    for i in range(1,len(pop)+1):
        rank.append(i)
        rank_sum += i

    Ps = len(pop)*[None]
    for ind, r in zip(inds,rank):
        Ps[ind] = r/rank_sum

    # SUS Implementation
    CPs, i = [], 1
    CPs.append(0)
    while i < len(Ps)+1:
        CPs.append(sum(Ps[0:i]))
        i += 1
    
    #number of pointers = number of population members
    point_sep = 1/len(Ps)
    pointer1 = np.random.uniform(0,point_sep)
    pointers = [pointer1 + n*point_sep for n in range(len(Ps))]
    
    mpool = []
    for pointer in pointers: # for each pointer
        for j in range(1,len(Ps)+1): # check all regions
            if pointer > CPs[j-1] and pointer <= CPs[j]:
                #app_Fs = sorted(Fs,reverse=True)[j-1]
                mpool.append(pop[j-1])
                break
    return mpool

def mutate(fem, child, pm):
    for i in range(fem.nely):
        for j in range(fem.nelx):
            r = np.random.uniform()
            if r < pm:
                if child[i][j] == 0:
                    child[i][j] = 1
                elif child[i][j] == 1:
                    child[i][j] = 0
    return child

def crossover(fem,mpool,pc,pm):
    # pop_mating: the number of mating pool members that will actually mate (about 80%)
    pop_mating = int(pc*len(mpool)) #number of mpool members selected to mate
    if pop_mating % 2 != 0: #ensuring an even number of parents are selected
        pop_mating = pop_mating + 1

    mate_ind = random.sample(range(len(mpool)),pop_mating) # mpool indices selected to mate

    while bool(mate_ind):
        p1, p2 = mpool[mate_ind[-2]], mpool[mate_ind[-1]]
        c1, c2 = [], []
        for i in range(fem.nely):
            row1, row2 = [], []
            for j in range(fem.nelx):
                if np.random.uniform() < 0.5:
                    row1.append(p1[i][j])
                    row2.append(p2[i][j])
                else:
                    row1.append(p2[i][j])
                    row2.append(p1[i][j])
            c1.append(row1)
            c2.append(row2)

        # mutation is embedded in crossover, so as to not alter the elite member
        c1 = mutate(fem,c1,pm)
        c2 = mutate(fem,c2,pm)

        mpool[mate_ind[-2]] = np.array(c1)
        mpool[mate_ind[-1]] = np.array(c2)

        mate_ind = mate_ind[:-2]
    return mpool

def find_catagor_bodies(fem, pop): #subcatagorize into feasible and infeasible bodies
    """ status: 1 - conn_infeas, 2 - struct_infeas, 3 - feas """
    # arrays for disconnected topologies
    areas, nc, status = [], [], []
    for memb in pop:
        #find objects in topo, and the number of objects
        lab, num_feat = label(memb)

        body_indices = []
        # save list, where each list member contains the element indices (i,j) for each disconnected object in topo
        for item in range(1,num_feat+1):
            body_indices.append([(i,j) for i in range(len(memb)) for j in range(len(memb[0])) if lab[i][j] == item])
    
        if num_feat > 1:
            A = 0
            for body in body_indices:
                if not is_feasible(fem, body):
                    A += len(body) #adds #/eles in disconnected body to area total
            areas.append(A)
            nc.append(num_feat)
            status.append(1)

        else: #meaning only 1 body
            if is_feasible(fem, body_indices[0]): # need [0] b/c [[]] structure - if the single object in topo is structurally feas (fixtures and load dofs/nodes satisfied)
                areas.append(len(body_indices[0])) #areas for feas is still needed to compute VF
                status.append(3)
            else:
                areas.append(len(body_indices[0]))
                status.append(2)

    return status, areas, nc

def is_feasible(fem,body):

    struct_feas = False
    fixt_feas = False
    load_feas = False

    nodes = []
    dofs = [] #the dofs associated/connected to the elements in each disconntected object
    for tup in body:
        nodes.append((fem.nely+1)*tup[1]+tup[0])
        nodes.append(nodes[-1]+1)
        nodes.append((fem.nely+1)*(tup[1]+1) + tup[0])
        nodes.append(nodes[-1]+1)
    nodes = unique(nodes)
    # nodes -> dofs
    for node in nodes:
        dofs.append(2*node)
        dofs.append(2*node + 1)
    dofs = sorted(unique(dofs)) #sorting i think will make the next part faster

    #(fixeddofs should already be sorted, by virtue of how its constructed)
    fixed_match_count = 0
    for obj_dof in dofs:
        if np.any([obj_dof==fix_dof for fix_dof in fem.fixeddofs]):
            fixed_match_count += 1
            if fixed_match_count == 4: #this would only work for cantilevered beam (i.e. for MBB beam, need to check different dof groupings)
                fixt_feas = True
                break

    for load_dof in fem.loaddofs:
        if load_dof in dofs:
            load_feas = True
        else:
            load_feas = False
    
    if fixt_feas and load_feas:
        struct_feas = True #this list contains whether of not an object has both neccesary bc dofs and load dofs to run
    else:
        struct_feas = False

    return struct_feas

def obj(pop, fem, f, f_tilde, status, areas, nc):
    """f - desired vol frac"""
    Fs, p = [], 3 #where best_dv is the elite solution from last generation???
    i, j, feas_count = 0, 0, 0
    f_tilde_opt = []
    cpen, spen, vfpen = 0,0,0
    c_cnt, s_cnt, vf_cnt = 0,0,0

    for memb in pop:
        if status[i] == 1: # conn_infeas
            c_cnt += 1
            cpen += viol1(fem, areas[i], nc[j])
            Fs.append(f_tilde + viol1(fem, areas[i], nc[j]))
            i += 1
            j += 1 #seperate index for nc, because only as long as #/discon pop members
            continue
    
        elif status[i] == 2: # struct_infeas but conn_feas
            #Fs.append(f_tilde + viol1(fem, areas[i], 1))
            s_cnt += 1
            spen += viol1(fem, areas[i], 1)
            Fs.append(f_tilde + viol1(fem, areas[i], 1)) #fem.dv**2)
            i += 1
            continue

        elif status[i] == 3: # completely feas
            U = fem.FEA(memb, p)
            c = fem.compliance(U, memb, p)
            if areas[i]/fem.dv - f <= 0: # if volfrac is satisfied
                Fs.append(c)
                feas_count += 1
                f_tilde_opt.append(Fs[-1])
            else:
                vf_cnt += 1
                vfpen += viol2(fem, areas[i],f)
                Fs.append(f_tilde + viol2(fem, areas[i],f))
            i += 1

    # compute violation averages
    avg_cpen = cpen/c_cnt if c_cnt else 0
    avg_spen = spen/s_cnt if s_cnt else 0
    avg_vfpen = vfpen/vf_cnt if vf_cnt else 0
    avg_tot = avg_cpen + avg_spen + avg_vfpen
    c_por, s_por, vf_por = c_cnt/len(pop), s_cnt/len(pop), vf_cnt/len(pop)
    #with this structure, order of Fs is not different than pop, making elitism easier
    return Fs, f_tilde_opt, feas_count, [avg_tot, avg_cpen, avg_spen, avg_vfpen], [c_por, s_por, vf_por]

def viol1(fem, A, nc, gam2=1):
    gam1 = fem.dv
    return gam1*(nc-1) + gam2*A

def viol2(fem, A, f,gam=1):
    return gam*((A/(fem.dv)) - f)

def unique(numbs):
    unique = []
    for numb in numbs:
        if numb in unique:
            continue
        else:
            unique.append(numb)
    return unique

def conv_check(iter, consec, elite_log):
    if iter == 0:
        return 0
    elif np.array_equal(elite_log[-1], elite_log[-2]):
        consec += 1
    else:
        consec = 0
    return consec