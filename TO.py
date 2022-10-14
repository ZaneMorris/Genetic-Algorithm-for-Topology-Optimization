#%%
import numpy as np
import matplotlib.pyplot as plt
#TO(60,20,0.5,3.0,1.1)
#%%
def TOP(nelx, nely, VF, p, rmin):
    #Initialize x as the matrix of the elements - densities of which are the design variables
    x = VF*np.ones((nely, nelx))
    dc = np.zeros(x.shape)
    count = 0
    change = 1.0

    while change > 0.01: #maximum density value changes by less than 1%
        count += 1
        x_old = x
        #FE Analysis
        U = FE(nelx,nely,x,p)
        #Obj Function and Sensitivity Analysis
        KE = lk() #calculates the element stiffness matrix for subsequent calcs
        c = 0.0

        #calculating objective function evals, and sensitivity evals
        for elx in range(1,nelx+1): #the "+ 1" was added so that the loop would reach all elements
            for ely in range(1,nely+1):
                n1 = (nely+1)*(elx-1)+ely #upper left ELEMENT node number in global node numbers (for the element currently being evaluated)
                n2 = (nely+1)*elx+ely #upper right ELEMENT node number in global node numbers (for the element currently being evaluated)
                indices = np.array([2*n1-1, 2*n1, 2*n2-1, 2*n2, 2*n2+1, 2*n2+2, 2*n1+1, 2*n1+2]) - 1
                Ue = U[list(indices)] # [8x1] element displacement vector, via indexing global disp. vector
                c = c + (x[ely-1,elx-1]**p)*(np.dot(np.transpose(Ue),np.dot(KE,Ue))) #basic, unfiltered compliance calculation
                dc[ely-1,elx-1] = (-p*x[ely-1,elx-1]**(p-1))*(np.dot(np.transpose(Ue),np.dot(KE,Ue))) #basic, unfiltered compliance sensitivities
        
        #Filtering of Sensitivites
        dc = check(nelx,nely,rmin,x,dc) #returns filtered sensitivites, which are used in OC

        #Design Update by the Optimality Criteria Method
        x = OC(nelx,nely,x,VF,dc)

        #Print Results
        change = np.max(abs(x-x_old))
        print('It: %.1f' % count)
        print('Obj: %.3f' % c)
        plt.imshow(x,cmap='binary')
        plt.pause(1e-6)

    #Plot Densities
    print('Solution Found')
    plt.show()
    return

def FE(nelx,nely,x,p):
    KE = lk() #returns element stiffness matrix for each element (they are the same)
    K = np.zeros((2*(nelx+1)*(nely+1), 2*(nelx+1)*(nely+1)))
    F = np.zeros((2*(nelx+1)*(nely+1), 1))
    U = np.zeros((2*(nelx+1)*(nely+1), 1))
    for elx in range(1,nelx+1):
        for ely in range(1,nely+1):
            n1 = (nely+1)*(elx-1)+ely #upper left ELEMENT node number in global node numbers (for the element currently being evaluated)
            n2 = (nely+1)*elx+ely #upper right ELEMENT node number in global node numbers (for the element currently being evaluated)
            edof = np.array([[2*n1-1],[2*n1],[2*n2-1],[2*n2],[2*n2+1],[2*n2+2],[2*n1+1],[2*n1+2]]) #denotes the degrees of freedom (2 per node, numbered in the same manner) that this particular element has
            #(DISREGARD) the "- 1" in edof is used to switch matlab indexing to pythonic indexing
            #again, below the index is changed to be pythonic, but 
            ix = np.ix_((edof-1).reshape(len(edof),), (edof-1).reshape(len(edof),))
            #K[edof-1,edof-1] = K[edof-1,edof-1] + x[ely-1,elx-1]**p * KE #forming the global stiffness matrix
            K[ix] = K[ix] + (x[ely-1,elx-1]**p)*KE #forming the global stiffness matrix
    
    #Define Loads and Supports (Half MBB Beam)
    
    #cantilever beam
    F[(nely+1)*(2*(nelx+1) - 1)] = -1
    fixeddofs = np.arange(0,2*(nely+1),1)

    # 1/2 MBB Beam
    #F[1] = -1 #index 1 here would mean y direction loading on node 1
    #fixeddofs = np.concatenate((np.arange(0,2*(nely+1),2), np.array([2*(nelx+1)*(nely+1) - 1]))) #0 and "- 1" revision for python indexing
    
    alldofs = np.arange(0,2*(nelx+1)*(nely+1))
    freedofs = np.setdiff1d(alldofs,fixeddofs)

    #Solve static FEM
    ix = np.ix_(freedofs,freedofs)
    U[freedofs] = np.linalg.inv(K[ix]).dot(F[freedofs]) #returns an [len(freedofs) x 1] vector
    U[fixeddofs] = 0
    return U

def check(nelx,nely,rmin,x,dc):
    #by making rmin < 1 in a function call, the filtered sens. will = original sens, and thus the filter will be inactive
    dcn = np.zeros((nely,nelx))
    for i in range(1,nelx+1): ####add +1 here and below?
        for j in range(1,nely+1):
            sum = 0.0
            # first for loop defines the range of elements that are within the filter radius in x dir, where k is the element number
            # 2nd for loop does the same in the y dir
            for k in range(int(max(i-np.floor(rmin),1)), int(min(i+np.floor(rmin), nelx)+1)): #+1 on cap of indices to ensure the entire filter field is studied
                for L in range(int(max(j-np.floor(rmin),1)), int(min(j+np.floor(rmin),nely)+1)): #+1 on cap of indices to ensure the entire filter field is studied
                    fac = rmin - ((i-k)**2 + (j-L)**2)**0.5
                    sum += max(0,fac)
                    dcn[j-1,i-1] = dcn[j-1,i-1] + max(0,fac)*x[L-1,k-1]*dc[L-1,k-1] #adding "- 1" to dcn indices for python
                    #this above changes the element at the center of the convolution (i,j) based on all surrounding densities
                    #in this way, it removes checkerboarding, but makes not as clean of edges
            dcn[j-1,i-1] = dcn[j-1,i-1] / (x[j-1,i-1]*sum)
    return dcn

def OC(nelx,nely,x,VF,dc):
    l1 = 0
    l2 = 100e3
    move = 0.2
    x_min = 0.001*np.ones(x.shape)
    x_max = np.ones(x.shape)
    while (l2-l1) > 1e-4: #while bisectioning bracket is greater than conv. tol.
        lmid = 0.5*(l2+l1)
        
        #x updated to satisfy dL/dx = dc/dx + l*dV/dx = 0, given a lagrange multiplier
        xnew = np.maximum(x_min, np.maximum(x-move, np.minimum(x_max, np.minimum(x+move, x*(-dc/lmid)**0.5)))) #here, 0.001 is x_min to avoid singularity
        
        #update lagrange multiplier l to satisfy dL/dl = V(x) - fV_0 = 0
        if (xnew.sum() - VF*nelx*nely) > 0: #if the current solution's volume is greater than the required volume fraction
            l1 = lmid
        else:
            l2 = lmid
    return xnew

def lk():
    E = 1.0
    nu = 0.3
    k = np.array([0.5-(nu/6), 0.125+(nu/8), -0.25-(nu/12), -0.125+(3*nu/8),
    -0.25+(nu/12), -0.125-(nu/8), nu/6, 0.125-(3*nu/8)])

    KE = (E/(1 - nu**2))*np.array([[k[0],k[1],k[2],k[3],k[4],k[5],k[6],k[7]],
                                   [k[1],k[0],k[7],k[6],k[5],k[4],k[3],k[2]],
                                   [k[2],k[7],k[0],k[5],k[6],k[3],k[4],k[1]],
                                   [k[3],k[6],k[5],k[0],k[7],k[2],k[1],k[4]],
                                   [k[4],k[5],k[6],k[7],k[0],k[1],k[2],k[3]],
                                   [k[5],k[4],k[3],k[2],k[1],k[0],k[7],k[6]],
                                   [k[6],k[3],k[4],k[1],k[2],k[7],k[0],k[5]],
                                   [k[7],k[2],k[1],k[4],k[3],k[6],k[5],k[0]]])
    return KE

def comeon():
    print('yea she works')
    return

x = np.array([[1e-4,1e-4,1e-4,1],[1,1,1,1]])

TOP(20,10,0.4,3.0,1.2)