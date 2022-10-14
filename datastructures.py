# Imports
import numpy as np

class FEM:
    def __init__(self, nelx, nely, fixeddofs, loaddofs):
        self.nelx = nelx
        self.nely = nely
        self.dv = self.nelx*self.nely
        self.fixeddofs = fixeddofs
        self.loaddofs = loaddofs 
        self.alldofs = np.arange(0,2*(self.nelx+1)*(self.nely+1)) 
    
    def lk(self):
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
    
    def FEA(self,x,p):
        """ x - population member - adapted from TO to work as list """
        # make x non-singular
        #x = np.where(np.array(x)==0,1e-4,1)
        x = np.where(x==0,1e-4,1)

        KE = self.lk() #returns element stiffness matrix for each element (they are the same)
        K = np.zeros((2*(self.nelx+1)*(self.nely+1), 2*(self.nelx+1)*(self.nely+1)))
        F = np.zeros((2*(self.nelx+1)*(self.nely+1), 1))
        U = np.zeros((2*(self.nelx+1)*(self.nely+1), 1))
        for elx in range(1,self.nelx+1):
            for ely in range(1,self.nely+1):
                n1 = (self.nely+1)*(elx-1)+ely #upper left ELEMENT node number in global node numbers (for the element currently being evaluated)
                n2 = (self.nely+1)*elx+ely #upper right ELEMENT node number in global node numbers (for the element currently being evaluated)
                edof = np.array([[2*n1-1],[2*n1],[2*n2-1],[2*n2],[2*n2+1],[2*n2+2],[2*n1+1],[2*n1+2]]) #denotes the degrees of freedom (2 per node, numbered in the same manner) that this particular element has
                #(DISREGARD) the "- 1" in edof is used to switch matlab indexing to pythonic indexing
                #again, below the index is changed to be pythonic, but 
                ix = np.ix_((edof-1).reshape(len(edof),), (edof-1).reshape(len(edof),))
                #K[edof-1,edof-1] = K[edof-1,edof-1] + x[ely-1,elx-1]**p * KE #forming the global stiffness matrix
                K[ix] = K[ix] + (x[ely-1,elx-1]**p)*KE #forming the global stiffness matrix
        
        #Define Loads and Supports (Half MBB Beam)
        F[self.loaddofs] = -1
        freedofs = np.setdiff1d(self.alldofs,self.fixeddofs)

        #Solve static FEM
        ix = np.ix_(freedofs,freedofs)
        U[freedofs] = np.linalg.inv(K[ix]).dot(F[freedofs]) #returns an [len(freedofs) x 1] vector
        U[self.fixeddofs] = 0
        return U

    def compliance(self, U, x, p):
        """ x - population member """
        # make x non-singular
        x = np.where(x==0,1e-4,1)

        KE = self.lk() #calculates the element stiffness matrix for subsequent calcs
        c = 0.0
        
        #calculating objective function evals, and sensitivity evals
        for elx in range(1,self.nelx+1): #the "+ 1" was added so that the loop would reach all elements
            for ely in range(1,self.nely+1):
                n1 = (self.nely+1)*(elx-1)+ely #upper left ELEMENT node number in global node numbers (for the element currently being evaluated)
                n2 = (self.nely+1)*elx+ely #upper right ELEMENT node number in global node numbers (for the element currently being evaluated)
                indices = np.array([2*n1-1, 2*n1, 2*n2-1, 2*n2, 2*n2+1, 2*n2+2, 2*n1+1, 2*n1+2]) - 1
                Ue = U[list(indices)] # [8x1] element displacement vector, via indexing global disp. vector
                c = c + (x[ely-1,elx-1]**p)*(np.dot(np.transpose(Ue),np.dot(KE,Ue))) #basic, unfiltered compliance calculation
        return float(c)



