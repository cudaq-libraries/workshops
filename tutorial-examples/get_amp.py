import numpy as np

def get_thetas_unpack_restricted(singles,doubles,n_occupied,n_virtual):

    theta_1=np.zeros((2*n_occupied,2*n_virtual))
    theta_2=np.zeros((2*n_occupied,2*n_occupied,2*n_virtual,2*n_virtual))

    for p in range(n_occupied):
        for q in range(n_virtual):
            theta_1[2*p,2*q]=singles[p,q]
            theta_1[2*p+1,2*q+1]=singles[p,q]

    for p in range(n_occupied):
        for q in range(n_occupied):    
            for r in range(n_virtual):
                for s in range(n_virtual):
                    theta_2[2*p,2*q,2*s,2*r]=doubles[p,q,r,s]
                    theta_2[2*p+1,2*q+1,2*r+1,2*s+1]=doubles[p,q,r,s]
                    theta_2[2*p,2*q+1,2*r+1,2*s]=doubles[p,q,r,s]
                    theta_2[2*p+1,2*q,2*r,2*s+1]=doubles[p,q,r,s]

    return theta_1,theta_2


def uccsd_get_amplitude(single_theta,double_theta,n_electrons,n_orb):
    
    n_occupied = n_electrons // 2
    n_virtual = n_orb - n_occupied

    single_amplitude,double_amplitude=get_thetas_unpack_restricted(single_theta,double_theta,n_occupied,n_virtual)
    
    singles_alpha=[]
    singles_beta=[]
    doubles_mixed=[]
    doubles_alpha=[]
    doubles_beta=[]

    occupied_alpha_indices=[i*2 for i in range(n_occupied)]
    virtual_alpha_indices=[i*2 for i in range(n_virtual)]

    occupied_beta_indices=[i*2+1 for i in range(n_occupied)]
    virtual_beta_indices=[i*2+1 for i in range(n_virtual)]

    # Same spin single excitation
    for p in occupied_alpha_indices:
        for q in virtual_alpha_indices:
            singles_alpha.append(single_amplitude[p,q])
        
    for p in occupied_beta_indices:
        for q in virtual_beta_indices:
            singles_beta.append(single_amplitude[p,q])
                
    #Mixed spin double excitation
    for p in occupied_alpha_indices:
        for q in occupied_beta_indices:
            for r in virtual_beta_indices:
                for s in virtual_alpha_indices:
                    doubles_mixed.append(double_amplitude[p,q,r,s]) 

    # same spin double excitation
    n_occ_alpha=len(occupied_alpha_indices)
    n_occ_beta=len(occupied_beta_indices)
    n_virt_alpha=len(virtual_alpha_indices)
    n_virt_beta=len(virtual_beta_indices)

    for p in range(n_occ_alpha-1):
        for q in range(p+1,n_occ_alpha):
            for r in range(n_virt_alpha-1):
                for s in range(r+1,n_virt_alpha):

                    # Same spin: all alpha
                    doubles_alpha.append(double_amplitude[occupied_alpha_indices[p],occupied_alpha_indices[q],\
                                    virtual_alpha_indices[r],virtual_alpha_indices[s]])

    for p in range(n_occ_beta-1):
        for q in range(p+1,n_occ_beta):
            for r in range(n_virt_beta-1):
                for s in range(r+1,n_virt_beta):

                    # Same spin: all beta
                    doubles_beta.append(double_amplitude[occupied_beta_indices[p],occupied_beta_indices[q],\
                                    virtual_beta_indices[r],virtual_beta_indices[s]])

                                        

    return singles_alpha+singles_beta+doubles_mixed+doubles_alpha+doubles_beta


