import numpy as np
import pandas as pd

import bspline # '0.1.1'
import bspline.splinelab as splinelab


class FQI():
    def __init__(self,
               S0,
               sigma,
               M,
               T,
               r = 0.05,
               mu = 0.05,
               N_MC = 10000,
               K = 120,
               risk_lambda = 0.001):
        
        
        self.S0 = S0 # initial stock price
        self.sigma = sigma #volatility
        self.M = M #maturity
        self.T = T #number of time stpes
        self.mu = mu # drift
        self.r = r
        self.N_MC = N_MC # number of Monte-Carlo paths
        self.K = K #strike
        self.risk_lambda = risk_lambda 

        self.delta_t = self.M / self.T                # time interval
        self.gamma = np.exp(- self.r * self.delta_t) 
        
        
        
    def Monte_Carlo(self):
        
        
        np.random.seed(42) # Fix random seed
        # stock price
        S = pd.DataFrame([], index=range(1, self.N_MC+1), columns=range(self.T+1))
        S.loc[:,0] = self.S0

        RN = pd.DataFrame(np.random.randn(self.N_MC,self.T), index=range(1, self.N_MC+1), columns=range(1, self.T+1))

        for t in range(1, self.T+1):
            S.loc[:,t] = S.loc[:,t-1] * np.exp((self.mu - 1/2 * self.sigma**2) * self.delta_t + self.sigma * np.sqrt(self.delta_t) * RN.loc[:,t])

        # Compute delta_S and delta-S_hat of (eq.3)
        delta_S = S.loc[:,1:self.T].values - np.exp(self.r * self.delta_t) * S.loc[:,0:self.T-1]
        delta_S_hat = delta_S.apply(lambda x: x - np.mean(x), axis=0)

        # state variable (eq.2), this is a dataframe:
        X = - (self.mu - 1/2 * self.sigma**2) * np.arange(self.T+1) * self.delta_t + np.vectorize(np.log)(S)   # delta_t here is due to their conventions

        
        X_min = np.min(np.min(X))
        X_max = np.max(np.max(X))


        p = 4              # order of spline (as-is; 3 = cubic, 4: B-spline?)
        ncolloc = 12
        tau = np.linspace(X_min,X_max,ncolloc)  # These are the sites to which we would like to interpolate

        # k is a knot vector that adds endpoints repeats as appropriate for a spline of order p
        # To get meaninful results, one should have ncolloc >= p+1
        k = splinelab.aptknt(tau, p) 
                                 
        # Spline basis of order p on knots k
        basis = bspline.Bspline(k, p)        

        num_t_steps = self.T + 1
        num_basis =  ncolloc # len(k) #

        data_mat_t = np.zeros((num_t_steps, self.N_MC,num_basis ))


        # fill it, expand function in finite dimensional space
        # in neural network the basis is the neural network itself
        
        for i in np.arange(num_t_steps):
            x = X[:,i]
            data_mat_t[i,:,:] = np.array([ basis(el) for el in x ])
 

        self.S = S
        self.delta_S_hat = delta_S_hat
        self.delta_S = delta_S
        self.data_mat_t = data_mat_t
        
        
        return S, X, delta_S_hat, data_mat_t
        
        
        
        
    def terminal_payoff(self, ST, K): # call option
        
        payoff = max(ST-K, 0)
        return payoff
#====================== DP ===================================================#
    def function_A_vec(self, t, delta_S_hat, data_mat, reg_param):
   
    
        X_mat = data_mat[t, :, :]
        num_basis_funcs = X_mat.shape[1]
        this_dS = delta_S_hat.values[:, t]
        hat_dS2 = (this_dS ** 2).reshape(-1, 1)
        A_mat = np.dot(X_mat.T, X_mat * hat_dS2) + reg_param * np.eye(num_basis_funcs)
    
    
        return A_mat
    
    def function_B_vec(self, t,
                   Pi_hat, 
                   delta_S_hat,
                   S,
                   data_mat
                  ):
   
        
       # delta_S_hat= self.delta_S_hat
       # S = self.S
       # data_mat = self.delta_S_hat
        gamma= self.gamma
        #risk_lambda= self.risk_lambda
        
        # coef = 1.0/(2 * gamma * risk_lambda)
        # override it by zero to have pure risk hedge
        coef = 0. # keep it
        
        delta_S_t = S.loc[:,t+1].values - (1/gamma) * S.loc[:,t]
        tmp = Pi_hat.values[:,t+1] * delta_S_hat.values[:, t] + coef * delta_S_t 
        X_mat = data_mat[t, :, :]  # matrix of dimension N_MC x num_basis
        
        B_vec = np.dot(X_mat.T, tmp)
    
  
        return B_vec    
        
        
        
    def optimal_hedge(self):
        
        # portfolio value
        Pi = pd.DataFrame([], index=range(1, self.N_MC+1), columns=range(self.T+1))
        Pi.iloc[:,-1] = self.S.iloc[:,-1].apply(lambda x: self.terminal_payoff(x, self.K))

        Pi_hat = pd.DataFrame([], index=range(1, self.N_MC+1), columns=range(self.T+1))
        Pi_hat.iloc[:,-1] = Pi.iloc[:,-1] - np.mean(Pi.iloc[:,-1])

        # optimal hedge
        a = pd.DataFrame([], index=range(1, self.N_MC+1), columns=range(self.T+1))
        a.iloc[:,-1] = 0

        reg_param = 1e-3
        for t in range(self.T-1, -1, -1):
            A_mat = self.function_A_vec(t, self.delta_S_hat, self.data_mat_t, reg_param).astype('float64')
            B_vec = self.function_B_vec(t, Pi_hat, self.delta_S_hat, self.S, self.data_mat_t).astype('float64')

   
            phi = np.dot(np.linalg.inv(A_mat), B_vec)
            a.loc[:,t] = np.dot(self.data_mat_t[t,:,:],phi)
            Pi.loc[:,t] = self.gamma * (Pi.loc[:,t+1] - a.loc[:,t] * self.delta_S.loc[:,t])
            Pi_hat.loc[:,t] = Pi.loc[:,t] - np.mean(Pi.loc[:,t])

        a = a.astype('float')
        Pi = Pi.astype('float')
        Pi_hat = Pi_hat.astype('float')
        self.a = a
        self.Pi = Pi
        
        return a, Pi       
    
    def function_C_vec(self,t, data_mat, reg_param):
   
    
        X_mat = data_mat[t, :, :]
        num_basis_funcs = X_mat.shape[1]
    
        C_mat = np.dot(X_mat.T , X_mat) + reg_param * np.eye(num_basis_funcs)
    
        return C_mat
    
    def function_D_vec(self, t, Q, R, data_mat):
    
        gamma = self.gamma
        X_mat = data_mat[t, :, :]
        par = R.values[:,t] + gamma* Q.values[:,t+1]
        
        D_vec = np.dot(X_mat.T, par)
    
        return D_vec
        
        
    def Optimal_Q_DP(self):
        
        R = pd.DataFrame([], index=range(1, self.N_MC+1), columns=range(self.T+1))
        R.iloc[:,-1] = - self.risk_lambda * np.var(self.Pi.iloc[:,-1])

        # The backward loop
        for t in range(self.T-1, -1, -1):
    
    
        # Compute rewards corrresponding to observed actions
    
            R.iloc[:,t] = self.gamma*self.a.iloc[:,t] * self.delta_S.iloc[:,t] - self.risk_lambda * np.var(self.Pi.iloc[:,t])
       
        Q = pd.DataFrame([], index=range(1, self.N_MC+1), columns=range(self.T+1))
        Q.iloc[:,-1] = - self.Pi.iloc[:,-1] - self.risk_lambda * np.var(self.Pi.iloc[:,-1])

        reg_param = 1e-3
        for t in range(self.T-1, -1, -1):
    ######################
            C_mat = self.function_C_vec(t,self.data_mat_t,reg_param).astype('float64')
            D_vec = self.function_D_vec(t, Q,R,self.data_mat_t).astype('float64')
            omega = np.dot(np.linalg.inv(C_mat), D_vec)
    
            Q.loc[:,t] = np.dot(self.data_mat_t[t,:,:], omega)
    
        Q = Q.astype('float')
        self.Q = Q
        
        return Q
    
    def off_policy_data(self):
        
        eta = 0.5 #  0.5 # 0.25 # 0.05 # 0.5 # 0.1 # 0.25 # 0.15
        reg_param = 1e-3
        np.random.seed(42) # Fix random seed

        # disturbed optimal actions to be computed 
        a_op = pd.DataFrame([], index=range(1, self.N_MC+1), columns=range(self.T+1))
        a_op.iloc[:,-1] = 0

        # also make portfolios and rewards
        # portfolio value
        Pi_op = pd.DataFrame([], index=range(1, self.N_MC+1), columns=range(self.T+1))
        Pi_op.iloc[:,-1] = self.S.iloc[:,-1].apply(lambda x: self.terminal_payoff(x, self.K))

        Pi_op_hat = pd.DataFrame([], index=range(1, self.N_MC+1), columns=range(self.T+1))
        Pi_op_hat.iloc[:,-1] = Pi_op.iloc[:,-1] - np.mean(Pi_op.iloc[:,-1])

        # reward function
        R_op = pd.DataFrame([], index=range(1, self.N_MC+1), columns=range(self.T+1))
        R_op.iloc[:,-1] = - self.risk_lambda * np.var(Pi_op.iloc[:,-1])

        # The backward loop
        for t in range(self.T-1, -1, -1):
    
            a_op.iloc[:,t] = self.a.iloc[:,t] # this is the optimal hedge found in DP
    
        # disturb these values by a random noise
    
            a_op.iloc[:,t] = a_op.iloc[:,t].apply(lambda x: x * np.random.uniform(1-eta,1+eta))
    
    
    # Compute portfolio values corresponding to observed actions
    
            Pi_op.iloc[:,t] = self.gamma*(Pi_op.iloc[:,t+1] - a_op.iloc[:,t]*self.delta_S.iloc[:,t] )
            Pi_op_hat.iloc[:,t] = Pi_op.iloc[:,t] - np.mean(Pi_op.iloc[:,t])
    
    # Compute rewards corrresponding to observed actions
    
            R_op.iloc[:,t] = self.gamma*a_op.iloc[:,t] * self.delta_S.iloc[:,t] - self.risk_lambda * np.var(Pi_op.iloc[:,t])
    
        a = a_op.copy()      # distrubed actions
        self.Pi = Pi_op.copy()    # disturbed portfolio values
        self.Pi_hat = Pi_op_hat.copy()
        self.R = R_op.copy()
        
        num_MC = a.shape[0] # number of simulated paths
        num_TS = a.shape[1] # number of time steps
        a_1_1 = a.values.reshape((1, num_MC, num_TS))
        a_1_2 = 0.5 * a_1_1**2
        ones_3d = np.ones((1, num_MC, num_TS))
        A_stack = np.vstack((ones_3d, a_1_1, a_1_2)) # This is the vector (1, a, 1/2 a^2)        
        
        data_mat_swap_idx = np.swapaxes(self.data_mat_t,0,2)
        A_2 = np.expand_dims(A_stack, axis=1) # becomes (3,1,10000,21)
        data_mat_swap_idx = np.expand_dims(data_mat_swap_idx, axis=0)  # becomes (1,12,10000,21)

        Psi_mat = np.multiply(A_2, data_mat_swap_idx) # this is a matrix of size 3 x num_basis x num_MC x num_steps
        Psi_mat = Psi_mat.reshape(-1, self.N_MC, self.T+1, order='F')
        
        Psi_1_aux = np.expand_dims(Psi_mat, axis=1)
        Psi_2_aux = np.expand_dims(Psi_mat, axis=0)
        S_t_mat = np.sum(np.multiply(Psi_1_aux, Psi_2_aux), axis=2) 
        self.S_t_mat = S_t_mat
        self.Psi_mat = Psi_mat
        
        del Psi_1_aux, Psi_2_aux, data_mat_swap_idx, A_2
        
        
    def function_S_vec(self,t, S_t_mat, reg_param):
   
    
        num_Qbasis = S_t_mat.shape[1]
        S_mat_reg = S_t_mat[:,:,t] + reg_param*np.eye(num_Qbasis)
    
    
        return S_mat_reg
        
    def function_M_vec(self,t,
                   Q_star, 
                   R, 
                   Psi_mat_t, 
                   gamma):
    
        gamma = self.gamma
        N_MC = R.shape[0]   
        M0 = (R.iloc[:,t] + gamma * Q_star.iloc[:,t+1]).to_numpy() # M0 has dimension (N_MC,)
        M0 = M0.reshape(-1, 1)
    
        M_t = np.sum(Psi_mat_t.T * M0 ,axis=0) 
    
    

        return M_t     
        
        
    def Optimal_Q_RL(self):
        
       

        # implied Q-function by input data (using the first form in Eq.(68))
        Q_RL = pd.DataFrame([], index=range(1,self.N_MC+1), columns=range(self.T+1))
        Q_RL.iloc[:,-1] = - self.Pi.iloc[:,-1] -self.risk_lambda * np.var(self.Pi.iloc[:,-1])

        # optimal action
        a_opt = np.zeros((self.N_MC,self.T+1))
        a_star = pd.DataFrame([], index=range(1, self.N_MC+1), columns=range(self.T+1))
        a_star.iloc[:,-1] = 0

        # optimal Q-function with optimal action
        Q_star = pd.DataFrame([], index=range(1, self.N_MC+1), columns=range(self.T+1))
        Q_star.iloc[:,-1] = Q_RL.iloc[:,-1]

        # max_Q_star_next = Q_star.iloc[:,-1].values 
        max_Q_star = np.zeros((self.N_MC,self.T+1))
        max_Q_star[:,-1] = Q_RL.iloc[:,-1].values

        num_basis = self.data_mat_t.shape[2]
        
        reg_param = 1e-3
        hyper_param =  1e-1

        # The backward loop
        for t in range(self.T-1, -1, -1):
    
            # calculate vector W_t
            S_mat_reg = self.function_S_vec(t,self.S_t_mat,reg_param).astype('float64')
            M_t = self.function_M_vec(t,Q_star, self.R, self.Psi_mat[:,:,t], self.gamma).astype('float64')
            W_t = np.dot(np.linalg.inv(S_mat_reg),M_t)  # this is an 1D array of dimension 3M
    
        # reshape to a matrix W_mat  
            W_mat = W_t.reshape((3, num_basis), order='F')  # shape 3 x M 
        
    # make matrix Phi_mat
            Phi_mat = self.data_mat_t[t,:,:].T  # dimension M x N_MC

    # compute matrix U_mat of dimension N_MC x 3 
            U_mat = np.dot(W_mat, Phi_mat)
    
    # compute vectors U_W^0,U_W^1,U_W^2 as rows of matrix U_mat  
            U_W_0 = U_mat[0,:]
            U_W_1 = U_mat[1,:]
            U_W_2 = U_mat[2,:]

    # IMPORTANT!!! Instead, use hedges computed as in DP approach:
    # in this way, errors of function approximation do not back-propagate. 
    # This provides a stable solution, unlike
    # the first method that leads to a diverging solution 
            A_mat = self.function_A_vec(t, self.delta_S_hat, self.data_mat_t, reg_param).astype('float64')
            B_vec = self.function_B_vec(t, self.Pi_hat, self.delta_S_hat, self.S, self.data_mat_t).astype('float64')
    # print ('t =  A_mat.shape = B_vec.shape = ', t, A_mat.shape, B_vec.shape)
            phi = np.dot(np.linalg.inv(A_mat), B_vec)
    
            a_opt[:,t] = np.dot(self.data_mat_t[t,:,:],phi)
            a_star.loc[:,t] = a_opt[:,t] 

            max_Q_star[:,t] = U_W_0 + a_opt[:,t] * U_W_1 + 0.5 * (a_opt[:,t]**2) * U_W_2       
    
    # update dataframes     
            Q_star.loc[:,t] = max_Q_star[:,t]
    
    # update the Q_RL solution given by a dot product of two matrices W_t Psi_t
            Psi_t = self.Psi_mat[:,:,t].T  # dimension N_MC x 3M  
            Q_RL.loc[:,t] = np.dot(Psi_t, W_t)
    
    # trim outliers for Q_RL
            up_percentile_Q_RL =  95 # 95
            low_percentile_Q_RL = 5 # 5
    
            low_perc_Q_RL, up_perc_Q_RL = np.percentile(Q_RL.loc[:,t],[low_percentile_Q_RL,up_percentile_Q_RL])
    
    # print('t = %s low_perc_Q_RL = %s up_perc_Q_RL = %s' % (t, low_perc_Q_RL, up_perc_Q_RL))
    
    # trim outliers in values of max_Q_star:
            flag_lower = Q_RL.loc[:,t].values < low_perc_Q_RL
            flag_upper = Q_RL.loc[:,t].values > up_perc_Q_RL
            Q_RL.loc[flag_lower,t] = low_perc_Q_RL
            Q_RL.loc[flag_upper,t] = up_perc_Q_RL
    
        return Q_RL, a_star
        
        
        
        