import autograd.numpy as np
from autograd import grad
from autograd.numpy import sqrt
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import fsolve


def orth_constraint_closure(beta_curr):
    def orth_constraint_f(beta):
        return np.dot(beta_curr, beta)
    return orth_constraint_f


def unit_constraint_f(beta):
    return np.dot(beta, beta) - 1


def objective_closure(V):
    """
    Create the objective function for a given set of word-pair vectors
    in which the we are maximizing the projection of each word-pair
    vector on the basis vector.

    Parameters
    ----------
    V: array
        An MxN-length array storing a given normalized word-vector
        pair (i.e., word_1 - word_2) where M is the number of word
        pairs and N is the dimensionality of the word vector space.

    Returns
    -------
    objective: function
        The objective function that takes as input a given candidate
        loading vector and returns the value of the objective 
        function.
    """
    def objective(b):
        # Compute the value of the objective function for a given
        # basis function `b`
        return -1 * np.sum(np.dot(V, b))
    return objective


def grad_Lagrangian_closure(V, beta_curr):
    """
    Create the gradient of the Lagrangian function for finding
    the ith loading vector. The roots of this gradient function 
    are the next loading vector.

    Parameters
    ----------
    V: array
        An Nxlength array storing a given normalized word-vector
        pair. (i.e., word_1 - word_2)

    Returns
    -------
    objective: function
        The objective function that takes as input a given basis
        vector and returns the term of the objective function
        corresponding to a given word-vector pair.
    """
    def gradLagrangian(X):
        # X is an (N+i)-length array where N is the dimensionality
        # of the word-vectors and i is the index of the next
        # loading vector we are trying to find (we assume we
        # already have solved for i-1 loading vectors. 
        # 
        # This array wraps all of the arguments to the Lagrangian 
        # into a single array. The first N elemnts are loadings matrix. 
        # The next i-1 elements are the Lagrange multipliers for the 
        # orthogonal constraint to the currently computed loadings
        # vectors and the final element is the Lagrange multiplier
        # for the unit-norm constraint.
        beta, _lam_orth, _lam_unit = X

        # The derivative of the Lagrangian with respect to the betas
        dbeta = ((1/len(V) * np.sum(V, axis=0)) \
            - np.dot(beta_curr.T, _lam_orth) \
            - (_lam_unit * 2 * beta))

        # The derivative of the Lagrangian with respect to the 
        # Lagrange multipliers
        dlambda_orth = -1 * np.dot(beta_curr, beta)
        dlambda_unit = -2 * beta
        
        return dbeta, dlambda_orth, dlambda_unit
    return gradLagrangian


def solve_Lagrangian_closure(
        grad_lagrangian, 
        eq_orth, 
        n_dims, 
        n_loadings
    ):
        """
        Generate the function object that we will feed to scipy's 
        fsolve. This function is basically a wrapper around the gradient 
        of the Lagrangian.
        """
        def solve_lagrangian(args):
            # scipy's fsolver function requires a one-dimensional
            # array, so we pack all of the arguments into `args`. Now
            # we must unpack them
            beta = args[:n_dims]
            _lambda_orth = args[n_dims:n_dims+n_loadings]
            _lambda_unit = args[-1]

            # Compute the gradient of the Lagrangian
            dbeta, dlam_orth, dlam_unit = grad_lagrangian([
                beta,
                _lambda_orth,
                _lambda_unit
            ])

            # Reconcatenate into an array because scipy's fsolver 
            # expects the target function to return a single array
            return np.concatenate([
                dbeta,
                eq_orth(beta),
                np.array([unit_constraint_f(beta)])
            ])
        return solve_lagrangian


def solve_dim(V, B):
    """
    Compute the next loading vector.

    V: array
        The collection of normalized vectors that represent
        the word-pairs.
    B: array
        DxN array of loading vectors where D are the first D
        loading vectors and N is the dimensionality of the word
        vector space.

    Returns
    -------
    b_new: array
        The next (D+1) loading vector
    """
    the_eq_orth = orth_constraint_closure(B)
    the_objective = objective_closure(V)
    
    # Compute the gradient of the Lagrangian function
    grad_lagrangian_f = grad_Lagrangian_closure(V, B)
    
    # Generate the function that evaluates the gradient of the 
    # Lagrangian at a given input
    lagrangian_wrapper_f = solve_Lagrangian_closure(
        grad_lagrangian_f, 
        the_eq_orth, 
        len(B[0]),  # Number of dimensions
        len(B)      # Number of loadings
    )

    # Create an initial guess for solver. We'll just make it a vector
    # of all 1's
    init = np.concatenate([np.array([1.]), np.ones(len(V[0]) + len(B))])

    # Compute the roots of the gradient of the Lagrangian. These will
    # maximize the objective function subject to the orthonormal 
    # constraints.
    solution = fsolve(lagrangian_wrapper_f, init)
    
    return solution[:len(B[0])]


def solve_init(V):
    the_objective = objective_closure(V)
    print("Solving first dimension...")
    obj = the_objective
    solution = minimize(
        obj,
        np.concatenate([np.array([1.]), np.zeros(len(V[0])-1)]),
        options={
            'maxiter': 10000,
            'ftol': 0.0001
        },
        constraints=[
            {'type': 'eq', 'fun': unit_constraint_f}
        ]
    )
    print(solution)
    #print(np.linalg.norm(solution.x))
    print()
    return solution.x, solution.success


def SCA(V):
    print('Normalizing input vectors...')
    V = np.array([
        v/np.linalg.norm(v)
        for v in V
    ])
    print('done.')

    # Calculate the first loading vector. We can derive this
    # analytically using Lagrange multipliers (see documentation)
    b_init = np.sum(V, axis=0) / np.linalg.norm(np.sum(V, axis=0))
    #b_init, status = solve_init(V)
    status = True

    # Initialize the collection of loading vectors
    B = [b_init] 

    # For each loading vector, store whether the optimizer terminated
    # sucessfully
    statuses = [status]

    # Iterate through all dimensions and compute each loading vector
    for i in range(1, len(V[0])):
        print(f'Solving dimension {i}...')
        new_b, status = solve_dim(
            V, 
            np.array(B)
        )
        statuses.append(status)
        B.append(new_b)
    return np.array(B), statuses


def main():
    E = np.array([
        [1.,2.,3.,4.,1.],
        [2.,4.,3.,4.,9.],
        [1.,2.,2.,1.,8.],
        [1.,2.5,2.,1.,8.4]
    ])
    BETA_FIXED = np.array([
        #[0.,0.,0.,1.],
        [0.,0.,1.,0.,0.]
    ])

    sca, statuses = SCA(E)
    print(statuses)
    for i1, x1 in enumerate(sca):
        for i2, x2 in enumerate(sca):
            print('element {},{}: {}'.format(i1, i2, np.dot(x1, x2)))

    print('------------')
    print(sca[0])
    E2 = np.power(E, 2)
    beta_init = np.sum(E2, axis=0) / np.linalg.norm(np.sum(E2, axis=0))
    print(beta_init)
    print('Their dot is ', np.dot(sca[0], beta_init))
    E_norm = np.array([e / np.linalg.norm(e) for e in E])
    print(np.mean([np.dot(x, sca[0]) for x in E_norm]))
    print(np.mean([np.dot(x, beta_init) for x in E_norm]))
    print('------------')

    angle_distrs = []

    E_norm = np.array([e / np.linalg.norm(e) for e in E])
    for beta_i, beta in enumerate(sca):
        angles = [np.dot(x, beta)**2 for x in E_norm]
        angle_distrs.append(angles)
    angle_distrs = np.array(angle_distrs)
    angle_means = np.mean(angle_distrs, axis=1)
    print(angle_means)

    print('---------------------------------------------------------')

    import pandas as pd
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    dir_vecs_rand = np.random.rand(5, 3) #- 0.5
    dir_vecs_rand = np.array([x / np.linalg.norm(x) for x in dir_vecs_rand])
    print(dir_vecs_rand.shape)
    sca_rand, solver_statuses =  SCA(
        dir_vecs_rand 
    )
    print(solver_statuses)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    print(dir_vecs_rand)
    print(dir_vecs_rand[:,0])
    ax.quiver(np.zeros(5), np.zeros(5), np.zeros(5), dir_vecs_rand[:,0], dir_vecs_rand[:,1], dir_vecs_rand[:,2], length=1, normalize=True)
    ax.quiver(np.zeros(3), np.zeros(3), np.zeros(3), sca_rand[:,0], sca_rand[:,1], sca_rand[:,2], color='red', length=1, normalize=True)
    ax.set_xlim((0,2))
    ax.set_ylim((0,2))
    ax.set_zlim((0,2))
    plt.show()


if __name__ == '__main__':
    main()
