import autograd.numpy as np
from autograd import grad
from autograd.numpy import sqrt
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import fsolve


def orth_constraint_closure(B):
    """
    A closure for creating the function encoding the orthogonal 
    constraint in the Lagrangian.

    Parameters
    ----------
    B: array
        DxN array of loading vectors where D are the first D
        loading vectors and N is the dimensionality of the word
        vector space.

    Returns
    -------
    orth_constraint_f: function
        A function encoding the orthogonality constraint
    """
    def orth_constraint_f(b):
        # b is the candidate basis vector
        return np.dot(B, b)
    return orth_constraint_f


def unit_constraint_f(b):
    """
    A function for encoding the unit-norm constraint in the 
    Lagrangian.
    
    Parameters
    ----------
    b: array
        A D-length array encoding a candidate loading vector

    Returns
    -------
    The deviation from the unit-norm constraint
    """
    return np.dot(b, b) - 1


def grad_Lagrangian_closure(V, beta_curr, squared_obj=False):
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
        if squared_obj:
            dbeta = ((2/len(V)) * np.dot(V.T, np.dot(V, beta)) \
                - np.dot(beta_curr.T, _lam_orth) \
                - (_lam_unit * 2 * beta)) 
        else:
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


def solve_dim(V, B, squared_obj=False, verbose=False):
    """
    Compute the next loading vector.

    V: array
        The collection of normalized vectors that represent
        the word-pairs.
    B: array
        DxN array of loading vectors where D are the first D
        loading vectors and N is the dimensionality of the word
        vector space.
    verbose: boolean
        If True, output logging messages.

    Returns
    -------
    b_new: array
        The next (D+1) loading vector
    """
    orth_constraint_f = orth_constraint_closure(B)
    
    # Compute the gradient of the Lagrangian function
    grad_lagrangian_f = grad_Lagrangian_closure(V, B, squared_obj)
    
    # Generate the function that evaluates the gradient of the 
    # Lagrangian at a given input
    lagrangian_wrapper_f = solve_Lagrangian_closure(
        grad_lagrangian_f, 
        orth_constraint_f, 
        len(B[0]),  # Number of dimensions
        len(B)      # Number of loadings
    )

    # Create an initial guess for solver. We'll just make it a vector
    # of all 1's
    init = np.concatenate([np.array([1.]), np.ones(len(V[0]) + len(B))])

    # Compute the roots of the gradient of the Lagrangian. These will
    # maximize the objective function subject to the orthonormal 
    # constraints.
    solution, infodict, ier, msg = fsolve(lagrangian_wrapper_f, init, full_output=1)

    if verbose: 
        print(f"Status: {ier}. Message: {msg}")

    return solution[:len(B[0])]


def _solve_first_dim_squared_obj(V):
    def obj_closure(V):
        def obj(b):
            return -1 * np.sum(np.dot(V, b))**2
        return obj

    obj_f = obj_closure(V)
    print("Solving first dimension...")
    solution = minimize(
        obj_f,
        np.concatenate([np.array([1.]), np.zeros(len(V[0])-1)]),
        options={
            'maxiter': 10000,
            'ftol': 0.0001
        },
        constraints=[
            {
                'type': 'eq', 
                'fun': unit_constraint_f
            }
        ]
    )
    return solution.x, solution.success


class SCA:
    """
    V: array
        The collection of vectors that represent the word-pairs.
    n_components: int
        Number of dimensions to compute.
    squared_obj: boolean
        If True, maximize the sum of squared distances of each 
        word-vector to the next loadings vector.

    Attributes
    -------
    components_: array
        A n_components x D array of loading vectors where D is the 
        diemsnionality of the word embedding space
    """

    def __init__(self, n_components=2, squared_obj=False):
        self.n_components = n_components
        self.squared_obj = squared_obj
        self.components_ = None

    def fit(self, V):
        print('Normalizing input vectors...')
        V = np.array([
            v/np.linalg.norm(v)
            for v in V
        ])
        print('done.')

        # Calculate the first loading vector. We can derive this
        # analytically using Lagrange multipliers (see documentation)
        if self.squared_obj:
            print("Solving dimension 0...")
            b_init, status = _solve_first_dim_squared_obj(V)
            print("Optimization status: ", status)
        else:
            b_init = np.sum(V, axis=0) / np.linalg.norm(np.sum(V, axis=0))

        # Initialize the collection of loading vectors
        B = [b_init] 

        # Iterate through all dimensions and compute each loading vector
        for i in range(1, self.n_components):
            print(f'Solving dimension {i}...')
            new_b = solve_dim(V, np.array(B), squared_obj=self.squared_obj)
            B.append(new_b)

        self.components_ = np.array(B)

    def transform(self, V):
        return np.dot(self.components_, V.T).T


if __name__ == '__main__':
    main()
