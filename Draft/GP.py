# Define the exponentiated quadratic kerel function
import scipy.spatial.distance
import numpy as np

def exponentiated_quadratic(xa,xb):
    """Exponentiated quadratic with \sigma=1"""
    # L2 distance (Squared Euclidian)
    sq_norm = -0.5*scipy.spatial.distance.cdist(xa,xb,'sqeuclidean')
    return np.exp(sq_norm)

# Sample from the Gaussian process distribution
nb_of_samples = 41
number_of_functions = 5
X= np.expand_dims(np.linspace(-4,4,nb_of_samples),1)
cov = exponentiated_quadratic(X,X)
print(cov)
ys = np.random.multivariate_normal(
    mean=np.zeros(nb_of_samples), cov=cov,
    size = number_of_functions
)