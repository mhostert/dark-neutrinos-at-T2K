import numpy as np
from scipy.stats import norm, uniform, multivariate_normal

def log_distance(x1, x2):
    return np.sqrt((np.log10(x1/x2)**2).sum(axis=-1))

def epa_kernel2d(x, sigma):
    return np.where((x[..., 0]/sigma[0])**2 + (x[..., 1]/sigma[1])**2 > 1,
                    0,
                    (1 - (x[..., 0]/sigma[0])**2 - (x[..., 1]/sigma[1])**2)/(4*sigma[0]*sigma[1]*(1-1/np.sqrt(3))))

def kde_Nd_weights(x, x_i, smoothing, distance='log', kernel='epa', ball_tree=None):
        """x is where to evaluate (could be a grid), x_i are the montecarlo points for training"""
        assert x.shape[-1] == x_i.shape[-1] #number of dimensions
        x_i = np.expand_dims(x_i, axis=list(range(1, len(x.shape))))
        x = np.expand_dims(x, axis=0) #add the axis for the number of points over which to sum.

        if ball_tree is None:
            if distance == 'lin':
                x_dist = (x - x_i)
            elif distance == 'log':
                x_dist = np.log10(x/x_i)
            else:
                print("distance {distance} not supported, enter either 'lin' or 'log'")
                return
            
            if kernel == 'gaus':
                smoothing = np.diag(smoothing)**2
                kde_weights = multivariate_normal.pdf(x_dist, cov=smoothing)
            elif kernel == 'epa':
                kde_weights = epa_kernel2d(x_dist, smoothing)
            else:
                print("kernel {kernel} not supported, enter either 'gaus' or 'epa'")

            if distance == 'log':
                kde_weights /= (np.log(10)**(x.shape[-1]) * np.prod(x_i, axis=-1))
        else:
            assert kernel=='epa'
            assert smoothing[0] == smoothing[1]
            assert x.shape == 1
            x = x.reshape(1, -1)
            indices, distances = ball_tree.query_radius(x, smoothing[0], return_distance=True)

            kde_weights = np.zeros(x_i.shape[0])
            kde_weights[indices[0]] = (1 - (distances[0]/smoothing[0])**2)/(4*smoothing[0]**2*(1-1/np.sqrt(3)))

        return kde_weights