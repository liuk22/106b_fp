from turtle import position
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.mixture import GMM
from matplotlib.patches import Ellipse

position_data = np.loadtxt("../data/position_data_2.txt")

def reject_outliers(data, m=10):
    #print(abs(data - np.mean(data)) < m * np.std(data))
    return data[abs(data - np.mean(data)) < m * np.std(data)]


fig, axs = plt.subplots(2)
W = (0, 1000)
#axs[0].set_xlim([0, 8])
axs[0].scatter(position_data[W[0] : W[1], 0], position_data[W[0] : W[1], 1])

def analyze(position_data, window_size=5):

    def vector_acute_angle(v1, v2): 
        return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    avg_delta_thetas = []
    avg_speeds = []
    buffer = []
    for pos in position_data:
        buffer.append(pos)

        if len(buffer) >= window_size:
            diffs = [buffer[i] - buffer[i-1] for i in range(1, len(buffer))]
            delta_thetas = [vector_acute_angle(diffs[i], diffs[i - 1]) for i in range(1, len(diffs))]
            avg_delta_theta = np.mean(delta_thetas)
            
            avg_speed = np.mean([np.linalg.norm(diff) for diff in diffs])

            buffer.pop(0)

            if np.isnan(avg_speed) or np.isnan(avg_delta_theta):
                continue
            else:
                avg_delta_thetas.append(avg_delta_theta)
                avg_speeds.append(avg_speed)
                
    return avg_delta_thetas, avg_speeds

# helper functions from https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html
def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
        
def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covars_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)


avg_delta_thetas, avg_speeds = analyze(position_data, 5)

X = np.vstack((avg_delta_thetas, avg_speeds)).T

gmm = GMM(n_components=2, covariance_type='full')

axs[1].set_xlim([-1, 5])
plot_gmm(gmm, X, ax=axs[1])

plt.show()