from turtle import position
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.mixture import GMM
from matplotlib.patches import Ellipse
import casadi

def reject_outliers_offline(data, m=3):
    #print(abs(data - np.mean(data)) < m * np.std(data))
    filtered = []
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    for pt in data:
        if all(np.abs(pt - mean) < m * std):
            filtered.append(pt)
    filtered = np.array(filtered)
    return filtered

def analyze(position_data, window_size=10):
    eps = 1e-7 # how small is zero? To reject xy points when we didn't find the bot for a frame
    vector_acute_angle = lambda v1, v2: np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    avg_phis = []
    avg_speeds = []
    buffer = []
    for pos in position_data:
        if abs(pos[0]) < eps and abs(pos[1]) < eps:
            continue 
        buffer.append(pos)

        if len(buffer) >= window_size:
            diffs = [buffer[i] - buffer[i-1] for i in range(1, len(buffer))]
            #phis = [vector_acute_angle(diffs[i], diffs[i - 1]) for i in range(1, len(diffs))]
            #avg_phi = np.mean([vector_acute_angle(diffs[i], diffs[i - 1]) for i in range(1, len(diffs))])
            avg_phi = vector_acute_angle(buffer[4] - buffer[0], buffer[9] - buffer[4])
            avg_phi /= window_size            
            avg_speed = np.mean([np.linalg.norm(diff) for diff in diffs])

            buffer.pop(0)
            if not (np.isnan(avg_speed) or np.isnan(avg_phi)):
                avg_phis.append(avg_phi)
                avg_speeds.append(avg_speed)
                
    return np.vstack((avg_phis, avg_speeds)).T

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

def determine_phi_v_primitives(position_data):
    X = analyze(position_data)
    gmm = GMM(n_components=1, covariance_type='full')
    gmm.fit(X)
    #(phi1, v1), (phi2, v2) = gmm.means_
    (phi, v) = gmm.means_[0]

    return (phi, v) # ((phi1, v1), (phi2, v2))

def plot_pos_and_phi_v_clusters(position_data):
    fig, ax = plt.subplots(2)

    ax[0].scatter(position_data[:,0], position_data[:,1])
    ax[0].set_title("Image Space Position Data Points")
    ax[0].set_ylabel("Y Pixel Position")
    ax[0].set_xlabel("X Pixel Position")

    X = analyze(position_data)

    gmm = GMM(n_components=1, covariance_type='full',random_state=32)
    ax[1].set_xlim([-1, 5])
    labels = gmm.fit(X).predict(X)
    ax[1].scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    ax[1].set_title("Image Space Dynamics Data Points")
    ax[1].set_ylabel("Velocity Magnitudes (pixels/timestep)")
    ax[1].set_xlabel("Turning Angle (radians)")
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covars_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)

    plt.show()

# def curvature_fit(position_data):
#     featurized_data = []
#     for x, y in position_data:
#         features = [x**2 + y**2, x, y]
#         featurized_data.append(features)
    
#     A = np.array(featurized_data)
#     b = np.ones(A.shape[0])
#     opti = casadi.Opti()

#     x = opti.variable(3, 1)
#     obj = casadi.norm_2(casadi.mtimes(A, x) - b)

#     opti.minimize(obj)

#     opti.subject_to([x[0] > 0])

#     p_opts = {"expand": False}
#     s_opts = {"max_iter": 1e4}

#     opti.solver('ipopt', p_opts, s_opts)
#     sol = opti.solve()

#     print(sol.value(x))

#     # a, b, c = np.linalg.lstsq(featurized_data, np.ones(featurized_data.shape[0]))[0]
#     # print("values are ", a, b, c)
#     # a
#     # if a == 0: 
#     #     return 0
#     # else:
#     #     rsq = (1-d)/a + (b**2 + c**2)/(2 * a**2)
      #     r = pow(rsq, 1.0/2)
      #     return 1/r

if __name__ == "__main__": 
    position_data = np.loadtxt("../data/backward_may_4.txt")
    #curvature_fit(position_data)
    print(determine_phi_v_primitives(position_data))
    plot_pos_and_phi_v_clusters(position_data)