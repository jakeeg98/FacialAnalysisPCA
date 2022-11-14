from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

#load dataset and center it around the origin.
def load_and_center_dataset(filename):
    x = np.load(filename)
    x = x - np.mean(x, axis=0)
    return x

#get the covariance matrix
def get_covariance(dataset):
    return (np.dot(np.transpose(dataset), dataset) / (len(dataset) - 1))

#get the 'm' largest eigenvalues and eigenvectors
def get_eig(S, m):
    w, v = eigh(S, subset_by_index=[len(S)-m, len(S)-1])
    i = np.flip(np.argsort(w))
    return np.diag(w[i]), v[:, i]

#get the eigenvalues and corresponding eigenvectors that explain more than a certain percentage of variance
def get_eig_prop(S, prop):
    w, v = eigh(S)
    percent = np.sum(w) * prop
    new_w, new_v = eigh(S, subset_by_value=[percent, np.inf])
    i = np.flip(np.argsort(new_w))
    return np.diag(new_w[i]), new_v[:, i]

#project the images
def project_image(image, U):
    alphas = np.dot(np.transpose(U), image)
    return np.dot(U, alphas)

#display images with original next to projection
def display_image(orig, proj):
    orig = np.reshape(orig, (32,32))
    proj = np.reshape(proj, (32,32))

    fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (9,3))
    ax1.set_title('Original')
    ax2.set_title('Projection')

    ax1Map = ax1.imshow(orig, aspect = 'equal', cmap='gray')
    fig.colorbar(ax1Map, ax=ax1)
    ax2Map = ax2.imshow(proj, aspect = 'equal', cmap='gray')
    fig.colorbar(ax2Map, ax=ax2)
    plt.show()



x = load_and_center_dataset('YaleB_32x32.npy')
S = get_covariance(x)
Lambda, U = get_eig(S, 2)
Lambda, U = get_eig_prop(S, 0.07)
projection = project_image(x[0], U)
display_image(x[0], projection)
