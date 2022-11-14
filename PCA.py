from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    # Your implementation goes here!
    x = np.load(filename)
    x = x - np.mean(x, axis=0)
    # print(len(x))
    # print(len(x[0]))
    # print(np.average(x))
    return x

def get_covariance(dataset):
    # Your implementation goes here!
    x = np.transpose(dataset)
    x = np.dot(x, dataset) / (len(dataset) - 1)
    return x

def get_eig(S, m):
    # Your implementation goes here!
    x, y = eigh(S, subset_by_index=[len(S)-m, len(S)-1])
    i = np.flip(np.argsort(x))
    return np.diag(x[i]), y[:, i]

def get_eig_prop(S, prop):
    # Your implementation goes here!
    x, y = eigh(S)
    proportion = np.sum(x) * prop
    updatedX, updatedY = eigh(S, subset_by_value=[proportion, np.inf])
    i = np.flip(np.argsort(updatedX))
    return np.diag(updatedX[i]), updatedY[:, i]

def project_image(image, U):
    imageAlpha = np.dot(np.transpose(U), image)
    return np.dot(U, imageAlpha)

def display_image(original, projection):
    original = np.reshape(original, (32, 32))
    projection = np.reshape(projection, (32, 32))
    fig, (origAx, projAx) = plt.subplots(nrows=1, ncols=2, figsize=(9, 3))
    origAx.set_title('Original')
    projAx.set_title('Projection')
    origKey = origAx.imshow(original, aspect='equal', cmap='Greys')
    fig.colorbar(origKey, ax=origAx)
    projKey = projAx.imshow(projection, aspect='equal', cmap='Greys')
    fig.colorbar(projKey, ax=projAx)
    plt.show()

x = load_and_center_dataset('YaleB_32x32.npy')
S = get_covariance(x)
Lambda, U = get_eig(S, 2)
Lambda, U = get_eig_prop(S, 0.07)
projection = project_image(x[0], U)
display_image(x[0], projection)



