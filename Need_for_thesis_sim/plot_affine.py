import numpy as np
import matplotlib.pyplot as plt

def apply_affine(tx, ty, angle, shear, sx, sy, nps_t):
    nps = nps_t.copy()
    # translation
    nps[:,0] += tx
    nps[:,1] += ty
    # rotation
    m = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    for i in range(len(nps)):
        nps[i] = np.dot(m,nps[i])
    # shear
    m = [[1, shear], [0, 1]]
    for i in range(len(nps)):
        nps[i] = np.dot(m,nps[i])
    # scale
    for i in range(len(nps)):
        nps[i][0] *= sx
        nps[i][1] *= sy
    return nps

def find_affine(nps_t):
    # translation
    nps = nps_t.copy()
    tx = -nps[0][0]
    ty = -nps[0][1]

    plot_points(nps, 'input')

    nps[:,0] += tx
    nps[:,1] += ty

    plot_points(nps, 'after_trans')

    # rotation
    if nps[1][1] != nps[3][1]:
        angle = np.arctan(abs(nps[3][1]) / abs(nps[3][0]))
    else:
        angle = 0
    if nps[3][1] > 0:
        angle = -angle

    m = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    for i in range(len(nps)):
        nps[i] = np.dot(m,nps[i])

    plot_points(nps, 'after_rot')

    # shear
    if nps[4][0] != nps[2][0]:
        shear = (nps[4][0] - nps[2][0]) / (nps[2][1] - nps[4][1])
    else:
        shear = 0

    m = [[1, shear], [0, 1]]
    for i in range(len(nps)):
        nps[i] = np.dot(m,nps[i])

    plot_points(nps, 'after_shear')

    # scale
    sx = np.abs(nps[3][0] - nps[1][0])
    sy = np.abs(nps[4][1] - nps[2][1])

    nps[:,0] *= sx
    nps[:,1] *= sy

    plot_points(nps, 'after_scale')

    return tx, ty, angle, shear, sx, sy, nps

def plot_points(nps, name):
    plt.figure(figsize=(5,5), dpi=300)
    plt.tight_layout()
    plt.plot(nps[:,0], nps[:,1], 'x', color='k', markersize=10)
    plt.xlim(-1.5,1.5)
    plt.ylim(-1.5,1.5)
    # print numbers 0,1,2,3,4 next to each point
    for i in range(5):
        offs = 0.08
        plt.text(nps[i][0]+offs, nps[i][1]+offs, 'p'+str(i), fontsize='x-large')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(name+'.png')

# make test points
nps = np.array([[0.0,0.0],[-1.0,0.0],[0.0,-1.0],[1.0,0.0],[0.0,1.0]])

nps_t = nps.copy()

#scale
# sx = 0.4
# sy = 0.6
# nps_t[:,0] *= sx
# nps_t[:,1] *= sy

# shear
shear = 0.4
m = [[1.0, shear], [0.0, 1.0]]
for i in range(len(nps_t)):
    nps_t[i] = np.dot(m,nps_t[i])

# rotate 
angle = np.pi/2.5
m = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
for i in range(len(nps_t)):
    nps_t[i] = np.dot(m,nps_t[i])

# translate
nps_t[:,0] -= 0.45
nps_t[:,1] -= 0.45

# scale
nps_t[:,0] *= 0.7
nps_t[:,1] *= 0.7

find_affine(nps_t)

# plt.show()