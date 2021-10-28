import numpy as np
from scipy import sparse
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import CG

def getGradientMatrix(nx, ny):

    e, e_neg = -np.ones(nx), np.ones(nx)
    #e_neg[-1] = 0
    Dx1D = sparse.spdiags([e, e_neg], [0, 1], nx-1, nx)
    e, e_neg = -np.ones(ny), np.ones(ny)
    #e_neg[-1] = 0
    Dy1D = sparse.spdiags([e, e_neg], [0, 1], ny-1, ny)
    Dx = sparse.kron(sparse.eye(ny), Dx1D)
    Dy = sparse.kron(Dy1D, sparse.eye(nx))
    D = sparse.vstack([Dx, Dy])
    return D, Dx, Dy


def getWeightedLap(Sigma):

    Sigma = Sigma.numpy()
    nx, ny = Sigma.shape
    SigmaX = (Sigma[1:,:] + Sigma[:-1,:])/2
    SigmaY = (Sigma[:,1:] + Sigma[:,:-1])/2
    SigmaF = np.hstack((SigmaX.reshape((nx-1)*ny),SigmaY.reshape((ny-1)*nx)))
    D, Dx, Dy = getGradientMatrix(nx, ny)
    SigmaMat = sparse.spdiags(SigmaF, 0, (nx-1)*ny + (ny-1)*nx, (nx-1)*ny + (ny-1)*nx )

    A = D.T @ SigmaMat @ D

    # Convert to pytorch
    As = sparse.coo_matrix(A, dtype=sparse.coo_matrix)

    values = As.data
    indices = np.vstack((As.row, As.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values.tolist())
    sh = As.shape

    Atrch = torch.sparse.FloatTensor(i, v, torch.Size(sh))

    return Atrch



class diffusionReaction(nn.Module):
    def __init__(self, nx, ny, nt, h, dt, nchan, nopen):
        super(diffusionReaction, self).__init__()
        # Setup the Laplacian
        self.L  = (1/h**2)*getWeightedLap(torch.ones(nx,ny))
        self.nt = nt
        self.dt = dt

        self.scale = nn.Parameter(torch.rand(1,nchan))
        self.K1 = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(nchan, nopen)))
        self.K2 = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(nopen, nopen)))
        self.K3 = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(nopen, nchan)))
        self.scale = nn.Parameter(torch.tensor([0.001, 0.01, 0.1, 1]))


    def forward(self, U0):

        nx, nchan = U0.shape
        U = U0.clone()
        for j in range(self.nt):
            # Diffusion
            for i in range(nchan):
                gamma = 1/(self.scale[i]*self.dt)
                U[:,i] = CG.pcg(self.L, gamma*U[:,i], gamma)

            # Reaction
            U = U + self.dt*torch.tanh(torch.tanh(torch.tanh(U@self.K1)@self.K2)@self.K3)

        return U



nx = 64
ny = 64
nt = 512
h  = 1/16
dt = 1/64
nchan = 4
nopen = 16

DRnet = diffusionReaction(nx, ny, nt, h, dt, nchan, nopen)

# Generate a 4 channels random data
U0 = torch.randn(nx*ny, 4)
Uout = DRnet(U0)

plt.figure(1)
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(U0[:, i].reshape(nx, ny).detach())
    plt.colorbar()

plt.figure(2)
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(Uout[:,i].reshape(nx,ny).detach())
    plt.colorbar()