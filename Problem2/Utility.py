from pypardiso import spsolve
from scipy import sparse
import numpy as np
# A = sparse.csr_matrix(A)
# KKTRes = sparse.csr_matrix(KKTRes)
# # print("Done building system")
# print("Time build KKT:", time.clock() - timeStart)
#
# # xInterpo = sparse.linalg.spsolve(A, KKTRes)
# xInterpo = spsolve(A, (KKTRes).toarray())

def k(x):
    ls = np.linspace(1,5, num=5, endpoint=True)
    val = 0
    for l in ls:
        val = val + np.sin(l * np.pi * x) / (l + 1)
    return 2 + val

def k_derivative(x):
    ls = np.linspace(1,5, num=5, endpoint=True)
    val = 0
    for l in ls:
        val = val + l*np.pi*np.cos(l*np.pi*x)/(l+1)

    return  val

def u1(x):
    return np.sin(np.pi * x)

def u1_d(x):
    return np.pi *np.cos(np.pi * x)

def u1_2d(x):
    return -np.pi * np.pi *np.sin(np.pi * x)

def analyticF_u1(x):
    kv = k(x)
    k_d = k_derivative(x)
    u_d = u1_d(x)
    u_2d = u1_2d(x)

    return kv * u_2d + k_d * u_d

def operatorMat(M, xs, h):
    A = sparse.lil_matrix((M, M))

    # A[0, 0] = k(xs[0]+0.5*h) - k(xs[0]-0.5*h)
    # A[0, 1] = k(xs[0]+0.5*h)
    # A[M-1, M-1] = k(xs[M-1]+0.5*h) - k(xs[M-1]-0.5*h)
    # A[M-1, M-2] = - k(xs[M-1]-0.5*h)
    # for i in range(1, M-1):
    for i in range(0, M):
        xi = xs[i]
        A[i, i] = (-k(xi+0.5*h) - k(xi-0.5*h)) / (h*h)
        if i < M-1:
            A[i, i+1] = k(xi+0.5*h)/ (h*h)
        if i >0:
            A[i, i-1] = k(xi-0.5*h) / (h*h)
        A = A
    return A

def makeTemporalLaplacainBlock(dim):
    TL = sparse.lil_matrix((3 * dim, 3 * dim))
    I = sparse.eye(dim)
    TL[:dim, :dim] = I
    TL[dim:2 * dim, :dim] = -2 * I
    TL[2 * dim:3 * dim, :dim] = I

    TL[:dim, dim:2 * dim] = -2 * I
    TL[dim:2 * dim, dim:2 * dim] = 4 * I
    TL[2 * dim:3 * dim, dim:2 * dim] = -2 * I

    TL[:dim, 2 * dim:3 * dim] = I
    TL[dim:2 * dim, 2 * dim:3 * dim] = -2 * I
    TL[2 * dim:3 * dim, 2 * dim:3 * dim] = I