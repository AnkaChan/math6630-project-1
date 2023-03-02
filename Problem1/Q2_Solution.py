import pypardiso

from Utility import *
from matplotlib import pyplot as plt

def solve(numSteps,):
    # ks = k(0.1)
    # print("k(0)=", ks)

    xRange = [0, 2]

    h = (xRange[1] - xRange[0]) / (numSteps + 2)
    xs_withBoundary = np.linspace(xRange[0], xRange[1], num=numSteps + 2, endpoint=True)
    xs = xs_withBoundary[1:-1]
    # print("h:", h, "\nxs:", xs)

    A = operatorMat(numSteps, xs, h=h)
    # print("A", A.toarray())

    u_ref = u1(xs)
    f_ref = analyticF_u1(xs)
    f_numerical = A @ u_ref
    # print("f_ref:", f_ref)

    u_sol = pypardiso.spsolve(sparse.csr_matrix(A), f_ref)

    return xs, u_ref, f_ref, f_numerical, u_sol



if __name__ == '__main__':
    plt.tight_layout()

    xs, u_ref, f_ref, f_numerical, u_sol = solve(100)
    fig, axs = plt.subplots(1, 4)

    axs[0].title.set_text('u(x)\n groundtruth')
    axs[1].title.set_text('f \n(analytical)')
    axs[2].title.set_text('f \n(numerical, obtained by A @ u)')
    axs[3].title.set_text('Solution')

    axs[0].plot(xs, u_ref)
    axs[1].plot(xs, f_ref)
    axs[2].plot(xs, f_numerical)
    axs[3].plot(xs, u_sol)
    plt.waitforbuttonpress()

    number = 8
    stepMul = 2
    numOfSteps = []
    for i in range(10):
        numOfSteps.append(number)

        number = number * stepMul

    errs = []
    for numStep in numOfSteps:
        xs, u_ref, f_ref, f_numerical, u_sol  = solve(numStep)

        err = np.linalg.norm(u_ref-u_sol)
        errs.append(err)



        # for i in range(4):
        #     axs[i].cla()
        # axs[0].plot(xs, u_ref)
        # axs[1].plot(xs, f_ref)
        # axs[2].plot(xs, f_numerical)
        # axs[3].plot(xs, u_sol)
        # plt.waitforbuttonpress(100)

    fig2, axs2 = plt.subplots(1, 1)

    axs2.plot(numOfSteps, errs,)
    axs2.set_yscale('log')
    axs2.set_xscale('log')
    axs2.title.set_text('errors')
    axs2.set(xlabel='Number of steps', ylabel='Errorl')


    plt.waitforbuttonpress()
