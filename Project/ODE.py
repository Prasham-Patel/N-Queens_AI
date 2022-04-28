import numpy as np
import scipy
from scipy.integrate import solve_ivp
from math import cos, pi
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class one_link:

    def __init__(self, m, r, I):
        self.m = m
        self.r = r
        self.I = I
        self.g = -9.81
        self.torq = 0

    def update_torq(self, T):
        self.torq = T

    def get_next_state(self, t, y):
        theta = y[0]
        theta_dot = y[1]
        theta_ddot = (self.torq - (self.m*self.r*(self.g)*cos(theta)))/(self.I + self.m*self.r)
        return [theta_dot, theta_ddot]

    def ODE45(self, tf, theta, theta_dot):
        t_eval = [tf]
        # t_eval = np.linspace(0, 5, num=500)
        sol = solve_ivp(self.get_next_state, [0, tf], [theta, theta_dot], t_eval= t_eval, method='RK45', dense_output=True)
        return sol

# def get_coords(th):
#     """Return the (x, y) coordinates of the bob at angle th."""
#     return 1 * np.cos(th), 1 * np.sin(th)
#
# def animate(i):
#     """Update the animation at frame i."""
#     x, y = get_coords(sol.y[0][i])
#     line.set_data([0, x], [0, y])
#     circle.set_center((x, y))


if __name__ == '__main__':
    print(cos(-2))
    sim = one_link(1, 0.5, 0.5)
    sol = sim.ODE45(5, -2, 0)
    print(sol.y[0])



    #
    # L=  1
    # # Initialize the animation plot. Make the aspect ratio equal so it looks right.
    # fig = plt.figure()
    # ax = fig.add_subplot(aspect='equal')
    # theta0 = sol.y[0][0]
    # # The pendulum rod, in its initial position.
    # x0, y0 = get_coords(theta0)
    # line, = ax.plot([0, x0], [0, y0], lw=3, c='k')
    # # The pendulum bob: set zorder so that it is drawn over the pendulum rod.
    # bob_radius = 0.08
    # circle = ax.add_patch(plt.Circle(get_coords(theta0), bob_radius,
    #                                  fc='r', zorder=3))
    # # Set the plot limits so that the pendulum has room to swing!
    # ax.set_xlim(-L * 1.2, L * 1.2)
    # ax.set_ylim(-L * 1.2, L * 1.2)
    #
    #
    #
    # print(len(sol.y[0]))
    # nframes = 500
    # interval = 10
    # ani = animation.FuncAnimation(fig, animate, frames=nframes, repeat=True,
    #                               interval=interval)
    # # ani.save("ani")
    # ani.save('the_movie.', writer='mencoder', fps=15)
    # plt.show()