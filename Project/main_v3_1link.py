import numpy as np
import random
import csv
import copy
import time
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
from WPI_AI.Project.Python.ODE import one_link
from math import pi

def get_coords(th):
    """Return the (x, y) coordinates of the bob at angle th."""
    return 1 * np.cos(th), 1 * np.sin(th)

def animate(i):
    """Update the animation at frame i."""
    x, y = get_coords(path[i].pos)
    line.set_data([0, x], [0, y])
    circle.set_center((x, y))

class Node():

    def __init__(self, p, v):
        self.pos = p
        self.vel = v

        self.Q_value_positive_force = 0
        self.prev_Q_value_positive_force = 0
        self.Q_value_negative_force = 0
        self.prev_Q_value_negative_force = 0
        self.force_positve = 1
        self.force_negative = -1

        self.alpha = 5

    def get_Q_value(self, action):
        if action == "positive":
            return self.Q_value_positive_force
        elif action == "negative":
            return self.Q_value_negative_force

    def update_Q_value(self, update_value, direction):
        if direction == "positive":
            self.prev_Q_value_positive_force = self.Q_value_positive_force
            self.Q_value_positive_force = update_value
        elif direction == "negative":
            self.prev_Q_value_negative_force = self.Q_value_negative_force
            self.Q_value_negative_force = update_value

    def get_max_action(self):
        actions = [self.Q_value_positive_force, self.Q_value_negative_force]
        action = max(actions)
        index = actions.index(action)
        if index == 0:
            return "positive"
        if index == 1:
            return "negative"

    def get_force(self, direction):
        if direction == "positive":
            delta_Q = (self.Q_value_positive_force-self.prev_Q_value_positive_force)
            if delta_Q == 0:
                return self.force_positve
            # elif delta_Q < 0:
            #     return 0
            elif not delta_Q == 0:
                self.force_positve = self.force_positve + self.alpha * (delta_Q/self.Q_value_positive_force)
                return self.force_positve
        if direction == "negative":
            delta_Q = (self.Q_value_negative_force - self.prev_Q_value_negative_force)
            if delta_Q == 0:
                return self.force_negative
            # elif delta_Q > 0:
            #     return 0
            elif not delta_Q == 0:
                self.force_negative = self.force_negative - self.alpha * (delta_Q/self.Q_value_negative_force)
                return self.force_negative

class world:

    def __init__(self, P_lim, V_lim, step):
        self.P_lim = P_lim
        self.V_lim = V_lim
        self.step = step
        self.row_length = int(((2*P_lim)/step)+1)
        self.col_length = int(((2*V_lim)/step)+1)
        self.epsilon = 0.1

        # GRID
        self.states = [[Node((i*step) - P_lim, (j*step) - V_lim) for j in range(self.col_length)] for i in range(self.row_length)]

    def simulation_setup(self, time_step, mass, radius, inertia):
        self.time_step = time_step
        self.mass = mass
        self.radius = radius
        self.inertia = inertia

        self.sim = one_link(self.mass, self.radius, self.inertia)

    def get_co_ordinates(self, pos, vel):
        i = (pos + self.P_lim)/self.step
        j = (vel + self.V_lim)/self.step
        return int(i), int(j)

    def define_start_pos(self, pos, vel):
        i, j = self.get_co_ordinates(pos, vel)
        self.start = self.states[i][j]

    def define_target_pos(self, pos, vel):
        i, j = self.get_co_ordinates(pos, vel)
        self.target = self.states[i][j]

    def check_lim(self, pos, vel):
        print("lim check", abs(pos), self.P_lim)
        if abs(pos) >= self.P_lim or abs(vel) >= self.V_lim:

            return False
        else:
            return True

# MOTION MODEL
    def motion(self, control_force):
        self.sim.update_torq(control_force)
        sol = self.sim.ODE45(self.time_step, self.current_pos, self.current_vel)
        self.current_pos = sol.y[0][0]
        self.current_vel = sol.y[1][0]

    def policy_Epsilon_greedy(self, current_node):
        p = np.random.random()
        if p <= self.epsilon:
            return random.choice(["positive", "negative"])
        elif p > self.epsilon:
            return current_node.get_max_action()

    def Q_learning(self):
        self.alpha = 0.2
        self.gamma = 0.9

        t_init = time.time()
        t_end = time.time()
        iter = 0

        while t_end - t_init <= 160:
            print(iter)
            current_node = self.start
            self.current_pos = current_node.pos
            self.current_vel = current_node.vel
            print(current_node.pos, current_node.vel)

            # RATE OF CHANGE OF ERROR
            prev_error_p = self.target.pos - self.current_pos
            prev_error_v = self.target.vel - self.current_vel
            t_new = time.time()
            while not current_node == self.target and not current_node == None and t_end - t_init <= 160:

                action = self.policy_Epsilon_greedy(current_node)
                print(action)
                p = np.random.random()
                if p <= self.epsilon:
                    if action == "positive":
                        force = 7
                    elif action == "negative":
                        force = -7
                elif p > self.epsilon:
                    force = current_node.get_force(action)

                print("force", force)
                self.motion(force)  # take action step
                # print(self.current_pos, self.current_vel)

                i, j = self.get_co_ordinates(self.current_pos, self.current_vel)
                # print(i, j)
                if abs(i) >= int((self.row_length) ) or abs(j) >= int(self.col_length) or i < 0 or j < 0:
                    reward = -5
                    Q_ns_na = 0
                    Q_s_a = current_node.get_Q_value(action)
                    new_state = None

                    # error term
                    error_p = self.target.pos - self.current_pos
                    error_v = self.target.vel - self.current_vel

                    # UPDATE STEP
                    # print(reward)
                    update_value = Q_s_a + self.alpha * (reward + (self.gamma * Q_ns_na) - Q_s_a)
                    current_node.update_Q_value(update_value, action)
                    current_node = new_state
                    prev_error_v = error_v
                    prev_error_p = error_p
                    reward = 0  # reset reward for next step
                    t_end = time.time()  # to cancel the sleep time

                else:
                    new_state = self.states[i][j]  # to do: must be within limit, will  give error if not

                    # error term
                    error_p = self.target.pos - self.current_pos
                    error_v = self.target.vel - self.current_vel

                    # REWARD
                    reward = 1 * np.sign(abs(prev_error_p) - abs(error_p)) + 0.1 * np.sign(
                        abs(prev_error_v) - abs(error_v))
                    # over shoot reward
                    if not np.sign(prev_error_p) == np.sign(error_p):
                        reward += -0

                    Q_ns_na = max([new_state.get_Q_value("positive"), new_state.get_Q_value("negative")])
                    Q_s_a = current_node.get_Q_value(action)

                    if new_state == self.target:
                        # target reach reward
                        reward += 10
                        print("____________________________DONE___________________________")
                        time.sleep(2)
                    elif reward <= 0:
                        reward += -0.1

                    # UPDATE STEP
                    # print(reward)
                    update_value = Q_s_a + self.alpha * (reward + (self.gamma * Q_ns_na) - Q_s_a)
                    print("update value", update_value)
                    current_node.update_Q_value(update_value, action)
                    current_node = new_state
                    prev_error_v = error_v
                    prev_error_p = error_p
                    reward = 0
                    t_end = time.time()  # to cancel the sleep time

                print(self.current_pos, self.current_vel)

            iter += 1
            # if iter == 100:
            #     exit(1)
            print("???????????????????????????????step?????????????????????????????")
            print(iter)
            # if iter > 50:
            #     break

        # testing loop
        path = []
        current_node = self.start
        self.current_pos = current_node.pos
        self.current_vel = current_node.vel
        print(current_node.pos, current_node.vel)

        # RATE OF CHANGE OF ERROR
        prev_error_p = self.target.pos - self.current_pos
        prev_error_v = self.target.vel - self.current_vel
        t_new = time.time()
        while t_end - t_init <= 5:
            path.append(current_node)
            action = current_node.get_max_action()
            print(action)
            force = current_node.get_force(action)

            print("force", force)
            self.motion(force)  # take action step
            # print(self.current_pos, self.current_vel)

            i, j = self.get_co_ordinates(self.current_pos, self.current_vel)
            # print(i, j)
            if abs(i) >= int((self.row_length) ) or abs(j) >= int(self.col_length) or i < 0 or j < 0:
                reward = -10
                Q_ns_na = 0
                Q_s_a = current_node.get_Q_value(action)
                new_state = None

                # error term
                error_p = self.target.pos - self.current_pos
                error_v = self.target.vel - self.current_vel

                # UPDATE STEP
                # print(reward)
                update_value = Q_s_a + self.alpha * (reward + (self.gamma * Q_ns_na) - Q_s_a)
                current_node.update_Q_value(update_value, action)
                current_node = new_state
                prev_error_v = error_v
                prev_error_p = error_p
                reward = 0  # reset reward for next step
                t_end = time.time()  # to cancel the sleep time

            else:
                new_state = self.states[i][j]  # to do: must be within limit, will  give error if not

                # error term
                error_p = self.target.pos - self.current_pos
                error_v = self.target.vel - self.current_vel

                # REWARD
                reward = 1 * np.sign(abs(prev_error_p) - abs(error_p)) + 0.1 * np.sign(
                    abs(prev_error_v) - abs(error_v))
                if not np.sign(prev_error_p) == np.sign(error_p):
                    reward += -2
                Q_ns_na = max([new_state.get_Q_value("positive"), new_state.get_Q_value("negative")])
                Q_s_a = current_node.get_Q_value(action)

                if new_state == self.target:
                    reward += 5
                    print("____________________________DONE___________________________")
                    # time.sleep(2)
                elif reward <= 0:
                    reward += -0.1

                # UPDATE STEP
                # print(reward)
                update_value = Q_s_a + self.alpha * (reward + (self.gamma * Q_ns_na) - Q_s_a)
                print("update value", update_value)
                current_node.update_Q_value(update_value, action)
                current_node = new_state
                prev_error_v = error_v
                prev_error_p = error_p
                reward = 0
                t_end = time.time()  # to cancel the sleep time

            print(self.current_pos, self.current_vel)

        iter += 1
        # if iter == 100:
        #     exit(1)
        print("???????????????????????????????step?????????????????????????????")
        print(iter)

        # print(len(path))
        # y = []
        # for i in range(len(path)):
        #     y.append(i)


        # plt.scatter(y, path)
        # plt.show()
        return path

    def learning(self):
        self.alpha = 0.2
        self.gamma = 0.9

        t_init = time.time()
        t_end = time.time()
        iter = 0
        while t_end - t_init <= 30:
            print(iter)
            current_node = self.start
            self.current_pos = current_node.pos
            self.current_vel = current_node.vel
            print(current_node.pos, current_node.vel)

            # RATE OF CHANGE OF ERROR
            prev_error_p = self.target.pos - self.current_pos
            prev_error_v = self.target.vel - self.current_vel
            t_new = time.time()
            while not current_node == self.target and not current_node == None and t_end - t_init <= 30:
                # print(current_node.pos)
                # print(current_node == self.target )
                action = self.policy_Epsilon_greedy(current_node)
                print(action)
                force = current_node.get_force(action)

                print("force",force)
                self.motion(force)  # take action step
                print(self.current_pos, self.current_vel)
                # exit(1)

                # AGENT MUST BE INSIDE THE POS AND VEL LIMIT
                i, j = self.get_co_ordinates(self.current_pos, self.current_vel)

                if not self.check_lim(self.current_pos, self.current_vel):
                    reward = -10
                    Q_ns_na = 0
                    Q_s_a = current_node.get_Q_value(action)
                    new_state = None

                elif abs(i) <= int((self.row_length - 1) / 2) or abs(j) >= int((self.col_length - 1) / 2):
                    reward = -10
                    Q_ns_na = 0
                    Q_s_a = current_node.get_Q_value(action)
                    new_state = None

                elif self.check_lim(self.current_pos, self.current_vel):
                    i, j = self.get_co_ordinates(self.current_pos, self.current_vel)
                    new_state = self.states[i][j]   # to do: must be within limit, will  give error if not

                # error term
                error_p = self.target.pos - self.current_pos
                error_v = self.target.vel - self.current_vel
                # if prev_error_p-error_p > 0;
                #     reward = 0.5
                # REWARD
                reward = 1*np.sign(abs(prev_error_p) - abs(error_p)) + 0.1*np.sign(abs(prev_error_v) - abs(error_v))
                if not np.sign(prev_error_p) == np.sign(error_p):
                    reward += -10
                Q_ns_na = max([new_state.get_Q_value("positive"), new_state.get_Q_value("negative")])
                Q_s_a = current_node.get_Q_value(action)

                if new_state == self.target:
                    reward += 5
                    print("____________________________DONE___________________________")
                    # time.sleep(3)
                else:
                    reward += -0.1

                # UPDATE STEP
                # reward = -reward
                print(reward)
                update_value = Q_s_a + self.alpha*(reward + (self.gamma*Q_ns_na) - Q_s_a)
                current_node.update_Q_value(update_value, action)
                current_node = new_state
                prev_error_v = error_v
                prev_error_p = error_p
                reward = 0
                t_end = time.time()  # to cancel the sleep time
                # if (time.time() - t_init) > 1:
                #     exit(1)
            print(self.current_pos, self.current_vel)

            iter += 1

        print("DONE")

        # TESTING LOOP
        # time.sleep(5)
        # path = []
        # current_node = self.start
        # self.current_pos = self.start.pos
        # self.current_vel = self.start.vel
        # while not current_node == self.target:
        #     path.append(current_node.pos)
        #     action = current_node.get_max_action()
        #     print(action)
        #     force = current_node.get_force(action)
        #
        #     # print(force)
        #     self.motion(force)  # take action step
        #     print(self.current_pos, self.current_vel)
        #     # exit(1)
        #     if not self.check_lim(self.current_pos, self.current_vel):
        #         reward = -10
        #         Q_ns_na = 0
        #         Q_s_a = current_node.get_Q_value(action)
        #         new_state = None
        #
        #     elif self.check_lim(self.current_pos, self.current_vel):
        #         i, j = self.get_co_ordinates(self.current_pos, self.current_vel)
        #         new_state = self.states[i][j]  # to do: must be within limit, will  give error if not
        #         error_p = self.target.pos - self.current_pos
        #         error_v = self.target.vel - self.current_vel
        #         # if prev_error_p-error_p > 0;
        #         #     reward = 0.5
        #         reward = 1 * np.sign(abs(prev_error_p) - abs(error_p)) + 0.1 * np.sign(abs(prev_error_v) - abs(error_v))
        #         if not np.sign(prev_error_p) == np.sign(error_p):
        #             reward += -10
        #         Q_ns_na = max([new_state.get_Q_value("positive"), new_state.get_Q_value("negative")])
        #         Q_s_a = current_node.get_Q_value(action)
        #
        #         if new_state == self.target:
        #             reward += 5
        #             print("____________________________DONE___________________________")
        #             time.sleep(3)
        #             break
        #         else:
        #             reward += -0.1
        #
        #     # print(reward)
        #     update_value = Q_s_a + self.alpha * (reward + (self.gamma * Q_ns_na) - Q_s_a)
        #     # print("new", update_value)
        #     current_node.update_Q_value(update_value, action)
        #     # print(current_node.Q_value_positive_force, current_node.Q_value_negative_force)
        #     current_node = new_state
        #     prev_error_v = error_v
        #     prev_error_p = error_p
        #     reward = 0
        #
        # print(len(path))
        # y = []
        # for i in range(len(path)):
        #     y.append(i)
        # plt.scatter(y, path)
        # plt.show()


if __name__ == '__main__':
    sim_world = world(pi, 5, 0.05)
    sim_world.simulation_setup(0.05, 0.8, 0.5, 0.5)
    sim_world.define_start_pos(0.5, 0)
    sim_world.define_target_pos(pi/2, 0)
    path = sim_world.Q_learning()

    L = 1
    # Initialize the animation plot. Make the aspect ratio equal so it looks right.
    fig = plt.figure()
    ax = fig.add_subplot(aspect='equal')
    theta0 = path[0].pos
    # # The pendulum rod, in its initial position.
    x0, y0 = get_coords(theta0)
    line, = ax.plot([0, x0], [0, y0], lw=3, c='k')
    # The pendulum bob: set zorder so that it is drawn over the pendulum rod.
    bob_radius = 0.08
    circle = ax.add_patch(plt.Circle(get_coords(theta0), bob_radius,
                                     fc='r', zorder=3))
    # Set the plot limits so that the pendulum has room to swing!
    ax.set_xlim(-L * 1.2, L * 1.2)
    ax.set_ylim(-L * 1.2, L * 1.2)

    # print(len(sol.y[0]))
    nframes = len(path)
    interval = 1
    ani = animation.FuncAnimation(fig, animate, frames=nframes, repeat=True,
                                  interval=interval)
    # # ani.save("ani")
    # ani.save('the_movie.', writer='mencoder', fps=15)
    plt.show()

