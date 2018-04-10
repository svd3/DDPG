import math

import gym
from gym import spaces
from gym.utils import seeding

from autograd import jacobian
import autograd.numpy as np
import autograd.numpy.random as random
from autograd.numpy.linalg import inv
from autograd.numpy import pi

from collections import namedtuple
from myenv.ellipse import ellipse

from gym.envs.classic_control import rendering

def clip(x):
    return (5. - np.log(1. + np.exp(-(x-5.))))/(1. + np.exp(-20*x)) + (-5. + np.log(1. + np.exp(x+5.)))/(1. + np.exp(20*x))

def is_pos_def(x):
    return np.all(np.linalg.eigvalsh(x) > 0)

class FireflyEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.counter = 0
        self.episode_len = 1000

        self.max_action = 1
        self.world_box = np.array([[5.0, 5.0], [-5.0, -5.0]])
        #self.min_position = np.array([-5.0, -5.0])
        self.xlow = np.append(self.world_box[1], [0., -1., -1.])
        self.xhigh = np.append(self.world_box[0], [2*pi, 1., 1.])
        self.low_state = np.append(self.xlow, -5*np.ones(15))       #
        self.high_state = np.append(self.xhigh, 5*np.ones(15))       #

        self.action_space = spaces.Box(-np.ones(2), np.ones(2))
        self.observation_space = spaces.Box(self.low_state, self.high_state)

        self.viewer = None
        #self.state = self.observation_space.sample()

        self.noise = np.array([0.01]*5 + [0.1]*2) #std
        self.dt = 0.1
        self.Q = np.eye(5) * (0.01**2)
        self.R = np.eye(2) * (0.1**2)
        self.P = np.eye(5) * 0.
        self.Id = np.eye(5)

        self.A = jacobian(self.dynamics)
        self.H = jacobian(self.obs)

        self.goal_position = np.array([0., 0.])
        self.goal_radius = 0.8

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def dynamics(self, x, action):
        dt = self.dt
        action = np.clip(action, -1., 1.)
        noise = self.noise
        px, py, angle, vel, ang_vel = x
        vel = 0.0*vel + action[0] + random.randn(1)*noise[0]
        ang_vel = 0.0*ang_vel + action[1] + random.randn(1)*noise[1]
        angle += ang_vel * dt + random.randn(1)*noise[2]
        angle %= 2*pi
        px += vel * np.cos(angle) * dt + random.randn(1)*noise[3]
        py += vel * np.sin(angle) * dt + random.randn(1)*noise[4]
        px = np.clip(px, -5, 5)
        py = np.clip(py, -5, 5)
        return np.array((px, py, angle, vel, ang_vel)).reshape(-1)

    def obs(self, state):
        noise = self.noise
        vel, ang_vel = state[-2:]
        vel = vel + random.randn(1)*noise[5]
        ang_vel = ang_vel + random.randn(1)*noise[6]
        return np.array((vel, ang_vel)).reshape(-1)

    def EKF(self, x, P, a, Y=None):
        Q = self.Q
        R = self.R
        Id = np.eye(5)
        x_ = self.dynamics(x, a) #x_ = np.dot(A(x), x)
        A = self.A(x,a)
        H = self.H(x_)
        P_ = np.dot(np.dot(A, P), A.T) + Q
        if not is_pos_def(P_):
            print("P_:", P_)
            print("P:", P)
            print("A:", A(x,a))
            APA = np.dot(np.dot(A, P), A.T)
            print("APA:", APA)
            print("APA +:", is_pos_def(APA))
        S = R + np.dot(np.dot(H, P_), H.T)
        K = np.dot(np.dot(P_, H.T), inv(S))
        if Y is None:
            Y = self.obs(x_)
        x = x_ + np.dot(K, Y - self.obs(x_)) #x = x_ + np.dot(K, Y - np.dot(H, x_))
        I_KH = (Id - np.dot(K, H))
        P = np.dot(I_KH, P_)
        P = (P + P.T)/2 # make symmetric to avoid computational overflows
        return x, P, K

    def step(self, action):
        #Y = self.obs(self.x)
        self.counter += 1
        self.x, self.P, K = self.EKF(self.x, self.P, action)
        if not is_pos_def(self.P):
            print("x:", self.x)
            print("P:", self.P)
            print(np.linalg.eigvalsh(self.P))
            #e = -min(np.linalg.eigvalsh(self.P))
            #self.P = self.P + np.eye(5)*(e + 1e-5)
        self.L = np.linalg.cholesky(self.P)
        ind_tril = np.tril_indices(self.L.shape[0])

        terminal, final_reward = self.terminal_state(self.x)
        done =  terminal or self.counter >= self.episode_len

        reward = self.reward_func(action) + final_reward

        self.state = np.append(self.x, self.L[ind_tril])
        return self.state, reward, done, {}

    def reset(self):
        #pos = random.uniform(self.world_box[1], self.world_box[0])
        pos = random.multivariate_normal(np.zeros(2), np.eye(2)*4)
        ang = math.atan2(-pos[1], -pos[0]) + random.uniform(-pi/4, pi/4)
        ang %= 2*pi
        self.x = np.append(pos, [ang, 0., 0.])
        self.P = np.eye(5) * (0.0001**2)
        self.L = np.linalg.cholesky(self.P)
        ind_tril = np.tril_indices(self.L.shape[0])
        self.counter = 0
        self.state = np.append(self.x, self.L[ind_tril])
        #print("pretag:", self.state)
        return self.state

    def terminal_state(self, x):
        x0 = np.zeros(4)
        x_ = np.append(x[:2], x[3:])
        Q = np.diag([1., 1., 8., 8.])
        dist_sqr = np.dot((x_ - x0).T.dot(Q),(x_ - x0))
        if np.sqrt(dist_sqr) <= self.goal_radius:
            print("Goal!!")
            return True, 20.
        #if np.sqrt(np.sum(x[3:]**2)) <= 0.1:
        #    print("Stopped!!")
        #    return True, -1.
        return False, 0.

    def reward_func(self, action):
        R = np.eye(4) * 1.25
        P_reduced = np.delete(self.P, 2, 0)
        P_reduced = np.delete(P_reduced, 2, 1)
        #P_reduced = P_reduced + np.eye(4)*1e-4
        P_ = inv(P_reduced)
        S_ = inv(R) + P_
        S = inv(S_)
        mu = np.append(self.x[:2], self.x[3:])
        a = -0.5 * np.dot(mu.T.dot(P_ - np.dot(P_.dot(S), P_)), mu)
        reward = np.exp(a) * np.sqrt(np.linalg.det(S)/np.linalg.det(P_reduced)) #- np.sum(mu[:2]**2)
        #reward -= 1.
        reward -= -0.1*action[0]**2 + 1*action[1]**2 + 1.
        return reward

    def render(self, mode = 'human'):

        screen_width = 500
        screen_height = 500

        world_width = self.world_box[0,0] - self.world_box[1,1]
        #self.max_position[0] - self.min_position[0]
        scale = screen_width/world_width
        #center = (np.zeros(2) - self.min_position) * scale
        center = (np.zeros(2) - self.world_box[1]) * scale
        #goal_position = (self.goal_position - self.min_position) * scale        #
        goal_position = (self.goal_position - self.world_box[1]) * scale
        agent_radius = 10
        goal_radius = 10

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            goal = rendering.make_circle(goal_radius, res=30)
            goal_ring = rendering.make_circle(self.goal_radius * scale, res=30, filled=False)
            goal.set_color(0.6274, 0.8313, 0.4078) #green
            goal_ring.set_color(0.6274, 0.8313, 0.4078) #green

            self.goal_motion = rendering.Transform(translation=center)
            goal.add_attr(self.goal_motion)
            goal_ring.add_attr(self.goal_motion)
            self.viewer.add_geom(goal)
            self.viewer.add_geom(goal_ring)

            agent = rendering.make_circle(agent_radius, res=30)
            agent.set_color(0.9882,  0.4313,  0.3176) #orange
            #head = rendering.make_circle(5)
            head = rendering.make_polygon([(0,5),(0,-5),(5,0)])
            head.set_color(.5, .5, .5)
            head.add_attr(rendering.Transform(translation=(10,0)))

            self.agent_motion = rendering.Transform(translation=(0,0))
            agent.add_attr(self.agent_motion)
            self.headtrans = rendering.Transform()
            head.add_attr(self.headtrans)
            head.add_attr(self.agent_motion)
            self.viewer.add_geom(agent)
            self.viewer.add_geom(head)
            self.viewer.add_geom(agent)

        self.goal_motion.set_translation(goal_position[0], goal_position[1])
        position = self.state[0:2]
        theta = self.state[2]                                                #
        #move = (position - self.min_position) * scale
        move = (position - self.world_box[1]) * scale
        self.agent_motion.set_translation(move[0], move[1])
        self.headtrans.set_rotation(theta)

        pts = np.vstack(ellipse(np.zeros(2), self.P[:2,:2], conf_int=5.991*scale**2)).T
        pts2 = [tuple(v) for v in pts]
        cov = rendering.make_polygon(pts2, False)
        cov.set_color(0.9882,  0.4313,  0.3176)
        cov.add_attr(rendering.Transform(translation=(move[0],move[1])))
        #self.viewer.add_geom(cov)
        self.viewer.geoms[-1] = cov

        #for roatation :: self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array = (mode =='rgb_array'))

    def close(self):
        if self.viewer: self.viewer.close()
