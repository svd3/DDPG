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

#from gym.envs.classic_control import rendering



class FireflyEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.counter = 0
        self.targetUpdatefreq = 100 # Not being used

        self.max_action = 0.01
        self.world_box = np.array([[5.0, 5.0], [-5.0, -5.0]])
        #self.min_position = np.array([-5.0, -5.0])
        self.xlow = np.append(self.world_box[1], [0., -1., -1.])
        self.xhigh = np.append(self.world_box[0], [2*pi, 1., 1.])
        self.low_state = np.append(self.xlow, -10*np.ones(15))       #
        self.high_state = np.append(self.xhigh, 10*np.ones(15))       #

        self.action_space = spaces.Box(-np.ones(2), np.ones(2))
        self.observation_space = spaces.Box(self.low_state, self.high_state)

        self.viewer = None
        #self.state = self.observation_space.sample()

        self.noise = np.array([0.01]*5 + [0.2]*2) #std
        dt = 0.1
        self.Q = np.eye(5) * (0.01**2)
        self.R = np.eye(2) * (0.2**2)
        self.P = np.eye(5) * (0.0001**2)
        self.Id = np.eye(5)

        # initialize state
        pos = random.uniform(self.world_box[0], self.world_box[1])
        ang = math.atan2(pos[1], pos[1]) -pi + random.uniform(-pi/8, pi/8)
        ang %= 2*pi
        self.x = np.append(pos, [ang, 0., 0.])
        self.L = np.linalg.cholesky(self.P)

        self.goal_position = np.array([0., 0.])

        self.A = jacobian(self.dynamics)
        self.H = jacobian(self.obs)
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def dynamics(self, x, action):
        dt = 0.1
        action = np.clip(action, -1., 1.)
        noise = self.noise
        px, py, angle, vel, ang_vel = x
        vel = 0.*vel + action[0] * dt + random.randn(1)*noise[0]
        ang_vel = 0.*ang_vel + action[1] * dt + random.randn(1)*noise[1]
        angle += ang_vel * dt + random.randn(1)*noise[2]
        angle %= 2*pi
        px += vel * np.cos(angle) * dt + random.randn(1)*noise[3]
        py += vel * np.sin(angle) * dt + random.randn(1)*noise[4]
        return np.array((px, py, angle, vel, ang_vel)).reshape(-1)

    def obs(self, x):
        vel, ang_vel = x[-2:]
        noise = self.noise
        vel = vel + random.randn(1)*noise[5]*vel
        ang_vel = ang_vel + random.randn(1)*noise[6]*ang_vel
        return np.array((vel, ang_vel)).reshape(-1)

    def EKF(self, x, P, a, Y=None):
        Q = self.Q
        R = self.R
        A = self.A
        H = self.H
        x_ = self.dynamics(x, a) #x_ = np.dot(A(x), x)
        P_ = np.dot(np.dot(A(x,a), P), A(x,a).T) + Q
        S = R + np.dot(np.dot(H(x_), P_), H(x_).T)
        K = np.dot(np.dot(P_, H(x_).T), inv(S))
        if Y is None:
            Y = self.obs(x_)
        x = x_ + np.dot(K, Y - self.obs(x_)) #x = x_ + np.dot(K, Y - np.dot(H, x_))
        Id = np.eye(P.shape[0])
        I_KH = (Id - np.dot(K, H(x_)))
        P = np.dot(I_KH, P_)
        return x, P, K

    def terminal_state(self, x):
        x0 = np.zeros(4)
        x_ = np.append(x[:2], x[3:])
        dist_sqr = (x_ - x0).T.dot(x_ - x0)
        if np.sqrt(dist_sqr) <= 0.4:
            return True
        return False

    def reward_func(self, action):
        R = np.eye(4) * 0.16
        P_reduced = np.delete(self.P, 2, 0)
        P_reduced = np.delete(P_reduced, 2, 1)
        P_ = inv(P_reduced)
        S_ = inv(R) + P_
        S = inv(S_)
        mu = np.append(self.x[:2], self.x[3:])
        a = -0.5 * np.dot(mu.T.dot(P_ - np.dot(P_.dot(S), P_)), mu)
        return np.exp(a) * np.sqrt(np.linalg.det(S)/np.linalg.det(P_reduced)) - 1.


    def step(self, action):
        #Y = self.obs(self.x)
        self.counter += 1
        self.x, self.P, K = self.EKF(self.x, self.P, action)
        self.L = np.linalg.cholesky(self.P)
        ind_tril = np.tril_indices(self.L.shape[0])
        done = self.terminal_state(self.x)
        reward = self.reward_func(action)

        self.state = np.append(self.x, self.L[ind_tril])
        if self.counter > 256:
            done = True
        return self.state, reward, done, {}

    def reset(self):
        pos = random.uniform(self.world_box[1]+1, self.world_box[0]-1)
        #pos = random.multivariate_normal(np.zeros(2), np.eye(2)*1.25)
        ang = math.atan2(-pos[1], -pos[0]) + random.uniform(-pi/8, pi/8)
        ang %= 2*pi
        self.x = np.append(pos, [ang, 0., 0.])
        self.P = np.eye(5) * (0.0001**2)
        self.L = np.linalg.cholesky(self.P)
        ind_tril = np.tril_indices(self.L.shape[0])
        self.counter = 0
        self.state = np.append(self.x, self.L[ind_tril])
        #state = self.state#self._get_state()
        return self.state

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

            goal = rendering.make_circle(goal_radius, res = 30)
            goal.set_color(0.6274, 0.8313, 0.4078) #green

            self.goal_motion = rendering.Transform(translation = center)
            goal.add_attr(self.goal_motion)
            self.viewer.add_geom(goal)

            agent = rendering.make_circle(agent_radius, res = 30)
            agent.set_color(0.9882,  0.4313,  0.3176) #orange
            #head = rendering.make_circle(5)
            head = rendering.make_polygon([(0,5),(0,-5),(5,0)])
            head.set_color(.5, .5, .5)
            head.add_attr(rendering.Transform(translation=(10,0)))

            self.agent_motion = rendering.Transform(translation = (0,0))
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
