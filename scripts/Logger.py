import pybullet as pb
import pybullet_data

import time
from datetime import datetime
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

import os
import pdb

class Logger():

    def __init__(self, params):

        # read in sim params
        self.tstart = params['tstart']
        self.tend = params['tend']
        self.dt = params['dt']
        self.sim_length = np.int((self.tend - self.tstart) / self.dt)

        # time steps
        self.t = np.zeros((self.sim_length, 1))

        # endeffector poses
        self.ee_pos = np.zeros((self.sim_length, 3))
        self.ee_ori = np.zeros((self.sim_length, 4))

        # object poses
        self.obj_pos = np.zeros((self.sim_length, 3))
        self.obj_ori = np.zeros((self.sim_length, 4))

        # contact pos/force measurements (A: endeff, B: object)
        self.contact_flag = np.zeros((self.sim_length, 1))
        self.contact_pos_onA = np.zeros((self.sim_length, 3))
        self.contact_pos_onB = np.zeros((self.sim_length, 3))
        self.contact_normal_onB = np.zeros((self.sim_length, 3))
        self.contact_distance = np.zeros((self.sim_length, 1))
        self.contact_normal_force = np.zeros((self.sim_length, 1))

        # contact friction measurements
        # self.lateral_friction1 = np.zeros((self.sim_length, 1))
        # self.lateral_frictiondir1 = np.zeros((self.sim_length, 3))

    def visualize_contact_data(self, save_fig=False):

        incontact_idxs = np.argwhere(self.contact_flag == 1)
        incontact_idxs = incontact_idxs[:, 0]

        # save_fig = True
        dst_dir = '../outputs/contact_data/'
        if (save_fig):
            cmd = 'mkdir -p {0}'.format(dst_dir)
            os.popen(cmd, 'r')

        fig = plt.figure(figsize=(5, 4))
        skip = 50

        for i in range(0, len(incontact_idxs), skip):
            idx = incontact_idxs[i]

            plt.xlim((-1.0, 0.2))
            plt.ylim((-0.2, 0.8))

            # plot end effector (A: endeff, B: object)
            plt.plot(self.ee_pos[idx, 0], self.ee_pos[idx, 1],
                     color='green', marker='o')
            plt.plot(self.contact_pos_onA[idx, 0], self.contact_pos_onA[idx, 1],
                     color='green', marker='o')
            circ = Circle((self.ee_pos[idx, 0], self.ee_pos[idx, 1]), 0.05,
                          facecolor='None', edgecolor='black', linestyle='--')
            plt.gca().add_patch(circ)

            # plot object (A: endeff, B: object)
            sz_arw = 0.05
            block_width = 0.15
            block_height = 0.15

            plt.plot(self.contact_pos_onB[idx, 0], self.contact_pos_onB[idx, 1],
                     color='red', marker='o')
            plt.arrow(self.contact_pos_onB[idx, 0], self.contact_pos_onB[idx, 1],
                      -sz_arw * self.contact_normal_onB[idx, 0],
                      -sz_arw * self.contact_normal_onB[idx, 1],
                      head_width=5e-3)

            xb = self.obj_pos[idx, 0] - 0.5*block_width
            yb = self.obj_pos[idx, 1] - 0.5*block_height
            roll, pitch, yaw = pb.getEulerFromQuaternion(self.obj_ori[idx, :])
            rect = Rectangle((xb, yb), block_width, block_height, angle=yaw, facecolor='None', edgecolor='black')
            plt.gca().add_patch(rect)

            plt.draw()
            plt.pause(1e-12)

            if(save_fig): 
                plt.savefig('{0}/{1:06d}.png'.format(dst_dir, i))

            plt.cla()