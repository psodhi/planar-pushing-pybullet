import pybullet as pb
import pybullet_data

import time
from datetime import datetime
import numpy as np
import json

from pyquaternion import Quaternion
from shapely.geometry import Point, Polygon, LinearRing
from shapely.ops import nearest_points

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.gridspec import GridSpec

import os
import pdb


class Logger():

    def __init__(self, params):

        # read in sim params
        self.tstart = params['tstart']
        self.tend = params['tend']
        self.dt = params['dt']
        self.sim_length = np.int((self.tend - self.tstart) / self.dt)

        # obj, endeff geometric params
        self.ee_radius = 0.051785
        self.block_width = 0.15
        self.block_height = 0.15

        # time steps
        self.t = np.zeros((self.sim_length, 1))

        # endeffector poses
        self.ee_pos = np.zeros((self.sim_length, 3))
        self.ee_ori = np.zeros((self.sim_length, 4))  # quaternion
        self.ee_ori_mat = np.zeros((self.sim_length, 9))  # rotation matrix

        # object poses
        self.obj_pos = np.zeros((self.sim_length, 3))
        self.obj_ori = np.zeros((self.sim_length, 4))  # quaternion
        self.obj_ori_mat = np.zeros((self.sim_length, 9))  # rotation matrix
        self.obj_ori_rpy = np.zeros((self.sim_length, 3))  # r,p,y angles

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
            circ = Circle((self.ee_pos[idx, 0], self.ee_pos[idx, 1]), self.ee_radius,
                          facecolor='None', edgecolor='black', linestyle='--')
            plt.gca().add_patch(circ)

            # plot object (A: endeff, B: object)
            sz_arw = 0.05

            plt.plot(self.contact_pos_onB[idx, 0], self.contact_pos_onB[idx, 1],
                     color='red', marker='o')
            plt.arrow(self.contact_pos_onB[idx, 0], self.contact_pos_onB[idx, 1],
                      -sz_arw * self.contact_normal_onB[idx, 0],
                      -sz_arw * self.contact_normal_onB[idx, 1],
                      head_width=5e-3)

            yaw = self.obj_ori_rpy[idx, 2]
            R = np.array([[np.cos(yaw), -np.sin(yaw)],
                          [np.sin(yaw), np.cos(yaw)]])
            offset = np.matmul(R, np.array(
                [[0.5*self.block_width], [0.5*self.block_height]]))
            xb = self.obj_pos[idx, 0] - offset[0]
            yb = self.obj_pos[idx, 1] - offset[1]
            rect = Rectangle((xb, yb), self.block_width, self.block_height, angle=(
                np.rad2deg(yaw)), facecolor='None', edgecolor='black')
            plt.gca().add_patch(rect)

            # debugging
            # plt.plot(self.obj_pos[idx, 0], self.obj_pos[idx, 1],
            #  color='black', marker='o')
            # plt.plot(xb, yb, color='black', marker='o')

            plt.draw()
            plt.pause(1e-12)

            if(save_fig):
                plt.savefig('{0}/{1:06d}.png'.format(dst_dir, i))

            plt.cla()

    def plot_traj_contact_data(self, save_fig=False):

        incontact_idxs = np.argwhere(self.contact_flag == 1)
        incontact_idxs = incontact_idxs[:, 0]

        fig1, axes1 = plt.subplots(5, 1, figsize=(10, 10))

        axes1[0].plot(self.ee_pos[:, 0])
        axes1[1].plot(self.ee_pos[:, 1])
        axes1[2].plot(self.obj_pos[:, 0])
        axes1[3].plot(self.obj_pos[:, 1])
        axes1[4].plot(np.rad2deg(self.obj_ori_rpy[:, 2]))

        axes1_name = ['x endeff (m)', 'y endeff (m)',
                      'x block (m)', 'y block (m)', 'block rotation (deg)']
        for i in range(5):
            axes1[i].set_xlabel('time step')
            axes1[i].set_ylabel(axes1_name[i])

        fig2, axes2 = plt.subplots(1, 1, figsize=(8, 5))

        axes2.plot(self.contact_distance[incontact_idxs[50:-1], :])

        plt.show()

    def transform_to_frame2d(self, pt, frame_pose_2d):
        yaw = frame_pose_2d[2, 0]
        R = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
        t = -frame_pose_2d[0:2]
        pt_tf = np.matmul(R, pt+t)

        return pt_tf
    
    def error_contact_factor(self):

        # verify ground truth unary contact factor error
        # (as used by Kuan-Ting et al., ICRA 2018)

        err_vec = np.zeros((self.sim_length, 3))

        fig = plt.figure(constrained_layout=True)
        gs = GridSpec(3, 1, figure=fig)
        ax1 = plt.subplot(gs.new_subplotspec((0, 0), rowspan=3))
        
        save_fig = False
        dst_dir = '../outputs/contact_data/visualize_factor/'
        if (save_fig):
            cmd = 'mkdir -p {0}'.format(dst_dir)
            os.popen(cmd, 'r')

        poly_obj = Polygon([(-0.5*self.block_width, -0.5*self.block_height),
                            (-0.5*self.block_width, 0.5*self.block_height),
                            (0.5*self.block_width, 0.5*self.block_height),
                            (0.5*self.block_width, -0.5*self.block_height)])

        for tstep in range(0, self.sim_length, 50):

            if (self.contact_flag[tstep] == False):
                continue

            frame_pose_2d = np.array([[self.obj_pos[tstep, 0]], [self.obj_pos[tstep, 1]],
                                      [self.obj_ori_rpy[tstep, 2]]])
            ee_center__world = np.array([[self.ee_pos[tstep, 0]],
                                         [self.ee_pos[tstep, 1]]])
            pt_contact__world = np.array([[self.ee_pos[tstep, 0] - self.ee_radius*self.contact_normal_onB[tstep, 0]],
                                          [self.ee_pos[tstep, 1] - self.ee_radius*self.contact_normal_onB[tstep, 1]]])

            ee_center__obj = self.transform_to_frame2d(
                ee_center__world, frame_pose_2d)
            pt_contact__obj = self.transform_to_frame2d(
                pt_contact__world, frame_pose_2d)

            # proj method 1 (pt2poly): using nearest_points
            dist = Point(ee_center__obj[0],
                         ee_center__obj[1]).distance(poly_obj)
            ee_center_poly__obj, p2 = nearest_points(poly_obj, Point(
                ee_center__obj[0], ee_center__obj[1]))

            err_vec[tstep, 0] = dist - self.ee_radius
            err_vec[tstep, 1] = pt_contact__obj[0] - ee_center_poly__obj.x
            err_vec[tstep, 2] = pt_contact__obj[1] - ee_center_poly__obj.y

            # # proj method 2 (pt2poly): using LinearRing
            # pol_ext = LinearRing(poly_obj.exterior.coords)
            # d = pol_ext.project(Point(ee_center__obj[0], ee_center__obj[1]))
            # p = pol_ext.interpolate(d)
            # closest_point_coords = list(p.coords)[0]
            # dist2 = np.linalg.norm(np.array([[closest_point_coords.x], [closest_point_coords.y]]) - ee_center__obj)
            # print("dist2: {0}".format(dist2))

            # ax1 = fig.add_subplot(gs[0, 0])
            # ax2 = fig.add_subplot(gs[0, 1])
            # ax3 = fig.add_subplot(gs[1, 1])
            # ax4 = fig.add_subplot(gs[2, 1])

            incontact_idxs = np.argwhere(self.contact_flag[0:tstep] == 1)
            incontact_idxs = incontact_idxs[:, 0]
            err_vec_plot = err_vec[0:tstep, :]
            # ax2.plot(err_vec_plot[incontact_idxs[50:-1], 0])
            # ax3.plot(err_vec_plot[incontact_idxs[50:-1], 1])
            # ax4.plot(err_vec_plot[incontact_idxs[50:-1], 2])

            ax1.set_xlim((-0.2, 0.2))
            ax1.set_ylim((-0.2, 0.2))

            sz_arw = 0.05 # self.ee_radius
            x, y = poly_obj.exterior.xy
            ax1.plot(x, y, color='grey')

            dxn = ee_center__world - pt_contact__world
            dxn = 1/np.linalg.norm(dxn) * dxn

            ax1.plot(ee_center__obj[0], ee_center__obj[1],
                     color='green', marker='o')
            ax1.plot(pt_contact__obj[0], pt_contact__obj[1],
                     color='red', marker='o')
            ax1.plot(ee_center_poly__obj.x, ee_center_poly__obj.y,
                     color='blue', marker='o')
            circ = Circle((ee_center__obj[0], ee_center__obj[1]), self.ee_radius,
                          facecolor='None', edgecolor='black', linestyle='--')
            ax1.add_patch(circ)

            plt.draw()
            plt.pause(1e-3)

            if(save_fig):
                plt.savefig('{0}/{1:06d}.png'.format(dst_dir, tstep))

            ax1.cla()
            # ax2.cla()
            # ax3.cla()
            # ax4.cla()

    def save_data_json(self, dstfile):

        #  use .tolist() to serialize np arrays
        data = {'ee_pos': self.ee_pos.tolist(),
                'ee_ori': self.ee_ori.tolist()}

        with open(dstfile, 'w') as outfile:
            json.dump(data, outfile, indent=4)
