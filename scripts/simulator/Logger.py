import pybullet as pb
import pybullet_data

import time
from datetime import datetime
import numpy as np
import json

from shapely.geometry import Point, Polygon, LinearRing
from shapely.ops import nearest_points

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.gridspec import GridSpec

import os
import pdb


class Logger():

    def __init__(self, params):

        self.params = params

        # read in sim params
        self.tstart = params['tstart']
        self.tend = params['tend']
        self.dt = params['dt']
        self.sim_length = np.int((self.tend - self.tstart) / self.dt)

        # obj, endeff geometric params
        self.ee_radius = params['ee_radius']
        self.block_width = params['block_width']
        self.block_height = params['block_height']

        # time steps
        self.t = np.zeros((self.sim_length, 1))

        # endeffector poses
        self.ee_pos = np.zeros((self.sim_length, 3))
        self.ee_ori = np.zeros((self.sim_length, 4))  # quaternion
        self.ee_ori_mat = np.zeros((self.sim_length, 9))  # rotation matrix
        self.ee_ori_rpy = np.zeros((self.sim_length, 3))  # r,p,y angles

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
        self.lateral_friction_onA = np.zeros((self.sim_length, 1))
        self.lateral_frictiondir_onA = np.zeros((self.sim_length, 3))
        self.lateral_friction_onB = np.zeros((self.sim_length, 1))
        self.lateral_frictiondir_onB = np.zeros((self.sim_length, 3))

    def transform_to_frame2d(self, pt, frame_pose_2d):
        yaw = frame_pose_2d[2, 0]
        R = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
        t = -frame_pose_2d[0:2]
        pt_tf = np.matmul(R, pt+t)

        return pt_tf

    def transform_from_frame2d(self, pt, frame_pose_2d):
        yaw = frame_pose_2d[2, 0]
        R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        t = frame_pose_2d[0:2]
        pt_tf = np.matmul(R, pt)
        pt_tf = t + pt_tf

        return pt_tf

    def proj_ee_center(self):

        poly_obj = Polygon([(-0.0-0.5*self.block_width, -0.5*self.block_height),
                            (-0.0-0.5*self.block_width, 0.5*self.block_height),
                            (-0.0+0.5*self.block_width, 0.5*self.block_height),
                            (-0.0+0.5*self.block_width, -0.5*self.block_height)])

        ee_center_poly = np.zeros((self.sim_length, 2))

        for tstep in range(0, self.sim_length, 1):

            if (self.contact_flag[tstep] == False):
                continue

            frame_pose_2d = np.array([[self.obj_pos[tstep, 0]], [self.obj_pos[tstep, 1]],
                                      [self.obj_ori_rpy[tstep, 2]]])
            ee_center__world = np.array([[self.ee_pos[tstep, 0]],
                                         [self.ee_pos[tstep, 1]]])

            ee_center__obj = self.transform_to_frame2d(
                ee_center__world, frame_pose_2d)

            # proj method 1 (pt2poly): using nearest_points
            dist = Point(ee_center__obj[0],
                         ee_center__obj[1]).distance(poly_obj)
            ee_center_poly__obj, p2 = nearest_points(poly_obj, Point(
                ee_center__obj[0], ee_center__obj[1]))

            ee_center_poly__world = self.transform_from_frame2d(
                np.array([[ee_center_poly__obj.x], [ee_center_poly__obj.y]]), frame_pose_2d)
            ee_center_poly[tstep, :] = ee_center_poly__world.transpose()

        return ee_center_poly

    def visualize_contact_info(self, save_fig=False):
        """Visualize contact info returned from getContactPoints()"""

        incontact_idxs = np.argwhere(self.contact_flag == 1)
        incontact_idxs = incontact_idxs[:, 0]

        save_fig = False
        dst_dir = '../local/outputs/contact_info/traj_circle/'
        if (save_fig):
            cmd = 'mkdir -p {0}'.format(dst_dir)
            os.popen(cmd, 'r')

        fig = plt.figure(figsize=(10, 8))
        skip = 50
        for i in range(0, len(incontact_idxs), skip):
            idx = incontact_idxs[i]

            plt.xlim((-1.0, 0.6))
            plt.ylim((-0.2, 0.8))
            sz_arw = 0.05

            # plot end effector (A: endeff, B: object)
            plt.plot(self.ee_pos[idx, 0], self.ee_pos[idx, 1],
                     color='green', marker='o')
            plt.plot(self.contact_pos_onA[idx, 0], self.contact_pos_onA[idx, 1],
                     color='green', marker='o')
            circ = Circle((self.ee_pos[idx, 0], self.ee_pos[idx, 1]), self.ee_radius,
                          facecolor='None', edgecolor='grey', linestyle='-')
            plt.gca().add_patch(circ)

            # plot object pose
            yaw = self.obj_ori_rpy[idx, 2]
            R = np.array([[np.cos(yaw), -np.sin(yaw)],
                          [np.sin(yaw), np.cos(yaw)]])
            offset = np.matmul(R, np.array(
                [[0.5*self.block_width], [0.5*self.block_height]]))
            xb = self.obj_pos[idx, 0] - offset[0]
            yb = self.obj_pos[idx, 1] - offset[1]
            rect = Rectangle((xb, yb), self.block_width, self.block_height, angle=(
                np.rad2deg(yaw)), facecolor='None', edgecolor='grey')
            plt.gca().add_patch(rect)

            plt.plot(self.contact_pos_onB[idx, 0], self.contact_pos_onB[idx, 1],
                     color='red', marker='o')
            plt.arrow(self.contact_pos_onB[idx, 0], self.contact_pos_onB[idx, 1],
                      -sz_arw * self.contact_normal_onB[idx, 0],
                      -sz_arw * self.contact_normal_onB[idx, 1],
                      head_width=5e-3, color="black")

            # # lateral friction forces
            # plt.arrow(self.contact_pos_onB[idx, 0], self.contact_pos_onB[idx, 1],
            #   sz_arw * self.lateral_frictiondir_onB[idx, 0],
            #   sz_arw * self.lateral_frictiondir_onB[idx, 1],
            #   head_width=5e-3, color="red")
            # plt.arrow(self.contact_pos_onA[idx, 0], self.contact_pos_onA[idx, 1],
            #           sz_arw * self.lateral_frictiondir_onA[idx, 0],
            #           sz_arw * self.lateral_frictiondir_onA[idx, 1],
            #           head_width=5e-3, color="green")

            plt.draw()
            plt.pause(1e-12)

            if(save_fig):
                plt.savefig('{0}/{1:06d}.png'.format(dst_dir, i))

            plt.cla()

    def visualize_contact_factor__world(self, save_fig=False):
        """ Visualize contact factor values as used by Kuan-Ting et al., ICRA 2018 (world frame) """

        incontact_idxs = np.argwhere(self.contact_flag == 1)
        incontact_idxs = incontact_idxs[:, 0]

        # ee_center_poly: ee_center projected on object polygon
        ee_center_poly = self.proj_ee_center()

        save_fig = True
        dst_dir = '../local/outputs/contact_factor__world/traj_circle/'
        if (save_fig):
            cmd = 'mkdir -p {0}'.format(dst_dir)
            os.popen(cmd, 'r')

        fig = plt.figure(figsize=(10, 8))
        skip = 50
        for tstep in range(0, self.sim_length, skip):

            if (self.contact_flag[tstep] == False):
                continue

            plt.xlim((-1.0, 0.6))
            plt.ylim((-0.2, 0.8))
            sz_arw = 0.05

            # plot end effector (A: endeff, B: object)
            plt.plot(self.ee_pos[tstep, 0], self.ee_pos[tstep, 1],
                     color='green', marker='o')
            plt.plot(ee_center_poly[tstep, 0], ee_center_poly[tstep, 1],
                     color='blue', marker='o')
            circ = Circle((self.ee_pos[tstep, 0], self.ee_pos[tstep, 1]), self.ee_radius,
                          facecolor='None', edgecolor='grey', linestyle='-')
            plt.gca().add_patch(circ)

            # plot contact point (computed using force dxn)
            pt_contact__world = np.array([[self.ee_pos[tstep, 0] - self.ee_radius*self.contact_normal_onB[tstep, 0]],
                                          [self.ee_pos[tstep, 1] - self.ee_radius*self.contact_normal_onB[tstep, 1]]])
            plt.plot(pt_contact__world[0], pt_contact__world[1], color='red', marker='o')
            plt.arrow(self.ee_pos[tstep, 0], self.ee_pos[tstep, 1],
                      -(self.ee_radius) * self.contact_normal_onB[tstep, 0],
                      -(self.ee_radius) * self.contact_normal_onB[tstep, 1],
                      head_width=5e-3, color="black")
            
            # test: contact point returned from getContactPoints()
            pt_contact_test__world = np.array([[self.contact_pos_onB[tstep, 0]], [
                                         self.contact_pos_onB[tstep, 1]]])
            plt.plot(pt_contact_test__world[0], pt_contact_test__world[1], color='yellow', marker='o', alpha=1.0)            

            # plot object pose
            yaw = self.obj_ori_rpy[tstep, 2]
            R = np.array([[np.cos(yaw), -np.sin(yaw)],
                          [np.sin(yaw), np.cos(yaw)]])
            offset = np.matmul(R, np.array(
                [[0.5*self.block_width], [0.5*self.block_height]]))
            xb = self.obj_pos[tstep, 0] - offset[0]
            yb = self.obj_pos[tstep, 1] - offset[1]
            rect = Rectangle((xb, yb), self.block_width, self.block_height, angle=(
                np.rad2deg(yaw)), facecolor='None', edgecolor='grey')
            plt.gca().add_patch(rect)

            plt.draw()
            plt.pause(1e-12)

            if(save_fig):
                plt.savefig('{0}/{1:06d}.png'.format(dst_dir, tstep))

            plt.cla()

    def visualize_contact_factor__obj(self):
        """ Visualize (and plot) contact factor values as used by Kuan-Ting et al., ICRA 2018 (object frame) """

        err_vec = np.zeros((self.sim_length, 3))

        fig = plt.figure(constrained_layout=True, figsize=(10, 8))
        nrows = 3
        ncols = 1
        gs = GridSpec(nrows, ncols, figure=fig)

        save_fig = True
        dst_dir = '../local/outputs/contact_factor__obj/traj_circle/'
        if (save_fig):
            cmd = 'mkdir -p {0}'.format(dst_dir)
            os.popen(cmd, 'r')

        poly_obj = Polygon([(-0.5*self.block_width, -0.5*self.block_height),
                            (-0.5*self.block_width, 0.5*self.block_height),
                            (0.5*self.block_width, 0.5*self.block_height),
                            (0.5*self.block_width, -0.5*self.block_height)])

        skip = 50
        for tstep in range(0, self.sim_length, skip):

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

            # test: contact point returned from getContactPoints()
            pt_contact_test__world = np.array([[self.contact_pos_onB[tstep, 0]],
                                          [self.contact_pos_onB[tstep, 1]]])
            pt_contact_test__obj = self.transform_to_frame2d(
                pt_contact_test__world, frame_pose_2d)

            # project 2d point on object polygon
            dist = Point(ee_center__obj[0],
                         ee_center__obj[1]).distance(poly_obj)
            ee_center_poly__obj, p2 = nearest_points(poly_obj, Point(
                ee_center__obj[0], ee_center__obj[1]))

            err_vec[tstep, 0] = dist - self.ee_radius
            err_vec[tstep, 1] = pt_contact__obj[0] - ee_center_poly__obj.x
            err_vec[tstep, 2] = pt_contact__obj[1] - ee_center_poly__obj.y

            ax = [None] * (nrows*ncols)
            ax[0] = fig.add_subplot(gs[:, 0])
            # ax[0] = plt.subplot(gs.new_subplotspec((0, 0), rowspan=3))
            # ax[1] = fig.add_subplot(gs[0, 1])
            # ax[2] = fig.add_subplot(gs[1, 1])
            # ax[3] = fig.add_subplot(gs[2, 1])

            incontact_idxs = np.argwhere(self.contact_flag[0:tstep] == 1)
            incontact_idxs = incontact_idxs[:, 0]
            err_vec_plot = err_vec[0:tstep, :]
            # ax[1].plot(err_vec_plot[incontact_idxs[50:-1], 0])
            # ax[2].plot(err_vec_plot[incontact_idxs[50:-1], 1])
            # ax[3].plot(err_vec_plot[incontact_idxs[50:-1], 2])

            ax[0].set_xlim((-0.2, 0.2))
            ax[0].set_ylim((-0.2, 0.2))

            sz_arw = 0.05  # self.ee_radius
            x, y = poly_obj.exterior.xy
            ax[0].plot(x, y, color='grey')
            ax[0].plot(ee_center__obj[0], ee_center__obj[1],
                       color='green', marker='o')
            ax[0].plot(pt_contact__obj[0], pt_contact__obj[1],
                       color='red', marker='o')
            ax[0].plot(ee_center_poly__obj.x, ee_center_poly__obj.y,
                       color='blue', marker='o')
            circ = Circle((ee_center__obj[0], ee_center__obj[1]), self.ee_radius,
                          facecolor='None', edgecolor='grey', linestyle='-')
            ax[0].add_patch(circ)

            dxn = ee_center__obj - pt_contact__obj
            dxn = 1/np.linalg.norm(dxn) * dxn
            ax[0].arrow(ee_center__obj[0, 0], ee_center__obj[1, 0],
                        -(self.ee_radius) * dxn[0, 0],
                        -(self.ee_radius) * dxn[1, 0],
                        head_width=2e-3, color="black")
            
            # test: contact point returned from getContactPoints()
            ax[0].plot(pt_contact_test__obj[0], pt_contact_test__obj[1],
                       color='yellow', marker='o', alpha=1.0)

            plt.draw()
            plt.pause(1e-3)

            if(save_fig):
                plt.savefig('{0}/{1:06d}.png'.format(dst_dir, tstep))

            ax[0].cla()
            # ax[1].cla()
            # ax[2].cla()
            # ax[3].cla()

    def plot_force_data(self, save_fig=False):

        fig = plt.figure(constrained_layout=True)
        nrows = 3
        ncols = 2
        gs = GridSpec(nrows, ncols, figure=fig)

        incontact_idxs = np.argwhere(self.contact_flag == 1)
        incontact_idxs = incontact_idxs[:, 0]

        ax = [None] * (nrows*ncols)
        ax[0] = fig.add_subplot(gs[0, 0])
        ax[1] = fig.add_subplot(gs[1, 0])
        ax[2] = fig.add_subplot(gs[2, 0])
        ax[3] = fig.add_subplot(gs[0, 1])
        ax[4] = fig.add_subplot(gs[1, 1])
        ax[5] = fig.add_subplot(gs[2, 1])

        # force values
        ax[0].plot(self.contact_normal_force[incontact_idxs, :])
        ax[1].plot(self.lateral_friction_onA[incontact_idxs, :])
        ax[2].plot(self.lateral_friction_onB[incontact_idxs, :])

        # force directions
        ax[3].plot(self.contact_normal_onB[incontact_idxs, 0], color='red')
        ax[3].plot(self.contact_normal_onB[incontact_idxs, 1], color='green')
        # ax[3].plot(self.contact_normal_onB[incontact_idxs, 2], color='blue')
        ax[3].legend(['x dxn', 'y dxn'])

        ax[4].plot(self.lateral_frictiondir_onA[incontact_idxs, 0], color='red')
        ax[4].plot(self.lateral_frictiondir_onA[incontact_idxs, 1],
                   color='green')
        # ax[4].plot(self.lateral_frictiondir_onA[incontact_idxs, 2], color='blue')
        ax[4].legend(['x dxn', 'y dxn'])

        ax[5].plot(self.lateral_frictiondir_onB[incontact_idxs, 0], color='red')
        ax[5].plot(self.lateral_frictiondir_onB[incontact_idxs, 1],
                   color='green')
        # ax[5].plot(self.lateral_frictiondir_onB[incontact_idxs, 2], color='blue')
        ax[5].legend(['x dxn', 'y dxn'])

        plt.show()

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

    def save_data2d_json(self, dstfile):

        ee_poses_2d = np.zeros((self.sim_length, 3))
        obj_poses_2d = np.zeros((self.sim_length, 3))
        ee_poses_2d[:, 0:2] = self.ee_pos[:, 0:2]
        ee_poses_2d[:, 2] = self.ee_ori_rpy[:, 2]
        obj_poses_2d[:, 0:2] = self.obj_pos[:, 0:2]
        obj_poses_2d[:, 2] = self.obj_ori_rpy[:, 2]

        #  use .tolist() to serialize np arrays
        data = {'params': self.params,
                'ee_poses_2d': ee_poses_2d.tolist(),
                'obj_poses_2d': obj_poses_2d.tolist(),
                'contact_normals_2d': (-self.contact_normal_onB[:, 0:2]).tolist(),
                'contact_normal_forces': self.contact_normal_force.tolist(),
                'contact_points_gt_2d': (self.contact_pos_onB[:, 0:2]).tolist()}

        with open(dstfile, 'w') as outfile:
            json.dump(data, outfile, indent=4)
