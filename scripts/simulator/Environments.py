from .Logger import Logger

import pybullet as pb
import pybullet_data

import time
from datetime import datetime

import math
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

import time


class EnvKukaBlock():

    def __init__(self, params, vis_flag=True):

        if vis_flag:
            pb.connect(pb.GUI)
            self.set_gui_params()
        else:
            pb.connect(pb.DIRECT)

        # read in sim params
        self.tstart = params['tstart']
        self.tend = params['tend']
        self.dt = params['dt']
        self.sim_length = np.int((self.tend - self.tstart) / self.dt)

        self.init_ee_pos = params['init_ee_pos']
        self.init_ee_ori = params['init_ee_ori']
        self.init_obj_pos = params['init_obj_pos']
        self.init_obj_ori = params['init_obj_ori']

        self.logger = Logger(params)

        # set additional path to access bullet lib models
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())

        # add pushing plane (slightly below robot arm base)
        self.plane_id = pb.loadURDF(
            "plane.urdf", [0, 0, -0.05], useFixedBase=True)

        # add kuka arm
        self.kuka_id = pb.loadURDF(
            "kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
        pb.resetBasePositionAndOrientation(
            self.kuka_id, [0, 0, 0], [0, 0, 0, 1])

        # add object being pushed
        self.obj_id = pb.loadURDF("../models/objects/block.urdf")
        pb.resetBasePositionAndOrientation(
            self.obj_id, self.init_obj_pos, self.init_obj_ori)

        # set gravity
        pb.setGravity(0, 0, -10)

        # set/get arm params
        self.kuka_ee_idx = 6
        self.num_joints = pb.getNumJoints(self.kuka_id)
        self.joint_ids = [i for i in range(self.num_joints)]

        # joint damping coefficents
        self.jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

        # reset joint states to nominal pose (overrides physics simulation)
        self.reset_joint_states = [0, 0, 0, 0.5 *
                                   math.pi, 0, -math.pi * 0.5 * 0.66, 0]
        for i in range(self.num_joints):
            pb.resetJointState(self.kuka_id, i, self.reset_joint_states[i])

        self.record_log_video = False
        if (self.record_log_video):
            self.log_id = pb.startStateLogging(
                pb.STATE_LOGGING_VIDEO_MP4, "../local/outputs/push_block_kuka.mp4")

    def set_gui_params(self):
        cam_tgt_pos = [-0.5, 0.32, -0.15]
        cam_dist = 1
        cam_yaw = 270
        cam_pitch = -16
        pb.resetDebugVisualizerCamera(
            cam_dist, cam_yaw, cam_pitch, cam_tgt_pos)

    def get_logger(self):
        return self.logger

    def log_step(self, tstep):

         # get contact, link, obj information (A: self.kuka_id, B: self.obj_id)
        contact_info = pb.getContactPoints(self.kuka_id, self.obj_id)
        link_state = pb.getLinkState(self.kuka_id, self.kuka_ee_idx)
        obj_pose = pb.getBasePositionAndOrientation(self.obj_id)

        # store in logger object
        self.logger.t[tstep, :] = self.tstart + tstep * self.dt

        self.logger.ee_pos[tstep, :] = link_state[0]
        self.logger.ee_ori[tstep, :] = link_state[1]
        self.logger.ee_ori_mat[tstep, :] = pb.getMatrixFromQuaternion(
            link_state[1])  # row-major order
        self.logger.ee_ori_rpy[tstep, :] = pb.getEulerFromQuaternion(
            self.logger.ee_ori[tstep, :])

        self.logger.obj_pos[tstep, :] = obj_pose[0]
        self.logger.obj_ori[tstep, :] = obj_pose[1]
        self.logger.obj_ori_mat[tstep, :] = pb.getMatrixFromQuaternion(
            obj_pose[1])  # row-major order
        self.logger.obj_ori_rpy[tstep, :] = pb.getEulerFromQuaternion(
            self.logger.obj_ori[tstep, :])

        if (len(contact_info) > 0):
            self.logger.contact_flag[tstep, :] = 1
            self.logger.contact_pos_onA[tstep, :] = contact_info[0][5]
            self.logger.contact_pos_onB[tstep, :] = contact_info[0][6]
            self.logger.contact_normal_onB[tstep, :] = contact_info[0][7]
            self.logger.contact_distance[tstep, :] = contact_info[0][8]
            self.logger.contact_normal_force[tstep, :] = contact_info[0][9]

            self.logger.lateral_friction_onA[tstep, :] = contact_info[0][10]
            self.logger.lateral_frictiondir_onA[tstep, :] = contact_info[0][11]
            self.logger.lateral_friction_onB[tstep, :] = contact_info[0][12]
            self.logger.lateral_frictiondir_onB[tstep, :] = contact_info[0][13]

    def reset_sim(self):

        # reset joint states to nominal pose (overrides physics simulation)
        for i in range(self.num_joints):
            pb.resetJointState(self.kuka_id, i, self.reset_joint_states[i])

        # reset block pose
        pb.resetBasePositionAndOrientation(
            self.obj_id, self.init_obj_pos, self.init_obj_ori)

    def simulate_reinitialize(self, traj_vec):
        """" Sticky contacts test: Re-initialize simulation with states from previous time step """

        self.reset_sim()
        reinit_obj_state = True
        reinit_ee_state = True

        for tstep in range(0, self.sim_length):

            # re-init object pose to value in previous time step
            if (reinit_obj_state):
                if (tstep == 0):
                    pb.resetBasePositionAndOrientation(
                        self.obj_id, self.init_obj_pos, self.init_obj_ori)
                else:
                    pb.resetBasePositionAndOrientation(
                        self.obj_id, self.logger.obj_pos[tstep-1, :], self.logger.obj_ori[tstep-1, :])

            # re-init arm joint states to exact IK solution
            pos_ee = [traj_vec[tstep, 0],
                      traj_vec[tstep, 1], traj_vec[tstep, 2]]
            joint_poses = pb.calculateInverseKinematics(
                self.kuka_id, self.kuka_ee_idx, pos_ee, self.init_ee_ori, jointDamping=self.jd)
            if (reinit_ee_state):
                for i in range(0, self.num_joints):
                    pb.resetJointState(self.kuka_id, i, joint_poses[i])
            else:
                for i in range(0, self.num_joints):
                    pb.setJointMotorControl2(bodyIndex=self.kuka_id, jointIndex=i, controlMode=pb.POSITION_CONTROL,
                                             targetPosition=joint_poses[i], targetVelocity=0, force=500, positionGain=0.3, velocityGain=1)

            pb.stepSimulation()
            self.log_step(tstep)
            # time.sleep(1. / 240.)

        if (self.record_log_video):
            pb.stopStateLogging(self.log_id)

    def simulate(self, traj_vec):
        self.reset_sim()

        for tstep in range(0, self.sim_length):

            pos_ee = [traj_vec[tstep, 0],
                      traj_vec[tstep, 1], traj_vec[tstep, 2]]

            # inverse kinematics
            joint_poses = pb.calculateInverseKinematics(
                self.kuka_id, self.kuka_ee_idx, pos_ee, self.init_ee_ori, jointDamping=self.jd)

            # motor control to follow IK solution
            for i in range(0, self.num_joints):
                pb.setJointMotorControl2(bodyIndex=self.kuka_id, jointIndex=i, controlMode=pb.POSITION_CONTROL,
                                         targetPosition=joint_poses[i], targetVelocity=0, force=500, positionGain=0.3, velocityGain=1)

            pb.stepSimulation()
            self.log_step(tstep)
            # time.sleep(1. / 240.)

        if (self.record_log_video):
            pb.stopStateLogging(self.log_id)
