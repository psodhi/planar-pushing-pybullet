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

    def __init__(self, vis_flag=True):

        if vis_flag:
            pb.connect(pb.GUI)
            self.set_gui_params()
        else:
            pb.connect(pb.DIRECT)

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
        self.object_id = pb.loadURDF("../models/objects/block.urdf")
        pb.resetBasePositionAndOrientation(
            self.object_id, [-0.4, 0, 0.1], [0, 0, 0, 1])

        # set gravity
        pb.setGravity(0, 0, -9.8)
        # set simulation length
        self.sim_length = 10000

        # set/get params
        self.kuka_endeff_idx = 6
        self.num_joints = pb.getNumJoints(self.kuka_id)
        self.joint_ids = [i for i in range(self.num_joints)]
        # joint damping coefficents
        self.jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

        # reset joint states to nominal pose (overrides physics simulation)
        self.reset_joint_states = [0, 0, 0, 0.5 *
                                   math.pi, 0, -math.pi * 0.5 * 0.66, 0]
        for i in range(self.num_joints):
            pb.resetJointState(self.kuka_id, i, self.reset_joint_states[i])

        # set initial endeff pose
        self.init_pos_endeff = [-0.4, -0.2, 0.01]
        self.init_ori_endeff = pb.getQuaternionFromEuler([0, -math.pi, 0])

    def set_gui_params(self):
        cam_tgt_pos = [-0.5, 0.32, -0.15]
        cam_dist = 1
        cam_yaw = 270
        cam_pitch = -16
        pb.resetDebugVisualizerCamera(
            cam_dist, cam_yaw, cam_pitch, cam_tgt_pos)

    def reset_sim(self):

        # reset joint states to nominal pose (overrides physics simulation)
        for i in range(self.num_joints):
            pb.resetJointState(self.kuka_id, i, self.reset_joint_states[i])

        # reset block pose
        pb.resetBasePositionAndOrientation(
            self.object_id, [-0.4, 0, 0.1], [0, 0, 0, 1])

    def simulate(self):
        self.reset_sim()

        theta = 0.0
        push_dir = np.array([math.sin(theta), math.cos(theta), 0.])
        push_step = 0.00005
        x = self.init_pos_endeff[0]
        y = self.init_pos_endeff[1]
        z = self.init_pos_endeff[2]

        for tstep in range(0, self.sim_length):
            pb.stepSimulation()

            # set end effector pose based on push direction/step
            dp = push_step * push_dir
            x += dp[0]
            y += dp[1]
            z += dp[2]
            pos_endeff = [x, y, z]

            # inverse kinematics
            joint_poses = pb.calculateInverseKinematics(
                self.kuka_id, self.kuka_endeff_idx, pos_endeff, self.init_ori_endeff, jointDamping=self.jd)

            # motor control to follow IK solution
            for i in range(0, self.num_joints):
                pb.setJointMotorControl2(bodyIndex=self.kuka_id, jointIndex=i, controlMode=pb.POSITION_CONTROL,
                                         targetPosition=joint_poses[i], targetVelocity=0, force=500, positionGain=0.3, velocityGain=1)


if __name__ == "__main__":
    envkb = EnvKukaBlock()
    envkb.simulate()
