from simulator.Environments import EnvKukaBlock
from simulator.Trajectories import Trajectories
from simulator.Logger import Logger
from simulator.Visualizer import Visualizer

import pybullet as pb
import numpy as np
import matplotlib.pyplot as plt
import pdb

import json


def init_params():

    # sim params
    params = {}
    params['tstart'] = 0.0
    params['tend'] = 15.0
    params['dt'] = 1e-3

    # end effector, object geometries
    params['ee_radius'] = 0.051
    params['block_width'] = 0.30
    params['block_height'] = 0.15

    # initial end effector pose
    params['init_ee_pos'] = [-0.4, -0.26, 0.01]
    params['init_ee_ori'] = pb.getQuaternionFromEuler([0, -np.pi, 0])

    # initial object pose
    params['init_obj_pos'] = [-0.4, 0, 0.1]
    params['init_obj_ori'] = [0, 0, 0, 1]

    return params


def main():
    vis_flag = True

    params = init_params()
    envkb = EnvKukaBlock(params, vis_flag)
    traj = Trajectories(params)

    num_runs = 1
    for run in range(0, num_runs):
        traj_vec = traj.get_traj_circle()
        # traj_vec = traj.get_traj_line()

        envkb.simulate(traj_vec)
        # envkb.simulate_reinitialize(traj_vec)

    logger = envkb.get_logger()
    dataset_name = "logCircle1"
    logger.save_data2d_json("../local/data/{0}.json".format(dataset_name))

    visualizer = Visualizer(params, logger)
    # visualizer.visualize_contact_info()
    visualizer.visualize_contact_factor__world()
    # visualizer.visualize_contact_factor__obj()
    # visualizer.plot_traj_contact_data()
    # visualizer.plot_force_data()
    

if __name__ == "__main__":
    main()
