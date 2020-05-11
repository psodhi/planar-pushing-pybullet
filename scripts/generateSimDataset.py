from simulator.EnvKukaArmBlock import EnvKukaArmBlock
from simulator.EnvFloatingArmBlock import EnvFloatingArmBlock
from simulator.Trajectories import Trajectories
from simulator.Logger import Logger
from simulator.Visualizer import Visualizer

import pybullet as pb
import numpy as np
import matplotlib.pyplot as plt
import pdb

import json


def init_params_kukablock():

    # sim params
    params = {}
    params['tstart'] = 0.0
    params['tend'] = 15.0
    params['dt'] = 1e-3
    params['gravity'] = 10

    # end effector, object geometries
    params['ee_radius'] = 0.02
    params['block_size_x'] = 0.4
    params['block_size_y'] = 0.15
    params['block_size_z'] = 0.05
    params['block_mass'] = 0.5

    # initial end effector pose
    params['init_ee_pos'] = [-0.4, -0.28, 0.01]
    params['init_ee_ori'] = pb.getQuaternionFromEuler([0, -np.pi, 0])

    # initial object pose
    params['init_obj_pos'] = [-0.4, 0, 0.1]
    params['init_obj_ori'] = [0, 0, 0, 1]

    # friction params
    params['mu_g'] = 0.8
    
    return params

def planar_pushing_kukablock():
    vis_flag = True

    params = init_params_kukablock()
    env = EnvKukaArmBlock(params, vis_flag)
    traj = Trajectories(params)

    num_runs = 1
    for run in range(0, num_runs):
        # traj_vec = traj.get_traj_line(0.0)
        traj_vec = traj.get_traj_circle()

        env.simulate(traj_vec)
        # env.simulate_reinitialize(traj_vec)

    logger = env.get_logger()
    dataset_name = "logKBCircle1"
    logger.save_data2d_json("../local/data/{0}.json".format(dataset_name))

    visualizer = Visualizer(params, logger)
    # visualizer.plot_force_data()
    # visualizer.plot_qs_model1_errors()
    visualizer.visualize_contact_info()
    # visualizer.visualize_contact_factor__world()
    # visualizer.visualize_contact_factor__obj()
    # visualizer.plot_traj_contact_data()

def init_params_floatingblock():
    
    # sim params
    params = {}
    params['tstart'] = 0.0
    params['tend'] = 1.0
    params['dt'] = 1e-3
    params['gravity'] = 10

    # end effector, object geometries
    params['ee_radius'] = 0.04
    params['block_size_x'] = 0.15
    params['block_size_y'] = 0.40
    params['block_size_z'] = 0.15
    params['block_mass'] = 0.5

    # initial end effector pose
    init_ee_pos_z = 0.5 * params['block_size_z']
    params['init_ee_pos'] = [-0.3, -0.05, init_ee_pos_z]
    params['init_ee_ori'] = pb.getQuaternionFromEuler([0, 0, -np.pi])

    # initial object pose
    params['init_obj_pos'] = [0, 0.0, 1.0]
    params['init_obj_ori'] = pb.getQuaternionFromEuler([0, 0, 0])

    # friction params
    params['mu_g'] = 0.8

    return params

def planar_pushing_floatingblock():

    params = init_params_floatingblock()

    first_run = True
    init_ee_y = [-0.052, -0.047, -0.04, 0, 0.04, 0.047, 0.052]
    num_runs = len(init_ee_y)
    for run_idx in range(0, num_runs):
        params['init_ee_pos'][1] = init_ee_y[run_idx]
        if (first_run):
            env = EnvFloatingArmBlock(params, True)
            env.simulate()
            first_run = False
        else:
            env = EnvFloatingArmBlock(params, False)
            env.simulate()

        logger = env.get_logger()
        dataset_name = "logFBTraj{0}".format(run_idx)
        logger.save_data2d_json("../local/data/{0}.json".format(dataset_name))

    visualizer = Visualizer(params, logger)
    # visualizer.plot_force_data()
    # visualizer.visualize_contact_info()
    visualizer.plot_qs_push_dynamics_errors()

def main():
    # planar_pushing_kukablock()
    planar_pushing_floatingblock()


if __name__ == "__main__":
    main()
