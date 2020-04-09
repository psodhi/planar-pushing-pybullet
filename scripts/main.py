from Environments import EnvKukaBlock
from Trajectories import Trajectories

import pybullet as pb
import numpy as np
import matplotlib.pyplot as plt
import pdb


def init_params():

    # sim params
    params = {}
    params['tstart'] = 0.0
    params['tend'] = 15.0
    params['dt'] = 1e-3

    # initial end effector pose
    params['init_ee_pos'] = [-0.4, -0.2, 0.01]
    params['init_ee_ori'] = pb.getQuaternionFromEuler([0, -np.pi, 0])

    # initial obj pose
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
        # traj_vec = traj.get_traj_line()
        traj_vec = traj.get_traj_circle()
        envkb.simulate(traj_vec)

    logger = envkb.get_logger()
    # logger.error_contact_factor()
    # logger.visualize_contact_data()
    # logger.plot_contact_data()
    # logger.save_data_json("../outputs/log.json")


if __name__ == "__main__":
    main()
