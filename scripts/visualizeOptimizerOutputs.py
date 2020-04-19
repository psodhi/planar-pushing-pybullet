import pybullet as pb
import numpy as np
import json

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

import pdb


def read_poses_g2o(srcfile):
    # todo
    data = np.loadtxt(srcfile, dtype="S100,i8,f8,f8,f8")


def read_poses_json(srcfile):
    data = []
    with open(srcfile) as f:
        data = json.load(f)

    nposes = len(data['poses'])
    poses = np.zeros((nposes, 3))
    for i in range(nposes):
        poses[i, :] = np.asarray(data['poses'][i][1])

    return poses


def draw_object(pose, params, color_val):
    yaw = pose[2]
    R = np.array([[np.cos(yaw), -np.sin(yaw)],
                  [np.sin(yaw), np.cos(yaw)]])
    offset = np.matmul(R, np.array(
        [[0.5*params['block_width']], [0.5*params['block_height']]]))
    rect = Rectangle((pose[0] - offset[0], pose[1] - offset[1]),
                     params['block_width'], params['block_height'],
                     angle=(np.rad2deg(yaw)), facecolor='None', edgecolor=color_val, alpha=0.75)
    return rect


def visualize_poses(dataset, poses1, poses2, poses3):

    nposes = poses1.shape[0]
    skip = 1

    params = dataset['params']
    ee_poses = dataset['ee_poses_2d']

    ds_idx = 0
    ds_skip = 50
    fig = plt.figure(figsize=(16, 12))
    for idx in range(0, nposes, skip):

        plt.xlim((-1.0, 0.6))
        plt.ylim((-0.2, 0.8))

        # plot end-effector
        plt.plot(ee_poses[ds_idx][0], ee_poses[ds_idx]
                 [1], color='black', marker='o')
        circ = Circle((ee_poses[ds_idx][0], ee_poses[ds_idx][1]), params['ee_radius'],
                      facecolor='None', edgecolor='grey', linestyle='-')
        plt.gca().add_patch(circ)

        # plot object
        rect1 = draw_object(poses1[idx, :], params, 'green')
        rect2 = draw_object(poses2[idx, :], params, 'blue')
        rect3 = draw_object(poses3[idx, :], params, 'red')
        plt.gca().add_patch(rect1)
        plt.gca().add_patch(rect2)
        plt.gca().add_patch(rect3)

        plt.plot(poses1[1:idx, 0], poses1[1:idx, 1],
                 color='green', linestyle='-')
        plt.plot(poses2[1:idx, 0], poses2[1:idx, 1],
                 color='blue', linestyle='-')
        plt.plot(poses3[1:idx, 0], poses3[1:idx, 1],
                 color='red', linestyle='-')

        ds_idx = ds_idx + ds_skip

        plt.draw()
        plt.pause(1e-3)
        plt.cla()


def main():
    dataset_name = "logCircle1"
    srcdir = "../local/outputs/optimizer"

    filename = "../local/data/{0}.json".format(dataset_name)
    with open(filename) as f:
        dataset = json.load(f)

    nsteps = 300
    for tstep in range(nsteps-1, nsteps):

        filename = "{0}/groundtruth/{1}UnconIter{2}.json".format(
            srcdir, dataset_name, str(tstep).zfill(3))
        poses_gt = read_poses_json(filename)

        filename = "{0}/odom_noisy/{1}UnconIter{2}.json".format(
            srcdir, dataset_name, str(tstep).zfill(3))
        poses_odom = read_poses_json(filename)

        filename = "{0}/contact/{1}UnconIter{2}.json".format(
            srcdir, dataset_name, str(tstep).zfill(3))
        poses_uncon = read_poses_json(filename)

        visualize_poses(dataset, poses_gt, poses_odom, poses_uncon)


if __name__ == "__main__":
    main()
