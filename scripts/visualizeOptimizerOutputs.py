import pybullet as pb
import numpy as np
import json
import os

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

    nposes = len(data['obj_poses_graph'])
    obj_poses_graph = np.zeros((nposes, 3))
    obj_poses_gt = np.zeros((nposes, 3))
    obj_poses_odom = np.zeros((nposes, 3))
    ee_poses_noisy = np.zeros((nposes, 3))

    for i in range(nposes):
        obj_poses_graph[i, :] = np.asarray(data['obj_poses_graph'][i][1])
        obj_poses_gt[i, :] = np.asarray(data['obj_poses_gt'][i][1])
        obj_poses_odom[i, :] = np.asarray(data['obj_poses_odom'][i][1])
        ee_poses_noisy[i, :] = np.asarray(data['ee_poses_noisy'][i][1])

    return (obj_poses_graph, obj_poses_gt, obj_poses_odom, ee_poses_noisy)


def draw_object(pose, params, color_val):
    yaw = pose[2]
    R = np.array([[np.cos(yaw), -np.sin(yaw)],
                  [np.sin(yaw), np.cos(yaw)]])
    offset = np.matmul(R, np.array(
        [[0.5*params['block_width']], [0.5*params['block_height']]]))
    rect = Rectangle((pose[0] - offset[0], pose[1] - offset[1]),
                     params['block_width'], params['block_height'],
                     angle=(np.rad2deg(yaw)), linewidth=2,
                     facecolor='None', edgecolor=color_val, alpha=0.75)
    return rect


def visualize_poses(dataset, ee_poses_noisy, poses1, poses2, poses3, poses4, dst_dir='', save_fig=False):

    nposes = poses1.shape[0]
    skip = 1

    params = dataset['params']
    ee_poses = dataset['ee_poses_2d']

    ds_idx = 0
    ds_skip = 50
    fig = plt.figure(figsize=(16, 12))

    save_fig = True
    if (save_fig):
        cmd = 'mkdir -p {0}'.format(dst_dir)
        os.popen(cmd, 'r')

    for idx in range(0, nposes, skip):

        plt.xlim((-0.6, 0.6))
        plt.ylim((-0.2, 0.8))

        # plot end-effector
        plt.plot(ee_poses[ds_idx][0], ee_poses[ds_idx]
                 [1], color='black', marker='o')
        circ = Circle((ee_poses[ds_idx][0], ee_poses[ds_idx][1]), params['ee_radius'],
                      facecolor='None', edgecolor='black', linestyle='-')

        plt.plot(ee_poses_noisy[idx][0], ee_poses_noisy[idx]
                 [1], color='grey', marker='o')
        circ = Circle((ee_poses_noisy[idx][0], ee_poses_noisy[idx][1]), params['ee_radius'],
                      facecolor='None', edgecolor='grey', linestyle='-')
        plt.gca().add_patch(circ)

        # plot object
        rect1 = draw_object(poses1[idx, :], params, 'green')
        rect2 = draw_object(poses2[idx, :], params, 'blue')
        rect3 = draw_object(poses3[idx, :], params, 'purple')
        rect4 = draw_object(poses4[idx, :], params, 'orange')

        plt.gca().add_patch(rect1)
        plt.gca().add_patch(rect2)
        plt.gca().add_patch(rect3)
        plt.gca().add_patch(rect4)

        plt.plot(poses1[1:idx, 0], poses1[1:idx, 1],
                 color='green', linestyle='-')
        plt.plot(poses2[1:idx, 0], poses2[1:idx, 1],
                 color='blue', linestyle='-')
        plt.plot(poses3[1:idx, 0], poses3[1:idx, 1],
                 color='purple', linestyle='-')
        plt.plot(poses4[1:idx, 0], poses4[1:idx, 1],
                 color='orange', linestyle='-')

        ds_idx = ds_idx + ds_skip

        plt.draw()
        plt.pause(1e-12)

        if(save_fig):
            plt.savefig('{0}/{1:06d}.png'.format(dst_dir, idx))

        plt.cla()


def write_video(img_dir, vidname):
    framerate = 10
    pattern = '*.png'
    cmd = 'ffmpeg -r {0} -pattern_type glob -i {1}/{2} -c:v libx264 {3}.mp4'.format(
        framerate, img_dir, pattern, vidname)
    print(cmd)
    os.popen(cmd, 'r')


def main():
    dataset_name = "logCircle1"
    std = 0.05
    srcdir = "../local/outputs/optimizer"
    dst_dir = "../local/outputs/optimizer/viz_results/{0}/std_{1}".format(
        dataset_name, std)

    filename = "../local/data/{0}.json".format(dataset_name)
    with open(filename) as f:
        dataset = json.load(f)

    nsteps = 300
    for tstep in range(nsteps-1, nsteps):

        filename = "{0}/contact/std_{1}/{2}UnconIter{3}.json".format(
            srcdir, std, dataset_name, str(tstep).zfill(3))
        poses_uncon, poses_gt, poses_odom, ee_poses_noisy = read_poses_json(
            filename)

        filename = "{0}/contact/std_{1}/{2}ConIter{3}.json".format(
            srcdir, std, dataset_name, str(tstep).zfill(3))
        poses_con, poses_gt, poses_odom, ee_poses_noisy = read_poses_json(
            filename)

        visualize_poses(dataset, ee_poses_noisy, poses_gt,
                        poses_odom, poses_uncon, poses_con, dst_dir, True)
        # write_video(dst_dir, dataset_name)


if __name__ == "__main__":
    main()
