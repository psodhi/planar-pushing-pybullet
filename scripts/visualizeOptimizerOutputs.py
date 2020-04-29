import pybullet as pb
import numpy as np
import json

import os
from subprocess import call
from shapely.geometry import Point, Polygon

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

import pdb

plt.rcParams.update({'font.size': 22})

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

def transform_to_frame2d(pt, frame_pose_2d):

    yaw = frame_pose_2d[2, 0]
    R = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
    t = -frame_pose_2d[0:2]
    pt_tf = np.matmul(R, pt+t)

    return pt_tf


def visualize_poses__object(dataset, ee_poses_noisy, poses1, poses2, poses3, poses4, figfile='', save_fig=False):
    params = dataset['params']

    poly_obj = Polygon([(-0.5*params['block_width'], -0.5*params['block_height']),
                        (-0.5*params['block_width'],
                         0.5*params['block_height']),
                        (0.5*params['block_width'],
                         0.5*params['block_height']),
                        (0.5*params['block_width'], -0.5*params['block_height'])])
    plt.xlim((-0.2, 0.2))
    plt.ylim((-0.25, 0.25))

    ee_center__world = (ee_poses_noisy[-1, 0:2][None]).transpose()
    frame_pose1_2d = (poses1[-1, :][None]).transpose()
    frame_pose2_2d = (poses2[-1, :][None]).transpose()
    frame_pose3_2d = (poses3[-1, :][None]).transpose()
    frame_pose4_2d = (poses4[-1, :][None]).transpose()
    ee_center1__obj = transform_to_frame2d(ee_center__world, frame_pose1_2d)
    ee_center2__obj = transform_to_frame2d(ee_center__world, frame_pose2_2d)
    ee_center3__obj = transform_to_frame2d(ee_center__world, frame_pose3_2d)
    ee_center4__obj = transform_to_frame2d(ee_center__world, frame_pose4_2d)

    # plt.plot(ee_center1__obj[0], ee_center1__obj[1], color='green', marker='o')
    # circ1 = Circle((ee_center1__obj[0], ee_center1__obj[1]), params['ee_radius'],
                #    facecolor='None', edgecolor='green', linestyle='-', linewidth=2, alpha=0.75)
    # plt.plot(ee_center2__obj[0], ee_center2__obj[1], color='blue', marker='o')
    # circ2 = Circle((ee_center2__obj[0], ee_center2__obj[1]), params['ee_radius'],
    #   facecolor='None', edgecolor='blue', linestyle='-', linewidth=2, alpha=0.75)
    plt.plot(ee_center3__obj[0], ee_center3__obj[1],
             color='purple', marker='o')
    circ3 = Circle((ee_center3__obj[0], ee_center3__obj[1]), params['ee_radius'],
                   facecolor='None', edgecolor='purple', linestyle='-', linewidth=2, alpha=0.75)
    plt.plot(ee_center4__obj[0], ee_center4__obj[1],
             color='orange', marker='o')
    circ4 = Circle((ee_center4__obj[0], ee_center4__obj[1]), params['ee_radius'],
                   facecolor='None', edgecolor='orange', linestyle='-', linewidth=2, alpha=0.75)

    # plt.gca().add_patch(circ1)
    # plt.gca().add_patch(circ2)
    plt.gca().add_patch(circ3)
    plt.gca().add_patch(circ4)

    x, y = poly_obj.exterior.xy
    plt.plot(x, y, color='black', linewidth=2)

    plt.draw()
    plt.pause(1e-12)
    
    if(save_fig):
        plt.savefig(figfile)

    plt.cla()

def draw_object__world(pose, params, color_val):
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

def visualize_poses__world(dataset, ee_poses_noisy, poses1, poses2, poses3, poses4, figfile, save_fig):

    params = dataset['params']

    plt.xlim((-0.6, 0.6))
    plt.ylim((-0.2, 0.8))
    
    # plt.text(-0.5, 0.7, "x{0}: {1}".format(idx, contact_flag[ds_idx][0]))
    # plot end-effector
    # plt.plot(ee_poses[ds_idx][0], ee_poses[ds_idx]
    #          [1], color='black', marker='o')
    # circ = Circle((ee_poses[ds_idx][0], ee_poses[ds_idx][1]), params['ee_radius'],
    #               facecolor='None', edgecolor='black', linestyle='-')

    plt.plot(ee_poses_noisy[-1][0], ee_poses_noisy[-1]
             [1], color='black', marker='o')
    circ = Circle((ee_poses_noisy[-1][0], ee_poses_noisy[-1][1]), params['ee_radius'],
                  facecolor='None', edgecolor='black', linestyle='-', linewidth='2')
    plt.gca().add_patch(circ)

    # # plot object
    # rect1 = draw_object__world(poses1[-1, :], params, 'green')
    # rect2 = draw_object__world(poses2[-1, :], params, 'blue')
    rect3 = draw_object__world(poses3[-1, :], params, 'purple')
    rect4 = draw_object__world(poses4[-1, :], params, 'orange')

    # plt.gca().add_patch(rect1)
    # plt.gca().add_patch(rect2)
    plt.gca().add_patch(rect3)
    plt.gca().add_patch(rect4)

    plt.plot(poses1[0:-1, 0], poses1[0:-1, 1],
             color='green', linestyle='-', label='groundtruth')
    plt.plot(poses2[0:-1, 0], poses2[0:-1, 1],
             color='blue', linestyle='-', label='odometry')
    plt.plot(poses3[0:-1, 0], poses3[0:-1, 1],
             color='purple', linestyle='-', label='iSAM2 (uncon)')
    plt.plot(poses4[0:-1, 0], poses4[0:-1, 1],
             color='orange', linestyle='-', label='iSAM2 (con)')
    plt.legend(loc="upper left")

    
    plt.draw()
    plt.pause(1e-12)

    if(save_fig):
        plt.savefig(figfile)

    plt.cla()


def write_video(imgsrcdir, viddst):

    framerate = 10    
    cmd = 'ffmpeg -y -r {0} -pattern_type glob -i '{1}/*.png' -c:v libx264 {2}.mp4'.format(
        framerate, imgsrcdir, viddst)
    call(cmd, shell=True)

    cmd = 'convert -quality 0.1 -layers Optimize -delay {0} -loop 0 {1}/*.png {2}.gif'.format(
        framerate, imgsrcdir, viddst)
    call(cmd, shell=True)

def main():
    dataset_name = "logCircle1"  # logCircle1, logLine1
    std_contact = 0.05
    srcdir = '../local/outputs/optimizer'

    viz_frame = 'world'  # world, object
    dstdir = '../local/outputs/optimizer/viz_results/{0}/std_{1}/{2}'.format(
        dataset_name, std_contact, viz_frame)

    filename = "../local/data/{0}.json".format(dataset_name)
    with open(filename) as f:
        dataset = json.load(f)

    fig = plt.figure(figsize=(16, 12))
    save_fig = True
    if (save_fig):
        cmd = "mkdir -p {0}".format(dstdir)
        os.popen(cmd, 'r')

    nsteps = 100
    for tstep in range(1, nsteps):

        filename = '{0}/contact/std_{1}/{2}UnconIter{3}.json'.format(
            srcdir, std_contact, dataset_name, str(tstep).zfill(3))
        poses_uncon, poses_gt, poses_odom, ee_poses_noisy = read_poses_json(
            filename)

        filename = '{0}/contact/std_{1}/{2}ConIter{3}.json'.format(
            srcdir, std_contact, dataset_name, str(tstep).zfill(3))
        poses_con, poses_gt, poses_odom, ee_poses_noisy = read_poses_json(
            filename)

        figfile = '{0}/{1:06d}.png'.format(dstdir, tstep)
        if (viz_frame == 'world'):
            visualize_poses__world(dataset, ee_poses_noisy, poses_gt,
                                   poses_odom, poses_uncon, poses_con, figfile, True)
        elif (viz_frame == 'object'): 
            if (tstep < 30): # hack: to skip contact_flag=0 poses
                continue
            visualize_poses__object(dataset, ee_poses_noisy, poses_gt,
                                 poses_odom, poses_uncon, poses_con, figfile, True)

    std_odom = 0.005
    vidname = '{0}_stdodom_{1}_stdcontact_{2}'.format(dataset_name, std_odom, std_contact)
    viddst = '../local/outputs/optimizer/viz_results/{0}'.format(vidname)
    write_video(dstdir, viddst)


if __name__ == "__main__":
    main()
