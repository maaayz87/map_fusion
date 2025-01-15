import pclpy.pcl as pcl
import pclpy
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import open3d as o3d 

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import GPSfixer as gfx
import os



if __name__ == '__main__':
    #文件路径参数
    path1 = "./path1.txt"
    path2 = "./path2.txt"
    gps1 = "./gps1.txt"
    gps2 = "./gps2.txt"


    #获取转换为绝对xy坐标的原始gps数据
    origin_gps1 = gfx.gpsfixer(gps1)
    origin_gps2 = gfx.gpsfixer(gps2)

    gps1_endpoint = origin_gps1[-1]
    gps2_startpoint = origin_gps2[0]
    translation =  gps2_startpoint - gps1_endpoint
    print(f"translation:{translation}")

    #获取建图得到的原始odom数据
    origin_path1 = gfx.pathfixer(path1)
    # origin_path1[:, 0] = -origin_path1[:, 0]
    # origin_path1[:, 1] = -origin_path1[:, 1]

    origin_path2 = gfx.pathfixer(path2)

    #获取地图对齐到gps坐标的旋转
    R1, fixed_gps1 = gfx.pathsAlign(origin_path1, origin_gps1)
    R2, fixed_gps2 = gfx.pathsAlign(origin_path2, origin_gps2)

    fixed_path1 = (R1 @ origin_path1.T).T
    fixed_path2 = (R2 @ origin_path2.T).T

    #根据gps信息拼接。两个变换（gps2起点和1终点的变换 和 gps1始末变换）
    trans_offset = fixed_gps1[-1] - fixed_gps1[0]
    fixed_path2 = fixed_path2 + translation + trans_offset
    fixed_gps2 = fixed_gps2 + translation + trans_offset


    X1,Y1,Z1 = fixed_path1[:,0], fixed_path1[:,1], fixed_path1[:,2]
    X2,Y2,Z2 = fixed_path2[:,0], fixed_path2[:,1], fixed_path2[:,2]
    #粗匹配效果
    fig,ax = plt.subplots()
    ax.plot(X1,Y1)
    ax.plot(X2,Y2)
    ax.set_aspect('equal')
    plt.show()

    #应用std，精匹配效果
    R = np.array([
      [0.997193, 0.0740147, -0.011311],
      [-0.0730337, 0.994807, 0.0708822],
      [0.0164986, -0.0698571, 0.997421]
    ])
    t = np.array([-5.43945, 1.05476, -1.20451])
    fixed_path2 = (R @ fixed_path2.T).T + t

    X2,Y2,Z2 = fixed_path2[:,0], fixed_path2[:,1], fixed_path2[:,2]
    print("after STD")
    fig,ax = plt.subplots()
    ax.plot(X1,Y1)
    ax.plot(X2,Y2)
    ax.set_aspect('equal')
    plt.show()

    #绘图
    # fig,ax = plt.subplots()
    # ax.plot(X1,Z1)
    # ax.plot(X2,Z2)

    # plt.show()


    #gps效果
    X1,Y1,Z1 = fixed_gps1[:,0], fixed_gps1[:,1], fixed_gps1[:,2]
    X2,Y2,Z2 = fixed_gps2[:,0], fixed_gps2[:,1], fixed_gps2[:,2]

    #绘图
    fig,ax = plt.subplots()
    ax.plot(X1,Y1)
    ax.plot(X2,Y2)
    ax.set_aspect('equal')
    plt.show()

    fig,ax = plt.subplots()
    ax.plot(X1,Z1)
    ax.plot(X2,Z2)
    plt.show()

