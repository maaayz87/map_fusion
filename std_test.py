#conda install -c conda-forge -c davidcaron pclpy

import pclpy.pcl as pcl
import pclpy
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt


import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import GPSfixer as gfx
import os





#点云转换成numpy数组
def pcToArray(point_cloud):
    point_list = []
    for point in point_cloud.points:
        point_list.append([point.x, point.y, point.z])
    point_array = np.asarray(point_list)
    return point_array


#numpy数组转换成点云
def arrayToPC(point_array, origin_cloud):
    for i in range(len(point_array)):
        origin_cloud.points[i].x = point_array[i][0]
        origin_cloud.points[i].y = point_array[i][1]
        origin_cloud.points[i].z = point_array[i][2]
    return origin_cloud


#读取pcd文件为pcl点云
def pcdHandler(pcd_path, IorR):
    if(IorR):
        #带intensity的激光点云
        point_cloud = pcl.PointCloud.PointXYZI()
    else:
        #带RGB的视觉特征点云
        point_cloud = pcl.PointCloud.PointXYZRGB()

    pcdreader = pcl.io.PCDReader()

    print("Reading PCD file: ",pcd_path)
    pcdreader.read(pcd_path, point_cloud)

    return point_cloud


if __name__ == '__main__':

    isVimap = False

    pcd1_path = "./map1/CornerMap.pcd"
    pcd2_path = "./map2/CornerMap.pcd"
    if (isVimap):
        pcd1_path = "./map1/viMap.pcd"
        pcd2_path = "./map2/viMap.pcd"
        print("isVimap!")
    odom1_path = "./odom1.txt"
    odom2_path = "./odom2.txt"
    gps1_path = "./gps1.txt"
    gps2_path = "./gps2.txt"
    path1 = "./path1.txt"
    path2 = "./path2.txt"
    fusion_save_path = "afterSTD.pcd"

    
    origin_gps1 = gfx.gpsfixer(gps1_path)
    origin_path1 = gfx.pathfixer(path1)

    origin_gps2 = gfx.gpsfixer(gps2_path)
    origin_path2 = gfx.pathfixer(path2)


    #获取地图对齐到gps坐标的旋转
    R1, fixed_gps1 = gfx.pathsAlign(origin_path1, origin_gps1)
    R2, fixed_gps2 = gfx.pathsAlign(origin_path2, origin_gps2)

    #获取原始区域点云地图
    pc1 = pcdHandler(pcd1_path, True)
    pc2 = pcdHandler(pcd2_path, True)

    #先转换成numpy数组做变换处理
    pc1_array = pcToArray(pc1)
    pc2_array = pcToArray(pc2)

    #点云旋转对齐GPS坐标
    pc1_array = (R1 @ pc1_array.T).T
    pc2_array = (R2 @ pc2_array.T).T

    #用gps初始点计算地图位置偏移，应用到第二张地图
    pc2_array[:, 0] += origin_gps2[0][0] - origin_gps1[0][0]
    pc2_array[:, 1] += origin_gps2[0][1] - origin_gps1[0][1]
    pc2_array[:, 2] += origin_gps2[0][2] - origin_gps1[0][2]

    ################
    R = np.array([
      [0.996033, 0.0855639, 0.0244253],
      [-0.0849626, 0.996079, -0.0246771],
      [-0.026441, 0.022504, 0.999397]
    ])
    t = np.array([-5.20736, 1.24065, 1.13551])

    #将变换应用到第二张地图
    pc2_array = (R @ pc2_array.T).T + t

    ################




    #将变换后的点云转换回pcl点云
    pc1 = arrayToPC(pc1_array, pc1)
    pc2 = arrayToPC(pc2_array, pc2)

    #效果展示用文件
    color_pc = pcl.PointCloud.PointXYZRGB()
    color_pc.header = pc1.header

    for i in range(len(pc1.points)):
        color_p = pcl.point_types.PointXYZRGB()
        color_p.x = pc1.points[i].x
        color_p.y = pc1.points[i].y
        color_p.z = pc1.points[i].z
        color_p.r = 255
        color_p.g = 0
        color_p.b = 0
        if (isVimap):
            color_p.r = pc1.points[i].r
            color_p.g = pc1.points[i].g
            color_p.b = pc1.points[i].b           
        color_pc.points.append(color_p)
    
    for i in range(len(pc2.points)):
        color_p = pcl.point_types.PointXYZRGB()
        color_p.x = pc2.points[i].x
        color_p.y = pc2.points[i].y
        color_p.z = pc2.points[i].z
        color_p.r = 0
        color_p.g = 0
        color_p.b = 255
        if (isVimap):
            color_p.r = pc2.points[i].r
            color_p.g = pc2.points[i].g
            color_p.b = pc2.points[i].b  
        color_pc.points.append(color_p)
    
    color_pc.width = len(color_pc.points)
    color_pc.height = 1

    #输出点云pcd文件
    pcl.io.savePCDFileASCII(fusion_save_path, color_pc)
    print("after STD Saved")
