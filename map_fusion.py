#conda install -c conda-forge -c davidcaron pclpy
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


mod = SourceModule("""
__global__ void findNearestNeighbors(float *source, float *target, int *result, int numSource, int numTarget)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numSource) {
        float minDist = 1e30;
        int closestPointIndex = -1;

        for (int i = 0; i < numTarget; ++i) {
            float dist = sqrtf((source[idx*3+0] - target[i*3+0]) * (source[idx*3+0] - target[i*3+0]) +
                               (source[idx*3+1] - target[i*3+1]) * (source[idx*3+1] - target[i*3+1]) +
                               (source[idx*3+2] - target[i*3+2]) * (source[idx*3+2] - target[i*3+2]));
            if (dist < minDist) {
                minDist = dist;
                closestPointIndex = i;
            }
        }
        result[idx] = closestPointIndex;
    }
}
""")

def find_nearest_neighbors_gpu(source_points, target_points):
    num_source = source_points.shape[0]
    num_target = target_points.shape[0]

    # Flatten points into 1D arrays (source and target points are (N, 3))
    source_points_flat = source_points.flatten().astype(np.float32)
    target_points_flat = target_points.flatten().astype(np.float32)
    
    # Allocate memory on GPU
    source_gpu = cuda.mem_alloc(source_points_flat.nbytes)
    target_gpu = cuda.mem_alloc(target_points_flat.nbytes)
    result_gpu = cuda.mem_alloc(num_source * np.int32().itemsize)

    # Copy data to GPU
    cuda.memcpy_htod(source_gpu, source_points_flat)
    cuda.memcpy_htod(target_gpu, target_points_flat)

    # Get kernel function
    func = mod.get_function("findNearestNeighbors")

    # Launch kernel
    func(source_gpu, target_gpu, result_gpu, np.int32(num_source), np.int32(num_target), block=(256, 1, 1), grid=(num_source // 256 + 1, 1))

    # Copy result back to CPU
    result = np.empty(num_source, dtype=np.int32)
    cuda.memcpy_dtoh(result, result_gpu)

    return result


def nearest_neighbors(source, target):
    # 计算源点与目标点之间的距离（欧氏距离）
    dist = np.linalg.norm(source[:, np.newaxis] - target, axis=2)
    nearest_idx = np.argmin(dist, axis=1)
    return nearest_idx

def compute_rigid_transform(source, target):
    # 计算质心
    centroid_source = np.mean(source, axis=0)
    centroid_target = np.mean(target, axis=0)

    # 去质心
    source_centered = source - centroid_source
    target_centered = target - centroid_target

    # 计算协方差矩阵 H
    H = source_centered.T @ target_centered

    # 计算 SVD
    U, _, Vt = np.linalg.svd(H)

    # 计算旋转矩阵
    R = Vt.T @ U.T

    # 特殊情况处理：如果旋转矩阵反射了点集，需要调整
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 计算平移向量
    t = centroid_target - R @ centroid_source

    return R, t

def ICP(src_cloud, tar_cloud, max_iter=100, tolerance=0.001):
    for i in range(max_iter):
        # 步骤 1：寻找源点集的最近邻点
        nearest_idx = find_nearest_neighbors_gpu(src_cloud, tar_cloud)

        # 步骤 2：根据最近邻点计算刚体变换
        matched_target = tar_cloud[nearest_idx]
        R, t = compute_rigid_transform(src_cloud, matched_target)

        # 步骤 3：应用刚体变换
        src_cloud = (R @ src_cloud.T).T + t

        # 计算当前误差（均方误差）
        error = np.mean(np.linalg.norm(src_cloud - matched_target, axis=1))

        print(f"第 {i+1} 次迭代, 误差: {error}")

        # 步骤 4：检查收敛条件
        if error < tolerance:
            print(f"ICP 收敛于第 {i+1} 次迭代，误差: {error}")
            break
    else:
        print(f"ICP 达到最大迭代次数 {max_iter}，误差: {error}")

    return R, t


#搜索用于icp的点云块
def radiusSearch(pc, point, radius):
    #在点云中搜索以point为中心半径为radius内的点
    pc_tree = KDTree(pc[:, :3])
    nearby_points = []
    indices = pc_tree.query_ball_point(point,radius)
    nearby_points = pc[indices]
    return nearby_points


def icpPointsPick(pc1, pc2, gps1, gps2, pair_num, radius):
    #构建地图2的gps点KD-Tree
    gps_tree = KDTree(gps2)
    #对每个gps1的点，在gps2中找到最近邻点，返回距离和索引
    distances, indices = gps_tree.query(gps1, k=1)
    #存储距离和索引
    closest_points = [(i, indices[i], distances[i]) for i in range(len(gps1))]
    #排序取需要的前n对索引
    min_dis = 0  # 预设的最小距离阈值
    filtered_points = [p for p in closest_points if p[2] > min_dis]
    closest_points_sorted = sorted(filtered_points, key=lambda x: x[2])[:pair_num]

    #取近邻gps点连线中点
    mid_points = []
    for i in range(len(closest_points_sorted)):
        idx1 = closest_points_sorted[i][0]
        idx2 = closest_points_sorted[i][1]
        print("distance:",closest_points_sorted[i][2])
        mid_points.append((gps1[idx1]+gps2[idx2]) / 2)
    mid_points = np.asarray(mid_points)

    print(f"finding {search_pair_num} couples...")
    #对每个mid_point,在pc1和pc2中找到对应的范围内点云
    icp_pc = []
    for i in range(len(mid_points)):
        #对pc1初始点去偏置并搜索
        mid_fix = mid_points[i] - gps1[0]
        icp_pc1 = radiusSearch(pc1,mid_fix,radius)
        icp_pc2 = radiusSearch(pc2,mid_fix,radius)
        icp_pc.append([icp_pc1,icp_pc2])
        print(' '*20, end='\r')
        print(f"{i+1}-{pair_num}", end='\r')
    
    ################### 保存点云到 .bin 文件
    # 指定保存路径
    save_path = "/home/myz/catkin_ws_lvi2.0/src/map_fusion/myz"

    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)
    print(f"saving bin...")
    dada = icp_pc[0][0]
    print(f"dimension: {dada.shape}")
    for i, (pc1_chunk, pc2_chunk) in enumerate(icp_pc):
        # 保存 pc1 的局部点云
        pc1_filename = os.path.join(save_path, f"{i :06d}.bin")  # 确保是六位数
        pc1_chunk.astype(np.float32).tofile(pc1_filename)
        # 保存 pc2 的局部点云
        pc2_filename = os.path.join(save_path, f"{i + search_pair_num:06d}.bin")  # 确保是六位数
        pc2_chunk.astype(np.float32).tofile(pc2_filename)
        print(' '*20, end='\r')
        print(f"{(i+1)*2}-{total_num}", end='\r')
    print(f"std bin Saved !!!!!")

    return icp_pc



#点云转换成numpy数组
def pcToArray(point_cloud):
    point_list = []
    for point in point_cloud.points:
        point_list.append([point.x, point.y, point.z, point.intensity])
    point_array = np.asarray(point_list)
    return point_array


#numpy数组转换成点云
def arrayToPC(point_array, origin_cloud):
    for i in range(len(point_array)):
        origin_cloud.points[i].x = point_array[i][0]
        origin_cloud.points[i].y = point_array[i][1]
        origin_cloud.points[i].z = point_array[i][2]
        origin_cloud.points[i].intensity = point_array[i][3]
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
    #文件路径参数
    pcd1_path = "./map1/GlobalMap.pcd"
    pcd2_path = "./map2/GlobalMap.pcd"
    odom1_path = "./odom1.txt"
    odom2_path = "./odom2.txt"
    gps1_path = "./gps1.txt"
    gps2_path = "./gps2.txt"
    path1 = "./path1.txt"
    path2 = "./path2.txt"
    fusion_save_path = "colormap.pcd"

    #算法参数
    icp_max_iter = 10000
    icp_tolerance = 0.001
    search_pair_num = 50
    total_num = search_pair_num * 2
    search_radius = 15 #15good
    voxel_size = 4
    
    #获取转换为绝对xy坐标的原始gps数据
    origin_gps1 = gfx.gpsfixer(gps1_path)
    origin_gps2 = gfx.gpsfixer(gps2_path)

    #获取建图得到的原始odom数据
    origin_path1 = gfx.pathfixer(path1)
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


    # 提取 xyz 和 intensity #myz
    pc1_xyz = pc1_array[:, :3]  # 取前三列
    pc2_xyz = pc2_array[:, :3]
    pc1_intensity = pc1_array[:, 3]  # 取第4列
    pc2_intensity = pc2_array[:, 3]

    # 对 xyz 进行旋转和偏移
    pc1_xyz = (R1 @ pc1_xyz.T).T
    pc2_xyz = (R2 @ pc2_xyz.T).T

    #用gps初始点计算地图位置偏移，应用到第二张地图
    pc2_xyz[:, 0] += origin_gps2[0][0] - origin_gps1[0][0]
    pc2_xyz[:, 1] += origin_gps2[0][1] - origin_gps1[0][1]
    pc2_xyz[:, 2] += origin_gps2[0][2] - origin_gps1[0][2]

    # 合并 xyz 和 intensity #myz
    pc1_array = np.hstack((pc1_xyz, pc1_intensity.reshape(-1, 1)))
    pc2_array = np.hstack((pc2_xyz, pc2_intensity.reshape(-1, 1)))

    #将初变换后点云转回pcl格式
    pc1_fixed = arrayToPC(pc1_array, pc1)
    pc2_fixed = arrayToPC(pc2_array, pc2)

    #点云体素化下采样用于优化ICP结果
    pc1_ds = pcl.PointCloud.PointXYZI()
    pc2_ds = pcl.PointCloud.PointXYZI()

    # pcl_voxel = pcl.filters.VoxelGrid.PointXYZI()
    # pcl_voxel.setLeafSize(voxel_size, voxel_size, voxel_size)
    # pcl_voxel.setInputCloud(pc1_fixed)
    # pcl_voxel.filter(pc1_ds)
    # pcl_voxel.setInputCloud(pc2_fixed)
    # pcl_voxel.filter(pc2_ds)

    pc1_ds = pclpy.octree_voxel_downsample(pc1_fixed, 4)
    pc2_ds = pclpy.octree_voxel_downsample(pc2_fixed, 4)

    pc1_ds_array = pcToArray(pc1_ds)
    pc2_ds_array = pcToArray(pc2_ds)

    #FPFH特征提取
    fpfh = pcl.features.FPFHEstimation.PointXYZ_Normal_FPFHSignature33()



    #icp点云块选取
    #icp_pc = icpPointsPick(pc1_ds_array, pc2_ds_array, origin_gps1, origin_gps2, search_pair_num, search_radius)
    icp_pc = icpPointsPick(pc1_array, pc2_array, origin_gps1, origin_gps2, search_pair_num, search_radius)#不用下采样
    print("len_icp:",len(icp_pc))
    #print(icp_pc)

    #对获取的点云对分别做icp，获取多组变换结果，取平均
    transformsR = []
    transformsT = []
    for i in range(len(icp_pc)):
        print("计算第",i+1,"对点云ICP......")
        R, t = ICP(icp_pc[i][1], icp_pc[i][0], icp_max_iter, icp_tolerance)
        print(t)
        transformsR.append(R)
        transformsT.append(t)
    R = np.zeros((3, 3))
    t = np.zeros((1, 3))
    for i in range(len(transformsR)):
        R += transformsR[i]
        t += transformsT[i]
    R /= len(transformsR)
    t /= len(transformsT)

    #将变换应用到第二张地图
    pc2_array = (R @ pc2_array.T).T + t

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
        color_pc.points.append(color_p)
    
    for i in range(len(pc2.points)):
        color_p = pcl.point_types.PointXYZRGB()
        color_p.x = pc2.points[i].x
        color_p.y = pc2.points[i].y
        color_p.z = pc2.points[i].z
        color_p.r = 0
        color_p.g = 0
        color_p.b = 255
        color_pc.points.append(color_p)
    
    color_pc.width = len(color_pc.points)
    color_pc.height = 1

    pcl.io.savePCDFileASCII(fusion_save_path, color_pc)
    print("color map Saved")

