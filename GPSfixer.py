import re
import math
import pyproj
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import procrustes



#对齐odom和gps，获取对齐所需变换
def pathsAlign(odom, gps):
    global showFig
    gps0 = np.array([[row[0]-gps[0][0],row[1]-gps[0][1],row[2]-gps[0][2]] for row in gps])
    H = odom.T @ gps0  # 协方差矩阵
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    print(f"pathsAlign:\n{R}")

    odomfix = (R @ odom.T).T

    if showFig:
        X1 = odomfix[:,0]
        Y1 = odomfix[:,1]
        X2 = gps0[:,0]
        Y2 = gps0[:,1]

        #绘图
        fig,ax = plt.subplots()
        ax.plot(X1,Y1)
        ax.plot(X2,Y2)
        ax.set_aspect('equal')
        plt.show()

    return R, gps0
    

#分别对odom和gps路径做插值下采样
def pathDownsample(path, target_points_num):
    path = np.array(path)
    N, D = path.shape

    original_indices = np.linspace(0, 1, N)
    target_indices = np.linspace(0, 1, target_points_num)
    
    interpolated_path = []
    for d in range(D):
        interp_func = interp1d(original_indices, path[:, d], kind='linear')
        interpolated_path.append(interp_func(target_indices))
    
    downsampled_path = np.stack(interpolated_path, axis=1)
    return downsampled_path

#经纬度转XY坐标
def millerToXY (lon, lat):
    xy_coordinate = []
    #地球周长
    L = 6381372*math.pi*2 
    #平面展开，将周长视为X轴
    W = L 
    #Y轴约等于周长一般
    H = L/2 
    #米勒投影中的一个常数，范围大约在正负2.3之间
    mill = -2.3 
    #将经度从度数转换为弧度
    x = lon*math.pi/180 
    # 将纬度从度数转换为弧度
    y = lat*math.pi/180 
    #这里是米勒投影的转换
    y = 1.25*math.log(math.tan(0.25*math.pi+0.4*y)) 
    # 这里将弧度转为实际距离 ，转换结果的单位是公里
    x = (W/2)+(W/(2*math.pi))*x
    y = (H/2)-(H/(2*mill))*y

    return x, y



#draw
def drawfigure(X, Y):
    global showFig
    if showFig:
        fig,ax = plt.subplots()
        ax.plot(X,Y)
        ax.set_aspect('equal')
        plt.show()

#读取odom文件，获取XY坐标
def odomfixer(filepath):
    global ds_num
    sgOdom = []
    Odoms = []

    with open(filepath, "r") as file:
        getData = False
        for line in file.readlines():
            if "position:" in line:
                getData = True
                continue
            elif "orientation:" in line:
                getData = False
                Odoms.append(sgOdom)
                sgOdom = []
                continue
            if getData:
                num = re.search(r'\d+\.\d+', line).group()
                sgOdom.append(float(num))

    Odoms = pathDownsample(Odoms, ds_num)
    print(f"len(odoms):{len(Odoms)}")

    X = Odoms[:,0]
    Y = Odoms[:,1]
    drawfigure(X, Y)

    return Odoms

#读取gps文件，获取经纬度和转换后的XY坐标
def gpsfixer(filepath):
    global ds_num
    sgGPS = []
    GPSs = []

    with open(filepath, "r") as file:
        for line in file.readlines():
            #判断是否为经纬度和海拔行
            if line.startswith("latitude") or line.startswith("longitude") or line.startswith("altitude"):
                num = re.search(r'\d+\.\d+', line).group()
                sgGPS.append(float(num))
            elif line.startswith("---"):
                GPSs.append(sgGPS)
                print(sgGPS)
                sgGPS = []
    
    fixedGPS = []

    for i, gps in enumerate(GPSs):
        x, y = millerToXY(gps[1], gps[0])
        fixedGPS.append([x, y, gps[2]])

    #fixedGPS = pathDownsample(fixedGPS, len(fixedGPS)//100*100)
    fixedGPS = pathDownsample(fixedGPS, ds_num)
    print(f"fixedGPS.shape:{fixedGPS.shape}")

    X = fixedGPS[:,0]
    Y = fixedGPS[:,1]
    drawfigure(X, Y)

    return fixedGPS


def pathfixer(filepath):
    global ds_num
    path = []
    Paths = []
    with open(filepath, "r") as file:
        for line in file.readlines():
            pose = line.split()
            x, y, z = float(pose[0]), float(pose[1]), float(pose[2])
            #print(x, y, z)
            path.append(x)
            path.append(y)
            path.append(z)
            Paths.append(path)
            path = []
    
    Paths = pathDownsample(Paths, ds_num)
    print(f"Paths.shape:{Paths.shape}")

    X = Paths[:,0]
    Y = Paths[:,1]
    drawfigure(X, Y)

    return Paths

if __name__ == "__main__":

    filepath = "gps1.txt"
    filepath2 = "path1.txt"
    showFig = False
    ds_num = 1000

    gps = gpsfixer(filepath)
    #odom = odomfixer(filepath2)
    path = pathfixer(filepath2)
    pathsAlign(path, gps)
