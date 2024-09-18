import numpy as np
import laspy
import CSF
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def csf_las(file, outfile):
    las = laspy.read(file)
    points = las.points
    xyz = np.vstack((las.x, las.y, las.z)).transpose()  # 点云的空间位置

    csf = CSF.CSF()
    csf.params.bSloopSmooth = False  # 粒子设置为不可移动
    csf.params.cloth_resolution = 0.1  # 布料网格分辨率2
    csf.params.rigidness = 3  # 布料刚性参数2
    csf.params.time_step = 0.65  # 步长
    csf.params.class_threshold = 0.03  # 点云与布料模拟点的距离阈值0.5
    csf.params.interations = 500  # 最大迭代次数500
    csf.setPointCloud(xyz)
    ground = CSF.VecInt()  # 地面点索引列表
    non_ground = CSF.VecInt()  # 非地面点索引列表
    csf.do_filtering(ground, non_ground)  # 执行滤波函数

    # 保存地面点
    out_file = laspy.LasData(las.header)
    out_file.points = points[np.array(ground)]
    out_file.write(outfile)

    return xyz[ground], xyz[non_ground]

# 使用最小二乘法拟合平面，返回平面系数
def fit_plane(x, y, z):
    A = np.c_[x, y, np.ones(x.shape[0])]
    C, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    return C

# 计算每个点到平面的距离
def distance_from_plane(C, x, y, z):
    return np.abs(C[0] * x + C[1] * y + C[2] - z) / np.sqrt(C[0]**2 + C[1]**2 + C[2]**2)

# 保留每组中最低的Z值点
def keep_lowest_z_points(las, precision=1):
    x_coords = np.round(las.x, precision)
    y_coords = np.round(las.y, precision)
    z_coords = las.z

    # 创建一个字典，根据(X, Y)坐标对点进行分组
    groups = {}
    for i in range(len(x_coords)):
        key = (x_coords[i], y_coords[i])
        if key not in groups:
            groups[key] = [(z_coords[i], i)]
        else:
            groups[key].append((z_coords[i], i))

    # 找到每个分组中Z值最小的点的索引
    min_z_indices = [min(group, key=lambda x: x[0])[1] for group in groups.values()]

    # 使用这些索引创建一个新的点数组
    filtered_points = las.points[min_z_indices]
    return filtered_points


# 应用过滤器，仅保留形成平面的点
def remove_higher_z_points_and_keep_plane(las, output_file_path, precision=1, distance_threshold=0.1):
    filtered_points = keep_lowest_z_points(las, precision)
    x_coords = np.round(filtered_points['X'] * las.header.scale[0] + las.header.offset[0], precision)
    y_coords = np.round(filtered_points['Y'] * las.header.scale[1] + las.header.offset[1], precision)
    z_coords = filtered_points['Z'] * las.header.scale[2] + las.header.offset[2]

    C = fit_plane(x_coords, y_coords, z_coords)
    print(f"平面系数: {C}")
    print(f"拟合的平面方程: z = {C[0]} * x + {C[1]} * y + {C[2]}")

    distances = distance_from_plane(C, x_coords, y_coords, z_coords)

    within_threshold = distances < distance_threshold
    print(f"在阈值内的点数: {np.sum(within_threshold)}")

    filtered_points = filtered_points[within_threshold]

    new_las = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
    new_las.points = filtered_points

    new_las.header.offsets = las.header.offsets
    new_las.header.scales = las.header.scales

    new_las.write(output_file_path)
    print(f"拟合后的平面已写入 {output_file_path}")

    return x_coords, y_coords, z_coords, C

def plot_fit(x, y, z, C):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, color='b', marker='.')

    xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100))
    zz = C[0] * xx + C[1] * yy + C[2]

    ax.plot_surface(xx, yy, zz, color='r', alpha=0.5)

    plt.title('Fitted Plane')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def detect_objects_and_save_separately(las_file_path, a, b, d, threshold_reflectance, eps, min_samples,
                                       output_base_path):
    las = laspy.read(las_file_path)

    x = np.array(las.x)
    y = np.array(las.y)
    z = np.array(las.z)
    reflectance = np.array(las.intensity)

    z_fit = a * x + b * y + d
    is_above_ground = (z > z_fit) & (reflectance > threshold_reflectance)

    points_to_cluster = np.vstack((x[is_above_ground], y[is_above_ground], z[is_above_ground])).T

    # 使用DBSCAN进行聚类
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points_to_cluster)
    labels = db.labels_

    # 过滤出非噪声点
    unique_labels = set(filter(lambda x: x >= 0, labels))

    for label in unique_labels:
        cluster = points_to_cluster[labels == label]
        
       
        
        cluster_reflectance = reflectance[is_above_ground][labels == label].astype(np.uint16)

        # 可以选择使用反射率的平均值或其他统计量
        cluster_intensity_mean = cluster_reflectance.mean()

        # 构建输出文件名
        output_file_path = f"{output_base_path}_cluster_{label}.las"

        # 保存到新的LAS文件
        header = laspy.LasHeader(point_format=las.header.point_format, version=las.header.version)
        object_las = laspy.LasData(header)
        object_las.x = cluster[:, 0]
        object_las.y = cluster[:, 1]
        object_las.z = cluster[:, 2]
        # 这里可以选择将平均反射率赋值给所有点，或者将原始反射率赋值（如果它们不同）
        object_las.intensity = cluster_reflectance  # 或者 object_las.intensity = np.full(cluster.shape[0], cluster_intensity_mean, dtype=np.uint16)
        object_las.write(output_file_path)
        print(f"Saved {cluster.shape[0]} points from cluster {label} to {output_file_path}")
        
     

if __name__ == '__main__':
    
    input_file_path = r'data\pp_data\2022-06-10 10-06-22_pp.las' # 干净背景
    output_file_path = r'tmp\tiqudimian.las' # 经csf提取后的输出文件
    csf_las(input_file_path, output_file_path)
    
    las = laspy.read(r"tmp\tiqudimian.las")
    x_coords,y_coords,z_coords,C = remove_higher_z_points_and_keep_plane(las,r"tmp\tiqudimian_nihe.las",precision=1, distance_threshold=0.1 )
    plot_fit(x_coords, y_coords, z_coords, C)
    
    output_base_path = r"result/" # 聚类结果输出后保存到的文件夹
    detect_objects_and_save_separately(las_file_path=r"data\pp_data\direction0-1cm-50m_pp.las", a=C[0], b=C[1], d=C[2], eps=0.2,
                                       min_samples=10, threshold_reflectance=10, output_base_path=output_base_path)
