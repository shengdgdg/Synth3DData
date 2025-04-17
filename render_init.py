import argparse
import sys
import os
import json
import time
import random
import bpy
import math
import OpenEXR
import Imath
import numpy as np
import matplotlib.pyplot as plt
from mathutils import Vector, Matrix, Euler
from scipy.ndimage import distance_transform_edt


def configure_render(output_dir: str, config: dict):
    """配置渲染参数"""
    render = bpy.context.scene.render
    render.engine = config.get('engine', 'BLENDER_EEVEE_NEXT')
    render.filepath = output_dir

    # 根据引擎类型配置参数
    if render.engine == 'CYCLES':
        cycles = bpy.context.scene.cycles
        cycles.samples = config.get('cycles_samples', 128)  # 采样数
        cycles.use_denoising = config.get('use_denoising', True)  # 去噪
        cycles.device = config.get('device', 'GPU')  # 设备选择
        cycles.max_bounces = config.get('max_bounces', 12)  # 光线反弹次数
        cycles.transparent_max_bounces = config.get('transparent_bounces', 8)  # 透明反弹次数

    elif render.engine == 'BLENDER_EEVEE_NEXT':
        eevee = bpy.context.scene.eevee
        # 采样设置
        eevee.taa_render_samples = config.get('taa_render_samples', 1)  # 渲染采样（1 - 64）
        eevee.taa_samples = config.get('eevee_taa_samples', 1)  # 抗锯齿采样（1 - 64）
        eevee.use_gtao = config.get('use_gtao', True)  # 全局光照

    # 分辨率设置
    resolution = config.get('resolution', (384, 672))
    render.resolution_x = resolution[1]
    render.resolution_y = resolution[0]
    bpy.context.scene.render.resolution_percentage = 100  # 分辨率百分比

    # 帧范围设置
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 1

    # 输出设置
    render.image_settings.file_format = 'PNG'
    render.filepath = os.path.join(output_dir, "rgb/Image####")

def configure_depth(output_dir: str):
    """配置深度图输出节点"""
    # 启用深度通道 
    bpy.context.scene.view_layers[0].use_pass_z = True

    # 配置合成器节点
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    nodes = tree.nodes
    links = tree.links

    # 添加渲染层和文件输出节点
    rl_node = nodes.new("CompositorNodeRLayers")
    output_node = nodes.new("CompositorNodeOutputFile")
    output_node.base_path = os.path.join(output_dir, "depth")  # 深度图输出目录
    output_node.format.file_format = 'OPEN_EXR'                # EXR格式保存深度 
    output_node.format.color_depth = '32'                       # 32位精度

    # 连接深度通道
    links.new(rl_node.outputs['Depth'], output_node.inputs['Image'])


def configure_normal(output_dir: str):
    """配置法向图输出节点"""
    # 启用法向通道
    bpy.context.scene.view_layers[0].use_pass_normal = True

    # 配置合成器节点
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    nodes = tree.nodes
    links = tree.links

    # 添加渲染层和文件输出节点
    rl_node = nodes.new("CompositorNodeRLayers")
    output_node = nodes.new("CompositorNodeOutputFile")
    output_node.base_path = os.path.join(output_dir, "normal")  # 法向图输出目录
    output_node.format.file_format = 'OPEN_EXR'  # EXR格式保存法向
    output_node.format.color_depth = '32'  # 32位精度

    # 连接法向通道
    links.new(rl_node.outputs['Normal'], output_node.inputs['Image'])

def get_camera_rays():
    # 获取当前场景和相机
    scene = bpy.context.scene
    camera = scene.camera
    render = scene.render

    # 获取相机参数
    resolution_x = render.resolution_x
    resolution_y = render.resolution_y
    sensor_width = camera.data.sensor_width
    focal_length = camera.data.lens

    # 计算相机坐标系到世界坐标系的变换矩阵
    camera_matrix = camera.matrix_world # c2w

    # 获取相机位置 (rays_o)
    rays_o = camera_matrix.translation

    # 创建像素网格坐标
    x = np.arange(resolution_x)
    y = np.arange(resolution_y)[::-1]
    X, Y = np.meshgrid(x, y)

    # 转换为标准化设备坐标 (NDC)
    x_ndc = (X + 0.5) / resolution_x
    y_ndc = (Y + 0.5) / resolution_y

    # 转换为传感器坐标
    aspect_ratio = resolution_x / resolution_y
    sensor_height = sensor_width / aspect_ratio
    x_sensor = (x_ndc - 0.5) * sensor_width
    y_sensor = (y_ndc - 0.5) * sensor_height

    # 构建射线方向 (相机坐标系)
    z_cam = -focal_length
    directions = np.dstack((
        x_sensor,
        y_sensor,
        np.full_like(x_sensor, z_cam)
    ))

    # 转换为世界坐标系
    rotation_matrix = camera_matrix.to_3x3().transposed()
    directions = np.dot(directions, rotation_matrix)
    rays_d = directions / np.linalg.norm(directions, axis=2, keepdims=True)

    return np.array(rays_o), rays_d


def get_valid_pos(output_dir):
    # 读取 EXR 文件
    exr_file = OpenEXR.InputFile(os.path.join(output_dir, 'depth/Image0001.exr'))
    dw = exr_file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    R_channel = np.frombuffer(exr_file.channel('R', FLOAT), dtype=np.float32).reshape(size[1], size[0])
    G_channel = np.frombuffer(exr_file.channel('G', FLOAT), dtype=np.float32).reshape(size[1], size[0])
    B_channel = np.frombuffer(exr_file.channel('B', FLOAT), dtype=np.float32).reshape(size[1], size[0])
    depth_map = np.stack((R_channel, G_channel, B_channel), axis=-1).mean(axis=-1)

    exr_file = OpenEXR.InputFile(os.path.join(output_dir, 'normal/Image0001.exr'))
    dw = exr_file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    R_channel = np.frombuffer(exr_file.channel('X', FLOAT), dtype=np.float32).reshape(size[1], size[0])
    G_channel = np.frombuffer(exr_file.channel('Y', FLOAT), dtype=np.float32).reshape(size[1], size[0])
    B_channel = np.frombuffer(exr_file.channel('Z', FLOAT), dtype=np.float32).reshape(size[1], size[0])
    normal_map = np.stack((R_channel, G_channel, B_channel), axis=-1)

    normal_z = np.dot(normal_map, np.array([0, 0, 1]))
    valid_location = (normal_z >= np.cos(10 * np.pi / 180))

    # 找出满足条件的点：原数组为True且距离≥10
    distances = distance_transform_edt(valid_location)
    valid_location = valid_location & (distances >= 10)

    valid_location = valid_location & (depth_map <= depth_map.mean() * 2)
    fig, ax = plt.subplots()
    im = ax.imshow(valid_location, cmap='viridis')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('valid')
    ax.set_title('Depth Map Visualization')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.savefig(os.path.join(output_dir, 'valid_location.png'))
    return valid_location, depth_map

def find_location(object_list, object_d, valid_location, ground_location, rays_o, rays_d, depth_map):
    # select points in valid_location
    valid_points = np.argwhere(valid_location)
    if len(valid_points) == 0:
        # print("no valid position for object")
        return None, None, None, None, None
    indices = np.random.choice(len(valid_points), 1, replace=False)[0]
    valid_point = valid_points[indices]
    
    # calculate 3d position
    row, col = valid_point
    valid_obj = np.arange(len(object_list))[(object_d/2<depth_map[row, col])&(depth_map[row, col]<object_d*2)]
    if len(valid_obj) == 0:
        return None, None, None, None, None
    selected_index = np.random.choice(valid_obj, 1, replace=False)[0]

    if ground_location[row, col]:
        depth = depth_map[row, col]
    else:
        depth = object_d[selected_index] + random.random() * min(object_d[selected_index], depth_map[row, col] - object_d[selected_index])
    start_location = rays_o + rays_d[row, col] * depth

    movement_range = np.random.randint(0, 10)
    min_row = int(max(0, row - movement_range))
    max_row = int(min(valid_location.shape[0], row + movement_range + 1))
    min_col = int(max(0, col - movement_range))
    max_col = int(min(valid_location.shape[1], col + movement_range + 1))
    valid_location[min_row:max_row, min_col:max_col] = False

    row2 = random.choice([min_row, max_row-1])
    col2 = random.choice([min_col, max_col-1])
    end_location = rays_o + rays_d[row2, col2] * depth_map[row2, col2]
    print("start_location", start_location, "end_location", end_location, "is_ground", ground_location[row, col])
    obj_name = object_list[selected_index]
    object_d = np.delete(object_d, selected_index)
    del object_list[selected_index]
    return obj_name, start_location, end_location, valid_location, object_d


def find_character_location(ground_location, center_location, rays_o, rays_d, depth_map):
    # select points in valid_location
    valid_points = np.argwhere(ground_location&center_location)
    if len(valid_points) == 0:
        print("no valid position for character")
        return None, None, None, None, None
    indices = np.random.choice(len(valid_points), 1, replace=False)[0]
    valid_point = valid_points[indices]
    
    # calculate 3d position
    row, col = valid_point
    depth = depth_map[row, col]
    location = rays_o + rays_d[row, col] * depth

    print("location", location)
    return location

def main(args):
    config_path = args.config
    scene_file = args.scene
    # 加载配置文件
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    if args.id is not None:
        output_dir = os.path.join(config['output_dir'], scene_file.split('/')[-1].split('.')[0]+f'-{args.id}', 'init')
    else:
        output_dir = os.path.join(config['output_dir'], scene_file.split('/')[-1].split('.')[0], 'init')

    bpy.ops.wm.open_mainfile(filepath=scene_file)

    configure_render(output_dir, config['render_settings'])
    configure_depth(output_dir)
    configure_normal(output_dir)

    print("Rendering video, normal and depth maps...")
    bpy.ops.render.render(animation=True)
    print(f"results saved to: {output_dir}")

    ground_location, depth_map = get_valid_pos(output_dir)

    center_location = np.zeros_like(ground_location).astype(bool)
    center_location[int(center_location.shape[0]*0.15):-int(center_location.shape[0]*0.15), 
                    int(center_location.shape[1]*0.15):-int(center_location.shape[1]*0.15)] = True
    
    if config['valid_location'] == 'ground' and np.sum(ground_location) == 0:
        print("no ground location, switch valid_location to center")
        config['valid_location'] = 'center'
    
    if config['valid_location'] == 'all':
        valid_location = np.ones_like(ground_location).astype(bool)
    elif config['valid_location'] == 'center':
        valid_location = center_location.copy()
    elif config['valid_location'] == 'ground':
        valid_location = ground_location.copy()
    else:
        assert False, "invalid valid_location"
    
    rays_o, rays_d = get_camera_rays()

    objects_config = {}
    with open(config['character_list'], 'r') as f:
        character_list = [l.strip() for l in f.readlines()]
    with open(config['action_list'], 'r') as f:
        action_list = [l.strip() for l in f.readlines()]
    selected_charatrer = random.sample(character_list, 1)[0]
    objects_config['charatrer'] = []
    # for character in selected_charatrer:
    location = find_character_location(ground_location, center_location, rays_o, rays_d, depth_map)
    chara_config = {
        'charatrer_path': selected_charatrer,
        'action_path': random.sample(action_list, 1)[0],
        'location': location.tolist()
    }
    objects_config['charatrer'].append(chara_config)

    object_d = []
    with open(config['object_list'], 'r', encoding='utf-8') as f:
        object_list = [l.strip() for l in f.readlines()]
    object_d = np.array([float(o.split(' -- ')[1]) for o in object_list])
    object_list = [o.split(' -- ')[0] for o in object_list]
    objects_config['objects'] = []
    min_object_count, max_object_count = config.get('object_count', (3, 5))
    object_count = random.randint(min_object_count, max_object_count)
    for i in range(object_count):
        obj, start_location, end_location, valid_location, object_d = find_location(object_list, object_d, valid_location, ground_location, rays_o, rays_d, depth_map)
        if obj is not None:
            obj_config = {
                'obj_path': obj,
                'start_location': start_location.tolist(),
                'end_location': end_location.tolist(),
            }
            objects_config['objects'].append(obj_config)


    with open(os.path.join(output_dir, "objects.json"), 'w') as f:
        json.dump(objects_config, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scene composition and rendering pipeline')
    parser.add_argument('--config', default="config.json", help='Path to JSON configuration file')
    parser.add_argument('--scene', default="scene/[BBB]BabbasCafe.blend", help='Path to the Blender scene file')
    parser.add_argument('--id', type=int, default=None)
    args = parser.parse_args()

    main(args)

# python render_init.py --config config.json