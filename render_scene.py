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
from mathutils import Vector, Matrix, Euler, Quaternion
from scipy.ndimage import distance_transform_edt
# conda activate /mnt/afs/miniconda/envs/blender
# pip install bpy OpenEXR matplotlib scipy
# pip install numpy==1.26

def add_sun():
    # 检查场景中是否已经存在太阳灯
    has_sun_light = False
    for obj in bpy.context.scene.objects:
        if obj.type == 'LIGHT' and obj.data.type == 'SUN':
            has_sun_light = True
            break

    # 如果没有太阳灯，则添加一个
    if not has_sun_light:
        # add sun light
        sun_light_data = bpy.data.lights.new(name="Sun", type='SUN')
        sun_light_data.energy = 5.0
        sun_light_data.color = (1.0, 1.0, 1.0)
        sun_light_object = bpy.data.objects.new(name="Sun", object_data=sun_light_data)
        bpy.context.collection.objects.link(sun_light_object)

def configure_render(output_dir: str, config: dict):
    """配置渲染参数"""
    render = bpy.context.scene.render
    render.engine = config.get('engine', 'BLENDER_EEVEE_NEXT')
    render.filepath = output_dir

    # 根据引擎类型配置参数
    if render.engine == 'CYCLES':
        cycles = bpy.context.scene.cycles
        cycles.samples = config.get('cycles_samples', 8)  # 采样数
        cycles.use_denoising = config.get('use_denoising', True)  # 去噪
        cycles.device = config.get('device', 'GPU')  # 设备选择
        cycles.max_bounces = config.get('max_bounces', 3)  # 光线反弹次数
        cycles.transparent_max_bounces = config.get('transparent_bounces', 2)  # 透明反弹次数

    elif render.engine == 'BLENDER_EEVEE_NEXT':
        eevee = bpy.context.scene.eevee
        # 采样设置
        eevee.taa_render_samples = config.get('taa_render_samples', 32)  # 渲染采样（1 - 64）
        eevee.taa_samples = config.get('eevee_taa_samples', 64)  # 抗锯齿采样（1 - 64）
        eevee.use_gtao = config.get('use_gtao', True)  # 全局光照

    # 分辨率设置
    resolution = config.get('resolution', (384, 672))
    render.resolution_x = resolution[1]
    render.resolution_y = resolution[0]
    bpy.context.scene.render.resolution_percentage = 100  # 分辨率百分比

    # 帧范围设置
    frame_settings = config.get('frame_range', [1, 49])
    bpy.context.scene.frame_start = frame_settings[0]
    bpy.context.scene.frame_end = frame_settings[1]

    # 输出设置
    render.image_settings.file_format = 'FFMPEG'
    render.ffmpeg.format = 'MPEG4'
    render.ffmpeg.codec = 'H264'
    render.filepath = os.path.join(output_dir, "rendered_video.mp4")


def load_depth(init_render_dir):
    exr_file = OpenEXR.InputFile(os.path.join(init_render_dir, 'depth/Image0001.exr'))
    dw = exr_file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    R_channel = np.frombuffer(exr_file.channel('R', FLOAT), dtype=np.float32).reshape(size[1], size[0])
    G_channel = np.frombuffer(exr_file.channel('G', FLOAT), dtype=np.float32).reshape(size[1], size[0])
    B_channel = np.frombuffer(exr_file.channel('B', FLOAT), dtype=np.float32).reshape(size[1], size[0])
    depth_map = np.stack((R_channel, G_channel, B_channel), axis=-1).mean(axis=-1)
    exr_file.close()
    return depth_map


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


def get_new_camera(rays_o, rays_d, row, col, depth, target_point):
    new_location = Vector(rays_o + rays_d[row,col]*depth)
    print(row, col, new_location)
    
    # 基础朝向四元数（-Z轴指向目标）
    base_quat = (target_point - new_location).normalized().to_track_quat('-Z', 'Y')

    # 添加随机旋转（使用四元数组合）
    angle_x = 0 # random.uniform(-10, 10) * np.pi/180
    angle_y = 0 # random.uniform(-10, 10) * np.pi/180
    
    # 创建旋转四元数
    rot_quat_x = Quaternion((1, 0, 0), angle_x)  # X轴旋转
    rot_quat_y = Quaternion((0, 1, 0), angle_y)  # Y轴旋转
    
    # 组合旋转（顺序：先Y后X，根据需求调整）
    new_rotation = base_quat @ rot_quat_y @ rot_quat_x

    return new_location, new_rotation


def set_camera_trajectory(config, depth_map, target_point=None):
    scene = bpy.context.scene
    camera = bpy.context.scene.camera
    rays_o, rays_d = get_camera_rays()
    
    row = np.random.randint(0, rays_d.shape[0])
    col = np.random.randint(0, rays_d.shape[1])
    frame_start = bpy.context.scene.frame_start
    frame_end = bpy.context.scene.frame_end
    max_camera_keyframes = config.get('max_camera_keyframes', 2)  # 关键帧数量
    camera_keyframes = random.choice(list(range(1, max_camera_keyframes+1)))  # 随机关键帧数量
    # camera_keyframes表示会有多少次位姿变换，实际上关键帧数量是camera_keyframes+1
    print(list(range(frame_start, frame_end+1, (frame_end-frame_start)//camera_keyframes)))
    for i in range(frame_start, frame_end+1, (frame_end-frame_start)//camera_keyframes):
        if i == frame_start:
            scene.frame_set(bpy.context.scene.frame_start)
            camera.keyframe_insert(data_path="location", frame=bpy.context.scene.frame_start)
            camera.keyframe_insert(data_path="rotation_quaternion", frame=bpy.context.scene.frame_start)
        
        else:
            depth = random.random()*min(depth_map[row,col]/2, depth_map.mean()/2)
            if target_point is None:
                target_point = Vector(rays_o + rays_d[rays_d.shape[0]//2,rays_d.shape[1]//2]*800)
            new_location, new_rotation = get_new_camera(rays_o, rays_d, row, col, depth, target_point)
            
            scene.frame_set(i)
            camera.location = new_location
            camera.rotation_quaternion = new_rotation
            camera.keyframe_insert(data_path="location", frame=i)
            camera.keyframe_insert(data_path="rotation_quaternion", frame=i)

            while True:
                new_row = row + int(random.uniform(-20, 20))
                new_col = col + int(random.uniform(-20, 20))
                if 0 <= new_row < rays_d.shape[0] and 0 <= new_col < rays_d.shape[1]:
                    break
            row = new_row
            col = new_col

    # 设置插值类型为贝塞尔曲线以实现平滑过渡
    for fcurve in camera.animation_data.action.fcurves:
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = 'BEZIER'
            keyframe.handle_left_type = 'AUTO_CLAMPED'
            keyframe.handle_right_type = 'AUTO_CLAMPED'

def import_person(character_path, source_arm_path=None, position=Vector((0,0,0))):
    # 导入骨骼动画
    if source_arm_path is not None:
        bpy.ops.import_scene.fbx(
            filepath=source_arm_path,
            use_anim=True,       # 关键：导入动画
            use_custom_props=False,
            use_image_search=False,
            use_alpha_decals=False,
            decal_offset=0.0,
            automatic_bone_orientation=True,  # 自动骨骼方向适配
            force_connect_children=False
        )
        imported_obj_1 = [obj for obj in bpy.context.selected_objects if obj.type not in ['CAMERA', 'LIGHT']]

        for obj1 in imported_obj_1:
            if obj1.animation_data is not None:
                source_arm = obj1
            if obj1.parent is None:
                obj1.location = position
                print(f"物体名称: {obj1.name}, 位置: {obj1.location}, 缩放: {obj1.scale}")

    # 导入人体
    bpy.ops.import_scene.fbx(filepath=character_path)
    imported_obj_2 = [obj for obj in bpy.context.selected_objects if obj.type not in ['CAMERA', 'LIGHT']]

    for obj2 in imported_obj_2:
        if obj2.animation_data is not None or 'Armature' in obj2.name:
            target_arm = obj2
        if obj2.parent is None:
            obj2.location = position
            print(f"物体名称: {obj2.name}, 位置: {obj2.location}, 缩放: {obj2.scale}")

    if source_arm_path is not None:
        setup_retargeting(target_arm, source_arm)

        for obj in imported_obj_1:
            if obj != source_arm:  # 避免删除源骨架本身（如果需要保留）
                bpy.data.objects.remove(obj, do_unlink=True)

def import_obj(obj_config):
    filepath = obj_config['obj_path']
    obj_type = obj_config.get('type', os.path.splitext(filepath)[-1][1:].lower())
    start_location = obj_config.get('start_location', (0, 0, 0))
    end_location = obj_config.get('end_location', (0, 0, 0))

    imported_object = []
    if obj_type == 'blend':
        with bpy.data.libraries.load(filepath, link=False) as (data_src, data_dst):
            data_dst.objects = data_src.objects
        for obj in data_dst.objects:
            if obj is not None:
                if obj.type not in ['CAMERA', 'LIGHT']:
                    bpy.context.collection.objects.link(obj)
                    imported_object.append(obj)
    elif obj_type in ['glb', 'gltf']:
        bpy.ops.import_scene.gltf(
            filepath=filepath,
            merge_vertices=True,
            import_shading='NORMALS',  # 或尝试 'VERTEX' 或 'NONE'
            # import_images=True,        # 确保导入关联的纹理
            bone_heuristic='BLENDER'   # 如果模型有骨骼动画
        )
        # bpy.ops.import_scene.gltf(filepath=filepath, merge_vertices=True, import_shading='NORMALS')
        # imported_object = bpy.context.selected_objects
        imported_object = [obj for obj in bpy.context.selected_objects if obj.type not in ['CAMERA', 'LIGHT']]
    else:
        raise ValueError(f"Unsupported format: {obj_type}")
    
    # 修复所有导入对象的材质贴图路径
    for obj in imported_object:
        # 设置物体位置
        if obj.parent is None:
            obj.location = Vector(start_location)
            print(f"物体名称: {obj.name}, 位置: {obj.location}")

        # 检查并修复材质贴图路径
        texture_dir = os.path.join(os.path.split(os.path.split(filepath)[0])[0], 'textures')  # 获取纹理目录
        if obj.active_material:
            for node in obj.active_material.node_tree.nodes:
                if node.type == 'TEX_IMAGE' and node.image:
                    # 提取贴图文件名（例如 "texture1.png"）
                    texture_name = os.path.basename(node.image.filepath)
                    # 生成新路径：指向 textures 文件夹
                    new_path = os.path.join(texture_dir, texture_name)
                    # 更新路径并强制重新加载
                    if os.path.exists(new_path):
                        node.image.filepath = new_path
                        node.image.reload()
                        print(f"已修复贴图路径: {new_path}")
                    else:
                        print(f"警告: 贴图文件不存在: {new_path}")

    for obj in imported_object:
        if obj.parent is None:
            obj.location = Vector(start_location)
            print(f"物体名称: {obj.name}, 位置: {obj.location}, 缩放: {obj.scale}")

def setup_retargeting(target_arm, source_arm):
    # 确保使用相同的骨架类型
    target_arm.data.pose_position = 'POSE'
    source_arm.data.pose_position = 'POSE'
    target_arm.data.use_mirror_x = False

    translator = {}
    for target_bone in target_arm.pose.bones:
        if 'Hips' in target_bone.name:
            root_bone_name = target_bone.name
        if ':' in target_bone.name:
            target_bone_name = target_bone.name.split(':')[-1]
        if ':' in source_arm.pose.bones[0].name:
            source_bone_name = source_arm.pose.bones[0].name.split(':')[0] + ':' + target_bone_name
        else:
            source_bone_name = target_bone_name
        translator[target_bone.name] = source_bone_name
    
    print('source bone num:', len([s for s in source_arm.pose.bones]))
    print('target bone num:', len([t for t in target_arm.pose.bones]))

    for target_bone in target_arm.pose.bones:
        source_bone_name = translator[target_bone.name]
        if source_bone_name in source_arm.pose.bones:
            source_bone = source_arm.pose.bones[source_bone_name]
            # 清理旧约束
            for const in target_bone.constraints:
                target_bone.constraints.remove(const)
            
            # 添加位置约束
            const_loc = target_bone.constraints.new('COPY_LOCATION')
            const_loc.target = source_arm
            const_loc.subtarget = source_bone_name
            
            # 添加旋转约束
            const_rot = target_bone.constraints.new('COPY_ROTATION')
            const_rot.target = source_arm
            const_rot.subtarget = source_bone_name

    # 烘焙动画
    bpy.context.view_layer.objects.active = target_arm
    target_arm.select_set(True)
    
    # 获取帧范围
    frame_start = bpy.context.scene.frame_start
    frame_end = bpy.context.scene.frame_end
    if source_arm.animation_data and source_arm.animation_data.action:
        action = source_arm.animation_data.action
        frame_start = int(action.frame_range[0])
        frame_end = int(action.frame_range[1])
    
    # 执行烘焙（保留约束）
    bpy.ops.nla.bake(
        frame_start=frame_start,
        frame_end=frame_end,
        step=1,
        only_selected=True,
        visual_keying=True,
        clear_constraints=True, 
        bake_types={'POSE'}
    )

def save_metadata(output_dir):
    scene = bpy.context.scene
    camera = bpy.context.scene.camera

    def get_focal_px(camera):
        sensor_width_mm = camera.data.sensor_width
        focal_mm = camera.data.lens
        resolution_x = scene.render.resolution_x
        return (focal_mm / sensor_width_mm) * resolution_x

    intrinsics = []
    extrinsics = []
    for frame in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()  # 确保数据更新

        K = [[get_focal_px(camera), 0, scene.render.resolution_x//2],
            [0, get_focal_px(camera), scene.render.resolution_y//2],
            [0, 0, 1]]
        intrinsics.append(K)

        matrix = camera.matrix_world    # Twc
        extrinsics.append([list(row) for row in matrix][:3])

    with open(os.path.join(output_dir, "intrinsics.txt"), 'w') as f:
        for intrinsic in intrinsics:
            f.write(str(intrinsic)+ '\n')
    with open(os.path.join(output_dir, "extrinsics.txt"), 'w') as f:
        for extrinsic in extrinsics:
            f.write(str(extrinsic)+ '\n')

    print(f"Saved camera parameters for {len(intrinsic)} frames to {output_dir}")

def main(args):
    # =================== 设置场景保存路径 ====================
    config_path = args.config
    scene_file = args.scene
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    if args.id is not None:
        scene_name = scene_file.split('/')[-1].split('.')[0] + f'-{args.id}'
    else:
        scene_name = scene_file.split('/')[-1].split('.')[0]

    init_render_dir = os.path.join(config['output_dir'], scene_name, 'init')

    if not os.path.exists(init_render_dir):
        if args.id is not None:
            command = f'python render_init.py --config {config_path} --scene "{scene_file}" --id {args.id}'
        else:
            command = f'python render_init.py --config {config_path} --scene "{scene_file}"'
        print(command)
        os.system(command)

    render_id = len(os.listdir(os.path.join(config['output_dir'], scene_name)))
    output_dir = os.path.join(config['output_dir'], scene_name, f"render{str(render_id).zfill(4)}")

    # =============== 初始化场景 =====================
    bpy.ops.wm.open_mainfile(filepath=scene_file)
    bpy.context.scene.camera.rotation_mode = 'QUATERNION'
    bpy.context.scene.render.resolution_percentage = 100  # 分辨率百分比

    # =============== 渲染配置 ================
    configure_render(output_dir, config['render_settings'])
    configure_depth(output_dir)

    # =============== 导入物体&人体 ==================
    add_sun()
    depth_map = load_depth(init_render_dir)
    
    with open(os.path.join(init_render_dir, 'objects.json'), encoding="utf-8") as f:
        objects_config = json.load(f)
    for charatrer_config in objects_config['charatrer']:
        import_person(
            character_path=charatrer_config['charatrer_path'],
            source_arm_path=charatrer_config['action_path'],
            position=Vector(charatrer_config['location']))
    for obj_config in objects_config['objects']:
        import_obj(obj_config)

    # ==================== 设置相机轨迹 ==========================
    random_vector = Vector((random.random()*2-1, random.random()*2-1, random.random()*2-1))
    camera_target = Vector(charatrer_config['location']) + Vector((0,0,1.5)) + random_vector
    set_camera_trajectory(config, depth_map, camera_target)

    # ================== 执行渲染（同时输出视频和深度图）=======================
    print("Rendering video and depth maps...")
    bpy.ops.render.render(animation=True)
    print(f"Video saved to: {config['output_dir']}")

    # ================= 保存元数据 ===================
    save_metadata(output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scene composition and rendering pipeline')
    parser.add_argument('--config', default="config.json", help='Path to JSON configuration file')
    parser.add_argument('--scene', default="scene/[BBB]BabbasCafe.blend", help='Path to the Blender scene file')
    parser.add_argument('--id', type=int, default=None)
    args = parser.parse_args()
    
    main(args)

'''
# python render_scene.py --config config.json --scene "scene/[BBB]BabbasCafe.blend" --id 0
python render_scene.py --config config.json --scene "scene/[BBB]BabbasCafe.blend" --id 1

python render_scene.py --config config.json --scene "scene/[BBB]BabbasCafe.blend" --id 3
python render_scene.py --config config.json --scene "scene/[BBB]BabbasCafe.blend" --id 3
python render_scene.py --config config.json --scene "scene/[BBB]BabbasCafe.blend" --id 3
python render_scene.py --config config.json --scene "scene/[BBB]BabbasCafe.blend" --id 3
python render_scene.py --config config.json --scene "scene/[BBB]BabbasCafe.blend" --id 3
'''