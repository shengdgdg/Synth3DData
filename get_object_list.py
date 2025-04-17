
import bpy
import math
import os
from mathutils import Vector

# 默认垂直视场角 (FOV) - 度数
DEFAULT_FOV_DEGREES = 10
# 距离边距系数 (例如 1.1 表示留出 10% 的边距)
PADDING_FACTOR = 1.0


def clear_scene():
    """清空当前 Blender 场景中的所有对象、材质等。"""
    # 确保处于对象模式
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    # 选择所有对象
    bpy.ops.object.select_all(action='SELECT')
    # 删除选中的对象
    bpy.ops.object.delete(use_global=False, confirm=False)

    # 清理孤立数据 (可选, 但推荐)
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    print("Scene cleared.")

def get_scene_mesh_objects():
    """获取当前场景中所有类型为 MESH 的对象。"""
    return [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']

def get_world_bounding_box(objects):
    """
    计算一组对象在世界坐标系下的组合轴对齐包围盒 (AABB)。

    Args:
        objects (list): bpy.types.Object 对象的列表。

    Returns:
        tuple: (min_corner, max_corner) 两个 Vector，如果无有效对象则返回 (None, None)。
               或者返回 None 如果无法计算（例如没有顶点）。
    """
    if not objects:
        return None, None

    min_corner = Vector((float('inf'), float('inf'), float('inf')))
    max_corner = Vector((float('-inf'), float('-inf'), float('-inf')))
    has_geometry = False

    for obj in objects:
        if not obj.bound_box:  # 对象可能没有边界框（例如空网格）
            continue

        # bound_box 的角点是局部坐标
        local_corners = [Vector(corner) for corner in obj.bound_box]
        # 转换到世界坐标
        world_corners = [obj.matrix_world @ corner for corner in local_corners]

        if not world_corners:
            continue

        has_geometry = True
        # 更新全局最小和最大角点
        for corner in world_corners:
            min_corner.x = min(min_corner.x, corner.x)
            min_corner.y = min(min_corner.y, corner.y)
            min_corner.z = min(min_corner.z, corner.z)
            max_corner.x = max(max_corner.x, corner.x)
            max_corner.y = max(max_corner.y, corner.y)
            max_corner.z = max(max_corner.z, corner.z)

    if not has_geometry:
        return None, None

    return min_corner, max_corner


def calculate_camera_distance(objects, fov_degrees=DEFAULT_FOV_DEGREES, padding=PADDING_FACTOR):
    """
    计算相机为了在给定 FOV 下“合适地”看到指定对象组所需的距离。

    Args:
        objects (list): 需要框入视图的 bpy.types.Object 列表 (通常是 MESH 类型)。
        fov_degrees (float): 相机的垂直视场角 (度数)。
        padding (float): 边距系数，大于 1.0 以留出空白。

    Returns:
        float: 计算出的相机距离。如果无法计算（如无对象或对象无尺寸），返回 None。
    """
    if not objects:
        print("Warning: No objects provided for distance calculation.")
        return None

    min_corner, max_corner = get_world_bounding_box(objects)

    if min_corner is None or max_corner is None:
        print("Warning: Could not determine world bounding box for the objects.")
        return None

    # 计算边界框在世界坐标系中的尺寸
    size = max_corner - min_corner
    max_dimension = max(size.x, size.y, size.z)

    if max_dimension <= 1e-6: # 检查物体是否有实际尺寸
        print(f"Warning: Object maximum dimension is near zero ({max_dimension}). Cannot calculate distance.")
        return None # 或者返回一个默认值？

    # 将 FOV 从度转换为弧度
    fov_rad = math.radians(fov_degrees)

    # 使用三角函数计算距离
    # tan(fov/2) = (半个物体尺寸) / distance
    # distance = (半个物体尺寸) / tan(fov/2)
    # 我们使用最大尺寸来确保整个物体在各个方向上都能容纳
    distance = (max_dimension / 2.0) / math.tan(fov_rad / 2.0)

    # 应用边距
    distance *= padding

    return distance


def find_blend_files(root_dir):
    blend_files = []
    # 遍历根目录及其子目录
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            # 检查文件扩展名是否为 .blend
            if file.endswith('.blend'):
                # 构建文件的完整路径
                file_path = os.path.join(root, file)
                blend_files.append(file_path)
    return blend_files

if __name__ == "__main__":
    print(f"Starting object processing...")
    print(f"Using default vertical FOV: {DEFAULT_FOV_DEGREES} degrees")
    print("-" * 30)

    blender_root = "E:/Recam/blender/object"
    file_list = find_blend_files(blender_root)

    results = {}

    for filepath in file_list:
        filepath = os.path.join(blender_root, filepath)
        print(f"Processing: {filepath}")
        if not os.path.exists(filepath):
            print(f"  Error: File not found. Skipping.")
            results[filepath] = "File not found"
            print("-" * 30)
            continue

        # 1. 清理场景
        try:
            clear_scene()
        except Exception as e:
            print(f"  Error clearing scene: {e}. Skipping file.")
            results[filepath] = f"Error clearing scene: {e}"
            print("-" * 30)
            continue

        # 2. 加载/导入文件
        try:
            if filepath.lower().endswith(".blend"):
                bpy.ops.wm.open_mainfile(filepath=filepath)
            elif filepath.lower().endswith((".gltf", ".glb")):
                bpy.ops.import_scene.gltf(filepath=filepath)
            else:
                print(f"  Error: Unsupported file type: {os.path.splitext(filepath)[1]}. Skipping.")
                results[filepath] = "Unsupported file type"
                print("-" * 30)
                continue
            print("  File loaded/imported successfully.")
        except Exception as e:
            print(f"  Error loading/importing file: {e}. Skipping.")
            results[filepath] = f"Error loading file: {e}"
            # 尝试再次清理，以防加载部分成功但出错
            try: clear_scene()
            except: pass
            print("-" * 30)
            continue

        # 3. 查找对象
        mesh_objects = get_scene_mesh_objects()
        if not mesh_objects:
            print("  Warning: No mesh objects found in the scene after loading.")
            results[filepath] = "No mesh objects found"
            print("-" * 30)
            continue
        else:
            print(f"  Found {len(mesh_objects)} mesh object(s).")

        # 4. 计算距离
        try:
            optimal_distance = calculate_camera_distance(mesh_objects, fov_degrees=DEFAULT_FOV_DEGREES, padding=PADDING_FACTOR)

            if optimal_distance is not None:
                print(f"  Calculated optimal camera distance: {optimal_distance:.4f}")
                results[filepath] = optimal_distance
            else:
                print("  Could not calculate optimal distance.")
                results[filepath] = "Calculation failed"

        except Exception as e:
            print(f"  Error during distance calculation: {e}")
            results[filepath] = f"Error calculating distance: {e}"

        print("-" * 30)

    # --- 打印总结 ---
    print("\n=== Processing Summary ===")
    with open("object.list", "w", encoding='utf-8') as f:
        for fname, res in results.items():
            if isinstance(res, (int, float)):
                f.write(f"{fname} -- {res:.4f}\n")
                print(f"{fname} -- {res:.4f}")
            else:
                print(f"{fname} -- {res}\n")

    print("=" * 25)

