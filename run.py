import os

'''
conda activate blender
python run.py

rsync -avz /storage0/chenxinyu/blender/output H800:/mnt/afs_james/chenxinyu/datasets/blender4
'''
with open('scene.list', 'r') as f:
    scene_list = [l.strip() for l in f.readlines()]

for scene in scene_list:
    for i in range(10):
        for _ in range(10):
            command = f'python render_scene.py --config config.json --scene "{scene}" --id {i}'
            print(command)
            os.system(command)