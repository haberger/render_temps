import blenderproc as bproc
import argparse
import os
import numpy as np
import bpy
from PIL import Image

parser = argparse.ArgumentParser()
# parser.add_argument('bop_parent_path', nargs='?', help="Path to the bop datasets parent directory")
# parser.add_argument('bop_dataset_name', nargs='?', help="Main BOP dataset")
# parser.add_argument('output_dir', nargs='?', help="Path to where the final files will be saved ")
# args = parser.parse_args()
args = {}
args['bop_parent_path'] = 'bop'
args['bop_dataset_name'] = 'ycbv'
args['output_dir'] = 'out'
args['obj_pose'] = 'predefined_poses/obj_poses_level2.npy'
args['cam_pose'] = 'predefined_poses/cam_poses_level2.npy'
args['poses'] = 'upper'
print(args['bop_parent_path'])
bproc.init()

img_size = (480, 640)
poses = np.load(args['obj_pose'])

if args['poses'] == 'upper':
    cam_poses = np.load(args['cam_pose'])
    poses = poses[cam_poses[:, 2, 3] >= 0]
# poses[:, :3, 3] *= 0.4
poses[:, :3, :3] = poses[:, :3, :3] / 1000.0
poses[:, :3, 3] = poses[:, :3, 3] / 1000.0
# load specified bop objects into the scene
print(poses.shape)

bop_objs = bproc.loader.load_bop_objs(bop_dataset_path=os.path.join(args['bop_parent_path'], args['bop_dataset_name']),
                                      mm2m=True)

# load BOP datset intrinsics
bproc.loader.load_bop_intrinsics(bop_dataset_path=os.path.join(args['bop_parent_path'], args['bop_dataset_name']))

light = bproc.types.Light()
light.set_type("POINT")
light.set_location([1, -1, 1])
light.set_energy(200)
light = bproc.types.Light()
light.set_type("POINT")
light.set_location([-1, -1, -1])
light.set_energy(200)
light = bproc.types.Light()
light.set_type("POINT")
light.set_location([-1, 0, -1])
light.set_energy(20)
light.set_type("POINT")
light.set_location([1, 0, 1])
light.set_energy(20)



# bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")

# set shading
for j, obj in enumerate(bop_objs):
    obj.set_shading_mode('auto')
    obj.hide(True)


cam2world = bproc.math.change_source_coordinate_frame_of_transformation_matrix(
    np.eye(4), ["X", "-Y", "-Z"]
)
bproc.camera.add_camera_pose(cam2world)
bproc.renderer.set_max_amount_of_samples(100)

# activate depth rendering
bproc.renderer.enable_distance_output(True)



black_img = Image.new('RGB', (img_size[1], img_size[0]))

for obj in bop_objs:
    obj.hide(False)
    name = args['bop_dataset_name']
    obj_data_dir = os.path.join(args['output_dir'], name, f'obj_{obj.get_cp("category_id")}')
    if not os.path.exists(obj_data_dir):
        os.makedirs(obj_data_dir)
        print(f"Directory '{obj_data_dir}' created.")
    else:
        print(f"Directory '{obj_data_dir}' already exists.")
    for idx_frame, obj_pose in enumerate(poses[0:2]):
        obj.set_local2world_mat(obj_pose)
        data = bproc.renderer.render()
        data.update(bproc.renderer.render_segmap(map_by="class", use_alpha_channel=True))
        # # Map distance to depth
        depth = bproc.postprocessing.dist2depth(data["distance"])[0]
        mask = np.uint8((depth < 1000) * 255)
        mask = Image.fromarray(mask)
        mask.save(os.path.join(obj_data_dir, "mask_{:06d}.png".format(idx_frame)))

        rgb = Image.fromarray(np.uint8(data["colors"][0]))
        img = Image.composite(rgb, black_img, mask)
        img.save(os.path.join(obj_data_dir, "{:06d}.png".format(idx_frame)))
    obj.hide(True)
np.save(os.path.join(args['output_dir'], name, "obj_poses.npy"), poses)
