import blenderproc as bproc
import argparse
import os
import numpy as np
import bpy

parser = argparse.ArgumentParser()
# parser.add_argument('bop_parent_path', nargs='?', help="Path to the bop datasets parent directory")
# parser.add_argument('bop_dataset_name', nargs='?', help="Main BOP dataset")
# parser.add_argument('output_dir', nargs='?', help="Path to where the final files will be saved ")
# args = parser.parse_args()
args = {}
args['bop_parent_path'] = 'bop'
args['bop_dataset_name'] = 'ycbv'
args['output_dir'] = 'out'
args['obj_pose'] = 'predefined_poses/obj_poses_level1.npy'
args['cam_pose'] = 'predefined_poses/cam_poses_level1.npy'
args['poses'] = 'upper'
print(args['bop_parent_path'])
bproc.init()


poses = np.load(args['obj_pose'])

if args['poses'] == 'upper':
    cam_poses = np.load(args['cam_pose'])
    poses = poses[cam_poses[:, 2, 3] >= 0]
poses[:, :3, 3] *= 0.4
poses[:, :3, :3] = poses[:, :3, :3] / 1000.0
poses[:, :3, 3] = poses[:, :3, 3] / 1000.0
# load specified bop objects into the scene
print(poses.shape)
bop_objs = bproc.loader.load_bop_objs(bop_dataset_path=os.path.join(args['bop_parent_path'], args['bop_dataset_name']),
                                      mm2m=True)

# load BOP datset intrinsics
bproc.loader.load_bop_intrinsics(bop_dataset_path=os.path.join(args['bop_parent_path'], args['bop_dataset_name']))

light_locations = []
for x in [-1, 1]:
    for y in [-1, 1]:
        for z in [0, 1]:
            light_locations.append([x, y, z])

for location in light_locations:
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location(location)
    light.set_energy(50)

# bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")

# set shading
for j, obj in enumerate(bop_objs):
    obj.set_shading_mode('auto')
    obj.hide(True)




# activate depth rendering
bproc.renderer.enable_distance_output(True)

# bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(100)
bproc.renderer.set_output_format(enable_transparency=True)
# add segmentation masks (per class and per instance)
bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name", "bop_dataset_name"],
                                          default_values={"category_id": 0, "bop_dataset_name": None})

cam2world = bproc.math.change_source_coordinate_frame_of_transformation_matrix(
    np.eye(4), ["X", "-Y", "-Z"]
)
bproc.camera.add_camera_pose(cam2world)

# Render five different scenes
for obj in bop_objs:
    obj.hide(False)
    name = args['bop_dataset_name']
    for idx_frame, obj_pose in enumerate(poses):
        obj.set_local2world_mat(obj_pose)

        # render the cameras of the current scene
        data = bproc.renderer.render()

        obj_data_dir = os.path.join(args['output_dir'], name, f'obj_{obj.get_cp("category_id")}')
        os.makedirs(obj_data_dir, exist_ok=True)
        # Write data to bop format
        bproc.writer.write_bop(obj_data_dir,
                            dataset=name,
                            depths=data["distance"],
                            depth_scale=1.0,
                            colors=data["colors"],
                            color_file_format="PNG",
                            append_to_existing_output=True)
    obj.hide(True)