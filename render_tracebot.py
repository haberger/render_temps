import blenderproc as bproc
import argparse
import os
import numpy as np
import bpy
from PIL import Image
import shutil
import sys
import yaml


def load_needle(tracebot_objs, path):
    objs = bproc.loader.load_blend(path)

    needle = {}
    needle['parts'] = []
    needle['annos'] = []
    for obj in objs:
        name = obj.get_name()
        if name[0:5] == 'Empty':
            continue
        elif name == 'needle':
            needle['whole'] = obj
            obj.hide(True)
            obj.disable_rigidbody()
        else:
            obj.hide(True)
            obj.disable_rigidbody()
            if name == 'needle_without':
                needle['annos'].append(obj)
                obj.set_cp("category_id", 3)
                model_path = os.path.join(config['models_dir'], 'needle', obj.get_name() + '.ply')
                obj.set_cp('model_path', model_path)
                continue
            elif name == 'needle_cap':
                needle['annos'].append(obj)
                needle['parts'].append(obj)
                obj.set_cp("category_id", 4)
                model_path = os.path.join(config['models_dir'], 'needle', obj.get_name() + '.ply')
                obj.set_cp('model_path', model_path)
                continue
            else:
                needle['parts'].append(obj)
                obj.set_cp("category_id", 3)
                model_path = os.path.join(config['models_dir'], 'needle', obj.get_name() + '.ply')
                obj.set_cp('model_path', model_path)
        obj.hide(True)
        obj.disable_rigidbody()
    tracebot_objs["needle"] = needle
    return tracebot_objs

def load_needle_vr(tracebot_objs, path):
    objs = bproc.loader.load_blend(path)

    needle_vr = {}
    needle_vr['parts'] = []
    needle_vr['annos'] = []
    for obj in objs:
        name = obj.get_name()
        if name[0:5] == 'Empty':
            continue
        elif name == 'needle_vr':
            needle_vr['whole'] = obj
            obj.hide(True)
            obj.disable_rigidbody()
        else:
            obj.hide(True)
            obj.disable_rigidbody()
            if name == 'needle_without_vr':
                needle_vr['annos'].append(obj)
                obj.set_cp("category_id", 3)
                model_path = os.path.join(config['models_dir'], 'needle_vr', obj.get_name() + '.ply')
                obj.set_cp('model_path', model_path)
                continue
            elif name == 'needle_cap_vr':
                needle_vr['annos'].append(obj)
                needle_vr['parts'].append(obj)
                obj.set_cp("category_id", 4)
                model_path = os.path.join(config['models_dir'], 'needle_vr', obj.get_name() + '.ply')
                obj.set_cp('model_path', model_path)
                continue
            else:
                needle_vr['parts'].append(obj)
                obj.set_cp("category_id", 3)
                model_path = os.path.join(config['models_dir'], 'needle_vr', obj.get_name() + '.ply')
                obj.set_cp('model_path', model_path)
        obj.hide(True)
        obj.disable_rigidbody()
    tracebot_objs["needle_vr"] = needle_vr
    return tracebot_objs

def load_needle_vl(tracebot_objs, path):
    objs = bproc.loader.load_blend(path)

    needle_vl = {}
    needle_vl['parts'] = []
    needle_vl['annos'] = []
    for obj in objs:
        name = obj.get_name()
        if name[0:5] == 'Empty':
            continue
        elif name == 'needle_vl':
            needle_vl['whole'] = obj
            obj.hide(True)
            obj.disable_rigidbody()
        else:
            obj.hide(True)
            obj.disable_rigidbody()
            if name == 'needle_without_vl':
                needle_vl['annos'].append(obj)
                obj.set_cp("category_id", 3)
                model_path = os.path.join(config['models_dir'], 'needle_vl', obj.get_name() + '.ply')
                obj.set_cp('model_path', model_path)
                continue
            elif name == 'needle_cap_vl':
                needle_vl['annos'].append(obj)
                needle_vl['parts'].append(obj)
                obj.set_cp("category_id", 4)
                model_path = os.path.join(config['models_dir'], 'needle_vl', obj.get_name() + '.ply')
                obj.set_cp('model_path', model_path)
                continue
            else:
                needle_vl['parts'].append(obj)
                obj.set_cp("category_id", 3)
                model_path = os.path.join(config['models_dir'], 'needle_vl', obj.get_name() + '.ply')
                obj.set_cp('model_path', model_path)
        obj.hide(True)
        obj.disable_rigidbody()
    tracebot_objs["needle_vl"] = needle_vl
    return tracebot_objs

def load_needle_vd(tracebot_objs, path):
    objs = bproc.loader.load_blend(path)

    needle_vd = {}
    needle_vd['parts'] = []
    needle_vd['annos'] = []
    for obj in objs:
        name = obj.get_name()
        if name[0:5] == 'Empty':
            continue
        elif name == 'needle_vd':
            needle_vd['whole'] = obj
            obj.hide(True)
            obj.disable_rigidbody()
        else:
            obj.hide(True)
            obj.disable_rigidbody()
            if name == 'needle_without_vd':
                needle_vd['annos'].append(obj)
                obj.set_cp("category_id", 3)
                model_path = os.path.join(config['models_dir'], 'needle_vd', obj.get_name() + '.ply')
                obj.set_cp('model_path', model_path)
                continue
            elif name == 'needle_cap_vd':
                needle_vd['annos'].append(obj)
                needle_vd['parts'].append(obj)
                obj.set_cp("category_id", 4)
                model_path = os.path.join(config['models_dir'], 'needle_vd', obj.get_name() + '.ply')
                obj.set_cp('model_path', model_path)
                continue
            else:
                needle_vd['parts'].append(obj)
                obj.set_cp("category_id", 3)
                model_path = os.path.join(config['models_dir'], 'needle_vd', obj.get_name() + '.ply')
                obj.set_cp('model_path', model_path)
        obj.hide(True)
        obj.disable_rigidbody()
    tracebot_objs["needle_vd"] = needle_vd
    return tracebot_objs

def load_needle_vu(tracebot_objs, path):
    objs = bproc.loader.load_blend(path)

    needle_vu = {}
    needle_vu['parts'] = []
    needle_vu['annos'] = []
    for obj in objs:
        name = obj.get_name()
        
        if name[0:5] == 'Empty':
            continue
        elif name == 'needle_vu':
            obj.hide(True)
            obj.disable_rigidbody()
            needle_vu['whole'] = obj
        else:
            obj.hide(True)
            obj.disable_rigidbody()
            if name == 'needle_without_vu':
                needle_vu['annos'].append(obj)
                obj.set_cp("category_id", 3)
                model_path = os.path.join(config['models_dir'], 'needle_vu', obj.get_name() + '.ply')
                obj.set_cp('model_path', model_path)
                continue
            elif name == 'needle_cap_vu':
                needle_vu['annos'].append(obj)
                needle_vu['parts'].append(obj)
                obj.set_cp("category_id", 4)
                model_path = os.path.join(config['models_dir'], 'needle_vu', obj.get_name() + '.ply')
                obj.set_cp('model_path', model_path)
                continue
            else:
                needle_vu['parts'].append(obj)
                obj.set_cp("category_id", 3)
                model_path = os.path.join(config['models_dir'], 'needle_vu', obj.get_name() + '.ply')
                obj.set_cp('model_path', model_path)

    tracebot_objs["needle_vu"] = needle_vu
    return tracebot_objs

def load_white_clamp(tracebot_objs, path):
    objs = bproc.loader.load_blend(path)

    white_clamp = {}
    white_clamp['parts'] = []
    white_clamp['annos'] = []
    for obj in objs:
        name = obj.get_name()
        if name == 'Empty':
            continue
        elif name == 'clamp_w':
            white_clamp['whole'] = obj
            white_clamp['annos'].append(obj)
        else:
            white_clamp['parts'].append(obj)
        obj.set_cp("category_id", 9)
        model_path = os.path.join(config['models_dir'], 'clamp', obj.get_name() + '.ply')
        obj.set_cp('model_path', model_path)

    tracebot_objs["white_clamp"] = white_clamp

    return tracebot_objs    

def load_red_clamp(tracebot_objs, path):
    objs = bproc.loader.load_blend(path)

    red_clamp = {}
    red_clamp['parts'] = []
    red_clamp['annos'] = []
    for obj in objs:
        name = obj.get_name()
        if name == 'Empty':
            continue
        elif name == 'clamp_r':
            red_clamp['whole'] = obj
            red_clamp['annos'].append(obj)
        else:
            red_clamp['parts'].append(obj)
        obj.set_cp("category_id", 10)
        model_path = os.path.join(config['models_dir'], 'clamp', obj.get_name() + '.ply')
        obj.set_cp('model_path', model_path)
    tracebot_objs["red_clamp"] = red_clamp
    return tracebot_objs  

def load_red_cap(tracebot_objs, path):
    objs = bproc.loader.load_blend(path)

    red_cap = {}
    red_cap['parts'] = []
    red_cap['annos'] = []
    for obj in objs:
        name = obj.get_name()
        if name == 'Empty':
            continue
        elif name == 'red_cap':
            red_cap['whole'] = obj
            red_cap['annos'].append(obj)
        else:
            red_cap['parts'].append(obj)
        obj.set_cp("category_id", 5)
        model_path = os.path.join(config['models_dir'], 'red_cap', obj.get_name() + '.ply')
        obj.set_cp('model_path', model_path)
    tracebot_objs["red_cap"] = red_cap
    return tracebot_objs  

def load_yellow_cap(tracebot_objs, path):
    objs = bproc.loader.load_blend(path)

    yellow_cap = {}
    yellow_cap['parts'] = []
    yellow_cap['annos'] = []
    for obj in objs:
        name = obj.get_name()
        if name == 'Empty':
            continue
        elif name == 'yellow_cap':
            yellow_cap['whole'] = obj
            yellow_cap['annos'].append(obj)
        else:
            yellow_cap['parts'].append(obj)
        obj.set_cp("category_id", 8)
        model_path = os.path.join(config['models_dir'], 'yellow_cap', obj.get_name() + '.ply')
        obj.set_cp('model_path', model_path)
    tracebot_objs["yellow_cap"] = yellow_cap
    return tracebot_objs  

def load_canister(tracebot_objs, path):
    objs = bproc.loader.load_blend(path)

    canister = {}
    canister['parts'] = []
    canister['annos'] = []
    for obj in objs:
        name = obj.get_name()
        if name == 'Empty':
            continue
        elif name == 'canister':
            canister['whole'] = obj
            canister['annos'].append(obj)
        else:
            canister['parts'].append(obj)
        obj.set_cp("category_id", 6)
        model_path = os.path.join(config['models_dir'], 'canister', obj.get_name() + '.ply')
        obj.set_cp('model_path', model_path)

    tracebot_objs["canister"] = canister

    return tracebot_objs    

def load_small_bottle(tracebot_objs, path):
    objs = bproc.loader.load_blend(path)

    small_bottle = {}
    small_bottle['parts'] = []
    small_bottle['annos'] = []
    for obj in objs:
        name = obj.get_name()
        if name == 'Empty':
            continue
        elif name == 'small_bottle':
            small_bottle['whole'] = obj
            small_bottle['annos'].append(obj)
        else:
            small_bottle['parts'].append(obj)
        obj.set_cp("category_id", 2)
        model_path = os.path.join(config['models_dir'], 'small_bottle', obj.get_name() + '.ply')
        obj.set_cp('model_path', model_path)

    tracebot_objs["small_bottle"] = small_bottle
    return tracebot_objs  

def load_medium_bottle(tracebot_objs, path):
    objs = bproc.loader.load_blend(path)

    medium_bottle = {}
    medium_bottle['parts'] = []
    medium_bottle['annos'] = []
    for obj in objs:
        name = obj.get_name()
        if name == 'Empty':
            continue
        elif name == 'medium_bottle':
            medium_bottle['whole'] = obj
            medium_bottle['annos'].append(obj)
        else:
            medium_bottle['parts'].append(obj)
        obj.set_cp("category_id", 1)
        model_path = os.path.join(config['models_dir'], 'medium_bottle', obj.get_name() + '.ply')
        obj.set_cp('model_path', model_path)

    tracebot_objs["medium_bottle"] = medium_bottle
    return tracebot_objs  

def load_large_bottle(tracebot_objs, path):
    objs = bproc.loader.load_blend(path)

    large_bottle = {}
    large_bottle['parts'] = []
    large_bottle['annos'] = []
    for obj in objs:
        name = obj.get_name()
        if name == 'Empty':
            continue
        elif name == 'large_bottle':
            large_bottle['whole'] = obj
            large_bottle['annos'].append(obj)
        else:
            large_bottle['parts'].append(obj)
        obj.set_cp("category_id", 7)
        model_path = os.path.join(config['models_dir'], 'large_bottle', obj.get_name() + '.ply')
        obj.set_cp('model_path', model_path)

    tracebot_objs["large_bottle"] = large_bottle
    return tracebot_objs 



def render(config):
    bproc.init()
    # mesh_dir = os.path.join(dirname, config["models_dir"])

    dataset_name = config["dataset_name"]

    target_path = os.path.join(config['output_dir'], 'bop_data', dataset_name, 'models')
    if not os.path.isdir(target_path):
        shutil.copytree(config['models_dir'], target_path)

    poses = np.load(config['obj_pose'])

    if config['poses'] == 'upper':
        cam_poses = np.load(config['cam_pose'])
        poses = poses[cam_poses[:, 2, 3] >= 0]
    poses[:, :3, 3] *= 0.4
    poses[:, :3, :3] = poses[:, :3, :3] / 1000.0
    poses[:, :3, 3] = poses[:, :3, 3] / 1000.0
    # load specified bop objects into the scene
    print(poses.shape)

    tracebot = {}
    tracebot = load_needle(tracebot, os.path.join(config["models_dir"], 'needle/needle.blend'))
    # tracebot = load_needle_vd(tracebot, os.path.join(config["models_dir"], 'needle_vd/needle_vd.blend'))
    # tracebot = load_needle_vu(tracebot, os.path.join(config["models_dir"], 'needle_vu/needle_vu.blend'))
    # tracebot = load_needle_vl(tracebot, os.path.join(config["models_dir"], 'needle_vl/needle_vl.blend'))
    # tracebot = load_needle_vr(tracebot, os.path.join(config["models_dir"], 'needle_vr/needle_vr.blend'))

    tracebot = load_red_clamp(tracebot, os.path.join(config["models_dir"], 'clamp/clamp_red.blend'))
    tracebot = load_white_clamp(tracebot, os.path.join(config["models_dir"], 'clamp/clamp_white.blend'))
    tracebot = load_red_cap(tracebot, os.path.join(config["models_dir"], 'red_cap/red_cap.blend'))
    tracebot = load_yellow_cap(tracebot, os.path.join(config["models_dir"], 'yellow_cap/yellow_cap.blend'))
    tracebot = load_canister(tracebot, os.path.join(config["models_dir"], 'canister/canister.blend'))
    tracebot = load_small_bottle(tracebot, os.path.join(config["models_dir"], 'small_bottle/small_bottle.blend'))
    tracebot = load_large_bottle(tracebot, os.path.join(config["models_dir"], 'large_bottle/large_bottle.blend'))
    tracebot = load_medium_bottle(tracebot, os.path.join(config["models_dir"], 'medium_bottle/medium_bottle.blend'))

    cam2world = bproc.math.change_source_coordinate_frame_of_transformation_matrix(
        np.eye(4), ["X", "-Y", "-Z"]
    )
    bproc.camera.add_camera_pose(cam2world)

    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.set_max_amount_of_samples(100)
    bproc.renderer.enable_distance_output(True)
    max_bounces = 50
    bproc.renderer.set_light_bounces(
        glossy_bounces=max_bounces, 
        max_bounces=max_bounces, 
        transmission_bounces=max_bounces, 
        transparent_max_bounces=max_bounces, 
        volume_bounces=max_bounces)

    black_img = Image.new('RGB', (config["cam"]["width"], config["cam"]["height"]))


    bproc.camera.set_intrinsics_from_K_matrix(np.reshape(config["cam"]["K"], (3, 3)), 
                                                config["cam"]["width"], 
                                                config["cam"]["height"])
    

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


    for key in tracebot.keys():
        for obj in tracebot[key]["parts"]:
            obj.hide(True) 
        tracebot[key]["whole"].hide(True) 
        for obj in tracebot[key]["annos"]:
            obj.hide(True) 

    for obj in tracebot.keys():
        for part in tracebot[obj]["parts"]:
            part.hide(False)
        name = config['dataset_name']
        obj_data_dir = os.path.join(config['output_dir'], name, f'obj_{part.get_cp("category_id")}')           
        if not os.path.exists(obj_data_dir):
            os.makedirs(obj_data_dir)
            print(f"Directory '{obj_data_dir}' created.")
        else:
            print(f"Directory '{obj_data_dir}' already exists.")
        for idx_frame, obj_pose in enumerate(poses[::10]):
            for part in tracebot[obj]["parts"]:
                part.set_local2world_mat(obj_pose)
                part.set_scale([1,1,1])
            # break
            data = bproc.renderer.render()
            data.update(bproc.renderer.render_segmap(map_by="class", use_alpha_channel=True))
            # # Map distance to depth
            depth = bproc.postprocessing.dist2depth(data["distance"])[0]
            mask = np.uint8((depth < 1000) * 255)
            mask = Image.fromarray(mask)
            mask.save(os.path.join(obj_data_dir, "mask_{:06d}.png".format(idx_frame)))
            rgb = Image.fromarray(np.uint8(data["colors"][0]))
            # rgb.save(os.path.join(obj_data_dir, "{:06d}rgb.png".format(idx_frame)))
            img = Image.composite(rgb, black_img, mask)
            img.save(os.path.join(obj_data_dir, "{:06d}.png".format(idx_frame)))
        # break
        for part in tracebot[obj]["parts"]:
            part.hide(True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    dirname = os.path.dirname(__file__) #TODO
    #dirname = "/home/v4r/David/BlenderProc"

    #read config
    with open(os.path.join(dirname, args.config_path), "r") as stream:
        config = yaml.safe_load(stream)
    render(config)


    