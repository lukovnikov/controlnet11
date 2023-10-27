import os
import subprocess
import fire
import glob


def main(device=3, simulate=False):
    expdirs = [
        "/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_dd_v5_exp_6",
        # "/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_dd_v5_exp_3",
        # "/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_legacy-NewEdiffipp_v5_exp_1",
        # "/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_posattn2_v5_exp_2",   
        # "/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_posattn5a_v5_exp_2",
        # "/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_posattn5a_v5_exp_4",
        # "/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_posattn5a_v5_exp_5",
        # "/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_cac_v5_exp_1",
        # "/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_global_v5_exp_1",
        # "/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_*",
    ]
    for expdir in expdirs:
        paths = glob.glob(expdir)
        paths = sorted(paths)
        for path in paths:
            print(path)
            examplesets = [
                # "evaldata/extradev.pkl",
                # "evaldata/threeballs1.pkl",
                # "evaldata/threecoloredballs1.pkl",
                # "evaldata/threesimplefruits1.pkl",
                # "evaldata/foursquares2.pkl",
                # "evaldata/openair1.pkl",
                # "evaldata/threeyellow1.pkl",
                "evaldata/threeorange1.pkl",
            ]
            examplesetstr = '"' + ",".join(examplesets) + '"'

            cmd = f'python generate_controlnet_pww.py --expdir={path} --examples={examplesetstr} --devices={device},'
            
            print("="*50)
            print("Runner: command:", cmd)
            print("="*50)
            
            if not simulate:
                ret = subprocess.call(cmd, shell=True)
            else:
                ret = "simulating cmd"
            
            print("Runner: returned value:", ret)
            print("="*50)
        
        
if __name__ == "__main__":
    fire.Fire(main)