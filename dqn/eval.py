import os,sys
import math
import random
import numpy as np
from itertools import count
from PIL import Image
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as M  
from torch.utils.tensorboard import SummaryWriter
torch.autograd.set_detect_anomaly(True)

from torchvision.utils import make_grid
from torchvision.utils import draw_bounding_boxes 
import torchvision.transforms as trn
import matplotlib.pyplot as plt
import torchvision.transforms.functional as FF
plt.rcParams["savefig.bbox"] = 'tight'

from config import args
from utils import normalize, inv_normalize, save_gif, print_config_info, print_memory_usage
from env import Environment
from dqn import Agent
from replay import ReplayMemory, save_memory, load_memory
from prioritized_memory import PrioritizedMemory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = T.Compose([T.ToPILImage(),
                    #T.Resize(args.image_size, interpolation=Image.CUBIC),
                    T.ToTensor(),
                    normalize,
                    ])


def evaluate(env, agent):
            
    agent.load_checkpoint(args.checkpoint)    

    #last model 
    done_training, avg_reward, avg_dur, avg_iou = view_current_policy(env, agent, prefix="eval")
   
    #eval model
    eval_best_ckpt = os.path.join(args.checkpoint_dir, "eval-dqn.pt")
    agent.load_checkpoint(filename=eval_best_ckpt)
    view_current_policy(env, agent, prefix="best-eval")

    return


def view_current_policy(env, agent, prefix=""):
    #print(f"{i_episode}: current policy")

    agent.evaluate()

    avg_rewards = 0
    avg_len = 0
    avg_iou = 0
    oob = 0

    num_samples = 10 if env.size == 1 and env.random_box else env.size
    random_box  = env.random_box 
    assert num_samples == 1 
    for s in range(num_samples):
        images = []
        rewards = 0
        ious    = 0
        solved = False
        
        state = env.reset(random_box=random_box).to(device)
        
        for t in count():
            #TODO: generate images with ground-truth bounding boxes
            images.append(env.render(show=False))
            
            encoded_state = agent.encode(state, env.curr_pos)
            action = agent.select_action(encoded_state, evaluate=True)
            next_state, reward, done, info = env.step(action.item(), verbose=False)
            print(t, action, info)
            rewards += reward
            ious += info["iou"]

            if done:
                solved = True if info["iou"] > args.iou_threshold else False
                break
            
            state = next_state.to(device)
      
        oob += int(solved)
        avg_len += t
        ious /= len(images)
        avg_rewards += rewards
        avg_iou += ious
        
        fn = os.path.join(args.gif_dir, f"{prefix}_{s:02d}_dur={t:02d}_rwd={rewards:.2f}_solved={solved}.gif")
        save_gif(images, fn, keep_last=5000)

    avg_len /= num_samples
    avg_rewards /= num_samples
    avg_iou /= num_samples

    print(f"  | evaluate policy:  solved = {oob}/{num_samples}  avg rewards = {avg_rewards:.2f}   avg steps = {avg_len:.2f} avg iou = {avg_iou:.4f}")
    done_training = True if oob == num_samples else False
    agent.evaluate()

    return done_training, avg_rewards, avg_len, avg_iou

def setup():

    
    if not os.path.exists(args.output_dir):
        print(f"{args.output_dir} does not exist")
        sys.exit(1)

    args.gif_dir = os.path.join(args.output_dir, "eval_gifs")
    args.checkpoint_dir = os.path.join(args.output_dir, args.checkpoint_dir)
    args.checkpoint = os.path.join(args.checkpoint_dir, "dqn.pt")

    #os.makedirs(args.checkpoint_dir, exist_ok=True)
    #os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.gif_dir, exist_ok=True)
   
    

    print_config_info(args)

    env = Environment(args.data_dir, transform, split="test",
                                             image_size=args.image_size,
                                             mode=args.mode,
                                             max_steps=args.max_steps,
                                             single=args.single, 
                                             random_box=args.random_box,
                                             iou_threshold=args.iou_threshold)
    print(env)

    num_channels = 4 if args.mode == "mask" else 3
    agent = Agent(env.num_actions, 
                  (num_channels,args.image_size),
                  args.arch, 
                  batch_size=args.batch_size,
                  gamma=args.gamma,
                  eps_start=args.eps_start, 
                  eps_end=args.eps_end, 
                  eps_decay=args.eps_decay).to(device)
    print(agent)


    return env, agent

if __name__ == "__main__":
    env, agent = setup()
    evaluate(env,agent)

    print("[All done!]")
