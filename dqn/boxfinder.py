import os
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


def train(env, test_env, agent, memory, writer):
    
    episode_scores = []
    best_score = -np.inf
    best_avg_reward = -np.inf
    start_episode = 1

    if args.resume:
        try:
            info = agent.load_checkpoint(args.checkpoint)    
            start_episode = info["i_episode"]
            best_score = info["best_score"]
        except:
            print("!! Unable to load previous checkpoint")

        try:
            memory = load_memory(args.memory) 
        except:
            print("!! Unable to load previous memory: {args.memory}")

    for i_episode in range(start_episode, args.num_episodes+1):
        print(f"episode {i_episode}: ", end=" ")
        total_reward = 0 
        losses       = []
        ious         = []
        images       = []

        # Initialize the environment and state
        # state: image + bounding box == image_size 
        state = env.reset()
        state = state.to(device)
        encoded_state = agent.encode(state, env.curr_pos)
        
        #env.render(show=True)         

        for t in count():

            if i_episode % 100 == 0:
                images.append(env.render(show=False))
            
            # epsilon greedy selection give current state 
            action       = agent.select_action(encoded_state)
            
            # get reward for (state,action)
            # determine if next_state is a terminal state
            # get next state
            next_state, reward, done, info = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            ious.append(info["iou"])
        
            #env.render(show=True)         

            # Observe new state
            enc_next_state = None
            if not done:
                next_state = next_state.to(device)
                enc_next_state = agent.encode(next_state, env.curr_pos)

            # Store the transition in memory
            # Use the encoded state which is smaller
            if args.prioritized:
                if enc_next_state is None:
                    enc_next_state = agent.encode(next_state, env.curr_pos)
                error = agent.get_sample_error(encoded_state, action, enc_next_state, reward, done)
                memory.push(error, encoded_state, action, enc_next_state, reward, done)
            else:
                memory.push(encoded_state, action, enc_next_state, reward, done)

            # Move to the next state
            state = next_state
            encoded_state = enc_next_state

            # Perform one step of the optimization 
            loss = agent.optimize(memory)
           
            # Logging
            total_reward += reward
            if loss is not None:
                losses.append(loss)

            if done: #or t == args.max_steps: #safety stop to prevent continuous task
                episode_scores.append(total_reward.item())
                episode_scores = episode_scores[-20:] #rolling average, last 20 episodes
                #print(f"{i_episode}: running rewards = {episode_scores}")
                break
       
        """
        if i_episode % 100 == 0:
            print(len(images))
            fn = os.path.join(args.gif_dir, f"train_{i_episode:06d}_dur={t:02d}_rwd={total_reward.item():.2f}.gif")
            save_gif(images, fn)
        """

        writer.add_scalar("iou/train", np.mean(ious), i_episode)
        writer.add_scalar("reward/train", total_reward, i_episode)
        writer.add_scalar("duration/train", t, i_episode)
        if len(losses) > 0:
            writer.add_scalar("loss/train", np.mean(losses), i_episode)
            
        # Update the target network
        if i_episode % args.target_update == 0:
            print("  | updating target network")
            agent.update_target()

        if best_score <= total_reward.item():
            print(f"  | new best_score: {best_score:.4f} -> {total_reward.item():.4f}") 
            best_score = total_reward.item()

        if i_episode % args.save_freq == 0:
            info = {"i_episode" : i_episode + 1,
                    "best_score" : best_score,
                   }
            agent.save_checkpoint(info, filename=args.checkpoint)
            save_memory(memory, args.memory)
            print_memory_usage()

            done_training, avg_reward, avg_dur, avg_iou = view_current_policy(test_env, agent, f"{i_episode:06d}")
            
            print(f"  | best avg reward ={best_avg_reward:.2f} (avg reward={np.mean(episode_scores):.2f}, avg iou={np.mean(ious):.2f})")
           
            if best_avg_reward <= avg_reward:
                best_avg_reward = avg_reward
                info = {"i_episode" : i_episode + 1,
                        "best_score" : best_avg_reward,
                       }
                eval_best_ckpt = os.path.join(args.checkpoint_dir, "eval-dqn.pt")
                agent.save_checkpoint(info, filename=eval_best_ckpt)


            writer.add_scalar("reward/val", avg_reward, i_episode)
            writer.add_scalar("duration/val", avg_dur, i_episode)
            writer.add_scalar("iou/val", avg_iou, i_episode)

            if done_training:
                print("Completely solved {env.size} images with current policy")
                break
    
    view_current_policy(env, agent, prefix="eval")
    
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

    for s in range(num_samples):
        images = []
        rewards = 0
        ious    = 0
        solved = False
        state = env.reset(random_box=random_box).to(device)
        
        for t in count():
            images.append(env.render(show=False))
            
            encoded_state = agent.encode(state, env.curr_pos)
            action = agent.select_action(encoded_state, evaluate=True)
            next_state, reward, done, info = env.step(action.item(), verbose=False)
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

    if not args.resume and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    args.gif_dir = os.path.join(args.output_dir, "gifs")
    args.log_dir = os.path.join(args.output_dir, "logs")
    args.checkpoint_dir = os.path.join(args.output_dir, args.checkpoint_dir)
    args.checkpoint = os.path.join(args.checkpoint_dir, "dqn.pt")
    args.memory = os.path.join(args.checkpoint_dir, "memory.pkl")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.gif_dir, exist_ok=True)
    
    print_config_info(args)

    env = Environment(args.data_dir, transform, split="train",
                                             image_size=args.image_size,
                                             mode=args.mode,
                                             max_steps=args.max_steps,
                                             single=args.single, 
                                             random_box=args.random_box,
                                             iou_threshold=args.iou_threshold)
    print(env)
    
    test_env = Environment(args.data_dir, transform, split="test",
                                             image_size=args.image_size,
                                             mode=args.mode,
                                             max_steps=args.max_steps,
                                             single=args.single, 
                                             random_box=args.random_box,
                                             iou_threshold=args.iou_threshold)

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


    if args.prioritized:
        memory = PrioritizedMemory(args.memory_size)
    else:
        memory = ReplayMemory(args.memory_size)
    writer = SummaryWriter(log_dir=args.log_dir)


    return env, test_env, agent, memory, writer

if __name__ == "__main__":
    env, test_env, agent, memory, writer = setup()
    train(env,test_env, agent,memory, writer)
    writer.close()
    if args.state_file is not None:
        os.remove(args.state_file) 

    print("[All done!]")
