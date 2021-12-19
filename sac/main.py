import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
import torchvision.transforms as T
from PIL import Image
import os,sys
import warnings

from utils import save_gif

import os
path="/home/adash/data"
if os.path.exists(path):
    os.environ['TORCH_HOME'] = os.path.join(path, 'model_zoo')
else:
    path="/home/memet/Projects/data"
    os.environ['TORCH_HOME'] = os.path.join(path, 'model_zoo')

print(f'| set TORCH_HOME: {os.path.join(path, "model_zoo")}')

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument("--data-dir", default="/home/memet/Projects/data/oxford-III-pets", type=str)
parser.add_argument("--output-dir", default="results", type=str)
parser.add_argument('--env-name', default="boxfinder",
                    help='environment')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--no-cuda', action="store_true",
                    help='dont run on CUDA (default: False)')
parser.add_argument("--mode", type=str,
        default="draw", choices=["mask", "draw", "poster"],
        help="type of state representation")
parser.add_argument("-a", "--arch", type=str,
        default="qnet",
        choices=["qnet", "qnet+googlenet", "qnet+resnet50", "qnetv2+googlenet", "qnetv2+resnet50"],
        help="Pretrained feature extractor from model zoo")
parser.add_argument("-T", "--max-steps", type=int,
        default=1000, 
        help="Max number of steps per episode")
parser.add_argument("-f", "--save-freq", type=int,
        default=50,
        help="Save model/memory/gifs frequency")
parser.add_argument("-it", "--iou-threshold", type=float,
        default=0.5,
        help="iou threshold")
parser.add_argument("-s", "--single", action="store_true",
        help="only run on a single image")

#For compute canada training
parser.add_argument('--state-file', default=None, type=str, help='')
parser.add_argument("--resume", action="store_true",
        help="resume from previous checkpoint")



args = parser.parse_args()


mean  = [0.485, 0.456, 0.406]
std   = [0.229, 0.224, 0.225]
normalize = T.Normalize(mean, std)
transform = T.Compose([T.ToPILImage(),
                    #T.Resize(224, interpolation=Image.CUBIC),
                    T.ToTensor(),
                    normalize,
                    ])

#Folder and file setup
mode=args.mode
log_dir = os.path.join(os.path.join(args.output_dir, "logs"))
gif_dir = os.path.join(os.path.join(args.output_dir, "gifs"))
ckpt_dir = os.path.join(os.path.join(args.output_dir, "checkpoints"))
agent_fn = os.path.join(ckpt_dir, f"sac_{args.env_name}_{mode}.pt")
memory_fn = os.path.join(ckpt_dir, f"sac_buffer_{args.env_name}_{mode}.pkl")

os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(gif_dir, exist_ok=True)

# Environment
warnings.warn("hard-coded for image size 224")
from env import Environment
env = Environment(args.data_dir, transform, mode=mode, split="train", iou_threshold=args.iou_threshold, single=args.single, image_size=224, max_steps=args.max_steps)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
warnings.warn("hard-coded for resnet50")
import torchvision.models as M  
encoder      = getattr(M, "resnet50")(pretrained=True)
num_inputs   = encoder.fc.in_features

activation = {} 
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
encoder.avgpool.register_forward_hook(get_activation('avgpool'))

for param in encoder.parameters():    
    param.requires_grad = False
agent = SAC(num_inputs, env.action_space, args)
ckpt = agent.load_checkpoint(agent_fn)

#Tesnorboard
writer = SummaryWriter(log_dir)

# Memory
memory = ReplayMemory(args.replay_size, args.seed)
memory.load_buffer(memory_fn)

# Training Loop
total_numsteps = 0
updates = 0

def encode(x):
    output = encoder(x)
    return activation['avgpool'].view(x.size(0), -1)

best_avg_reward = ckpt["args"].best_avg_reward if ckpt is not None else -np.inf 
start_episode = ckpt["args"].i_episode if ckpt is not None else 1
for i_episode in itertools.count(start_episode, 1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
            #print("random:", action)
        else:
            state_t = encode(state) 
            action = agent.select_action(state_t)[0]  # Sample action from policy
            #print("agent:", action)

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1

        next_state, reward, done, _ = env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        #mask = 1 if episode_steps == env.max_steps else float(not done)
        mask = float(not done)

        memory.push(encode(state).squeeze(1), action, reward, encode(next_state).squeeze(1), mask) # Append transition to memory

        state = next_state

    if total_numsteps > args.num_steps:
        memory.save_buffer(memory_fn)
        break

    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    if i_episode % 10 == 0 and args.eval is True:
        avg_reward = 0.
        episodes = 1 if args.single else 10
        total_solved = 0 
        for t  in range(episodes):
            solved=False
            state = env.reset()
            episode_reward = 0
            done = False
            images = [env.render(show=False)]
            for _ in range(0,50): #safety kill
                state_t = encode(state) 
                action = agent.select_action(state_t, evaluate=True)[0]
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward

                state = next_state
                images.append(env.render(show=False))

                if done:
                    solved=True
                    total_solved += 1
                    break
            avg_reward += episode_reward
           
            try:
                fn = os.path.join(gif_dir, f"{i_episode:02d}_{t:02d}_{episode_reward:.4f}_solved={solved}.gif") 
                save_gif(images, fn, keep_last=1000)
            except:
                print("Evaluate image save failed.")

        avg_reward /= episodes


        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")

        args.i_episode = i_episode + 1
        if avg_reward >= best_avg_reward or total_solved == episodes:
            best_avg_reward = avg_reward
            args.best_avg_reward = avg_reward
            agent.save_checkpoint(agent_fn, args)
       
        if i_episode % args.save_freq == 0:
            agent.save_checkpoint(os.path.join(ckpt_dir, f"sac_{args.env_name}_{mode}-latest.pt"), args) 
        memory.save_buffer(memory_fn)

        if total_solved == episodes: #All were solved 
            print(f"Solved! Stopping training")
            break

env.close()
if args.state_file is not None:
    os.remove(args.state_file) 

