
import argparse
desc="ECE 559B Final Project: Box-Finder"
parser = argparse.ArgumentParser(description=desc)


parser.add_argument("--vis", action="store_true",
        help="show training")

parser.add_argument("--resume", action="store_true",
        help="resume from previous checkpoint")

parser.add_argument("--random-box", action="store_true",
        help="start from a random location")

parser.add_argument("-d", "--data-dir", type=str,
        default="/home/memet/Projects/data/oxford-III-pets",
        help="path to data dir")

parser.add_argument("-c", "--checkpoint-dir", type=str,
        default="checkpoints",
        help="dir to save/load checkpoints")

parser.add_argument("-o", "--output-dir", type=str,
        default="results",
        help="dir to save results")

parser.add_argument("-a", "--arch", type=str,
        default="qnet",
        choices=["qnet", "qnet+googlenet", "qnet+resnet50", "qnetv2+googlenet", "qnetv2+resnet50"],
        help="Pretrained feature extractor from model zoo")

parser.add_argument("-s", "--single", action="store_true",
        help="only run on a single image")

parser.add_argument("-it", "--iou-threshold", type=float,
        default=0.5,
        help="iou threshold")

parser.add_argument("-f", "--save-freq", type=int,
        default=50,
        help="Save model/memory/gifs frequency")

parser.add_argument("-b", "--batch-size", type=int,
        default=64,
        help="Batch size for updating target_net")

parser.add_argument("-g", "--gamma", type=float,
        default=0.999,
        help="decay factor")

parser.add_argument("-es", "--eps-start", type=float,
        default=0.9,
        help="epsilon-greedy initial value")

parser.add_argument("-ee", "--eps-end", type=float,
        default=0.05,
        help="epsilon-greedy min value")

parser.add_argument("-ed", "--eps-decay", type=int,
        default=10000,
        help="epsilon decay rate (step/decay)")

parser.add_argument("-t", "--target-update", type=int,
        default=100,
        help="update target_net with policy_net weights every N episodes")

parser.add_argument("-i", "--image-size", type=int,
        default=224,
        help="image size, must be 224x224 if using pretrained networks")

parser.add_argument("-n", "--num-episodes", type=int,
        default=1000,
        help="max number of episodes to run")

parser.add_argument("-T", "--max-steps", type=int,
        default=0, 
        help="Max number of steps per episode")

parser.add_argument("-m", "--memory-size", type=int,
        default=10000,
        help="maximum size for the replay buffer")

parser.add_argument("--mode", type=str,
        default="mask", choices=["mask", "draw", "poster"],
        help="type of state representation")

parser.add_argument("--prioritized", action="store_true",
        help="use prioritized replay memory")


#For compute canada training
parser.add_argument('--state-file', default=None, type=str, help='')

args = parser.parse_args()
