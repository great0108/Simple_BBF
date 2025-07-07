# https://github.com/NoSavedDATA/PyTorch-BBF-Bigger-Better-Faster-Atari-100k

import gymnasium as gym
import argparse
import matplotlib.pyplot as plt
from collections import deque
import tqdm
import numpy as np
import torch

from experience_replay import *
from model import BBF_Model
from BBF import BBF
from utils import *

import locale
locale.getpreferredencoding = lambda: "UTF-8"


parser = argparse.ArgumentParser('arguments for bbf')

args = parser.parse_args()
parser.add_argument('--env_name', default="Breakout", type=str, help='atari environment name')
parser.add_argument('--total_step', default=102000, type=float, help='total step')

parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate')
parser.add_argument('--ema_decay', default=0.995, type=float, help='Target network EMA rate')
parser.add_argument('--initial_gamma', default=0.97, type=float, help='initial gamma')
parser.add_argument('--final_gamma', default=0.997, type=float, help='final gamma')
parser.add_argument('--initial_n', default=10, type=int, help='initial n-step')
parser.add_argument('--final_n', default=3, type=int, help='final n-step')
parser.add_argument('--num_buckets', default=51, type=int, help='number of buckets for distributional RL')
parser.add_argument('--reset_freq', default=40000, type=int, help='model reset frequency in grad step')
parser.add_argument('--replay_ratio', default=2, type=int, help='number of train steps in one step')
parser.add_argument('--weight_decay', default=2, type=int, help='number of train steps in one step')

parser.add_argument('--save_path', default="./checkpoints", type=str, help='save path for model')
parser.add_argument('--load_path', default="./checkpoints/BBF_2000_steps.pth", type=str, help='load path for model')
parser.add_argument('--train', default=False, help='True: train model, False: eval model')
parser.add_argument('--eval_runs', default=50, help='episode numbers for evaluate model')

args = parser.parse_args()


if __name__ == "__main__":
    env = gym.make(f"ALE/{args.env_name}-v5")
    n_actions = env.action_space.n

    model = BBF_Model(
        n_actions, # env action space size
        hiddens=2048, # representation dim
        scale_width=4,  # cnn channel scale
        num_buckets=51,  # buckets in distributional RL
        Vmin=-10,  # min value in distributional RL
        Vmax=10, # max value in distributional RL
        resize=(96, 72) # input resize
    ).cuda()

    agent = BBF(
        model,
        env,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        ema_decay=args.ema_decay, # target model ema decay
        initial_gamma=args.initial_gamma, # starting gamma
        final_gamma=args.final_gamma, # final gamma
        initial_n=args.initial_n, # starting n-step
        final_n=args.final_n, # final n-step
        num_buckets=args.num_buckets, # buckets in distributional RL
        reset_freq=args.reset_freq, # reset schedule in grad step
        replay_ratio=args.replay_ratio, # update number in one step
        weight_decay=args.weight_decay # weight decay in optimizer,
    )

    if args.load_path:
        agent.load(args.load_path)

    if args.train:
        agent.learn(
            total_timesteps=args.total_step,
            save_freq=1000,
            save_path=args.save_path,
            name_prefix="BBF", # save file name prefix
            project_name="Atari-100k-BBF",  # wandb project name
            exp_name=f"BBF-{args.env_name}" # wandb experience name
        )


    def eval_phase(eval_runs=50, max_eval_steps=27000, num_envs=1):
        progress_bar = tqdm.tqdm(total=eval_runs)
        
        scores=[]
        
        state, info = env.reset()
        state = model.preprocess(state).unsqueeze(0)
        print(f"init state {state.shape}")
        
        states = deque(maxlen=4)
        for i in range(4):
            states.append(state)
        
        
        eps_reward=torch.tensor([0]*num_envs, dtype=torch.float)
        
        reward=np.array([0]*num_envs)
        terminated=np.array([False]*num_envs)
        
        last_lives=np.array([0]*num_envs)
        life_loss=np.array([0]*num_envs)
        resetted=np.array([0])

        finished_envs=np.array([False]*num_envs)
        done_flag=0
        last_grad_update=0
        eval_run=0
        step=np.array([0]*num_envs)
        while eval_run<eval_runs:
            #if resetted[0]>0:
            #    states = env.noop_steps(states)
            
            Q_action = model.env_step(torch.cat(list(states),-3).unsqueeze(0))
            action = epsilon_greedy(Q_action.squeeze(), n_actions, 5000, 0.001, num_envs).cpu()
            
            state, reward, terminated, truncated, info = env.step(action.numpy())
            state = model.preprocess(state).unsqueeze(0)
            states.append(state)
            reward = np.array([reward])
            terminated = np.array([terminated])
            truncated = np.array([truncated])
            
            eps_reward+=reward

            
            done_flag = np.logical_or(terminated, truncated)
            lives = np.array([info['lives']])
            life_loss = (last_lives-lives).clip(min=0)
            resetted = (lives-last_lives).clip(min=0)
            last_lives = lives        
            
            step+=1
            
            log_t = done_flag.astype(float).nonzero()[0]
            if len(log_t)>0:# or (step>max_eval_steps).any():
                progress_bar.update(1)
                for log in log_t:
                    #wandb.log({'eval_eps_reward': eps_reward[log].sum()})
                    if finished_envs[log]==False:
                        scores.append(eps_reward[log].clone())
                        eval_run+=1
                        #finished_envs[log]=True
                    step[log]=0
                    
                eps_reward[log_t]=0            
                for i, log in enumerate(step>max_eval_steps):
                    if log==True and finished_envs[i]==False:
                        scores.append(eps_reward[i].clone())
                        step[i]=0
                        eval_run+=1
                        eps_reward[i]=0
                        #finished_envs[i]=True

                state, info = env.reset()
                state = model.preprocess(state).unsqueeze(0)

                states = deque(maxlen=4)
                for i in range(4):
                    states.append(state)
                
        return scores


    if not args.train:
        scores = eval_phase(args.eval_runs, max_eval_steps=20000, num_envs=1)    
        scores = torch.stack(scores)
        scores, _ = scores.sort()
        
        _25th = args.eval_runs//4

        iq = scores[_25th:-_25th]
        iqm = iq.mean()
        iqs = iq.std()

        print(f"Scores Mean {scores.mean()}")
        print(f"Inter Quantile Mean {iqm}")
        print(f"Inter Quantile STD {iqs}")

        
        plt.xlabel('Episode (Sorted by Reward)')
        plt.ylabel('Reward')
        plt.plot(scores)
        
        # new_row = {'env_name': env_name, 'mean': scores.mean().item(), 'iqm': iqm.item(), 'std': iqs.item()}
        # df = pd.read_csv('results.csv',sep=',')
        # df.loc[len(df.index)] = new_row    
        # df.to_csv('results.csv', index=False)

        # with open(f'results/{env_name}.txt', 'w') as f:
        #     f.write(f" Scores Mean {scores.mean()}\n Inter Quantile Mean {iqm}\n Inter Quantile STD {iqs}")
        
        # print(scores)

    # os.system("powercfg -h off")
    # os.system("rundll32.exe powrprof.dll SetSuspendState")