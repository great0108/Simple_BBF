# https://github.com/NoSavedDATA/PyTorch-BBF-Bigger-Better-Faster-Atari-100k

import gymnasium as gym
import argparse
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import chain
import tqdm
import os
from multiprocessing import freeze_support

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from experience_replay import *
from model import *
from utils import *

import locale
locale.getpreferredencoding = lambda: "UTF-8"

import wandb

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

parser.add_argument('--load_path', default=None, type=str, help='load path for model')
parser.add_argument('--train', default=True, help='True: train model, False: eval model')
parser.add_argument('--eval_runs', default=50, help='episode numbers for evaluate model')

args = parser.parse_args()


if __name__ == "__main__":
    freeze_support()
    env_name = args.env_name

    wandb.init(
        project="Atari-100k-BBF",
        name=f"BBF-{env_name}",
    )

    # Optimization
    batch_size = args.batch_size
    lr=args.learning_rate
    eps=1e-8

    # Target network EMA rate
    critic_ema_decay=args.ema_decay

    # Return function
    initial_gamma=torch.tensor(1-args.initial_gamma).log()
    final_gamma=torch.tensor(1-args.final_gamma).log()

    initial_n = args.initial_n
    final_n = args.final_n

    num_buckets=args.num_buckets

    # Reset Schedule and Buffer
    reset_every=args.reset_freq # grad steps, not steps.
    schedule_max_step=reset_every//4
    total_steps=args.total_step

    prefetch_cap=1 # actually, no prefetch is being done


    Transition = namedtuple('Transition',
                            ('state', 'reward', 'action', 'c_flag'))
    memory = PrioritizedReplay_nSteps_Sqrt(total_steps+5, total_steps=schedule_max_step, prefetch_cap=prefetch_cap)

    env = gym.make(f"ALE/{env_name}-v5")
    n_actions = env.action_space.n

    # state, info = env.reset()
    # n_observations = len(state)

    model=DQN(n_actions).cuda()
    model_target=DQN(n_actions).cuda()

    model_target.load_state_dict(model.state_dict())

    if args.load_path:
        model.load_state_dict(torch.load(args.load_path)['model_state_dict'])
        model_target.load_state_dict(torch.load(args.load_path)['model_target_state_dict'])


    perception_modules=[model.encoder_cnn, model.transition]
    actor_modules=[model.prediction, model.projection, model.a, model.v]

    params_wm=[]
    for module in perception_modules:
        for param in module.parameters():
            if param.requires_grad==True: # They all require grad
                params_wm.append(param)

    params_ac=[]
    for module in actor_modules:
        for param in module.parameters():
            if param.requires_grad==True:
                params_ac.append(param)


    optimizer = torch.optim.AdamW(chain(params_wm, params_ac),
                                    lr=lr, weight_decay=0.1, eps=1.5e-4)

    mse = torch.nn.MSELoss(reduction='none')

    def optimize(step, grad_step, n):
        
        model.train()
        model_target.train()

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=False):
            with torch.no_grad():
                states, next_states, rewards, action, c_flag, idxs, is_w = memory.sample(n, batch_size, grad_step)
                z = model_target.encode(states[:,1:6])[0]
            terminal=1-c_flag
            #print(f"STUFF HERE {states.shape, rewards.shape, c_flag.shape, action.shape, n}")
        
            q, max_action, _, z_pred = model(states[:,0][:,None], action[:,:5].long())
            
            max_action  = model.get_max_action(next_states[:,n-1][:,None])
            next_values = model_target.evaluate(next_states[:,n-1][:,None].contiguous(), max_action)
            

            action = action[:,0,None].expand(batch_size,num_buckets)
            action=action[:,None]
            with torch.no_grad():
                gammas_one=torch.ones(batch_size,n,1,dtype=torch.float,device='cuda')
                gamma_step = 1-torch.tensor(( (schedule_max_step - min(grad_step, schedule_max_step)) / schedule_max_step) * (initial_gamma-final_gamma) + final_gamma).exp()
                gammas=gammas_one*gamma_step

                returns = []
                for t in range(n):
                    ret = 0
                    for u in reversed(range(t, n)):
                        # print(c_flag.shape, gammas.shape, rewards.shape)
                        ret += torch.prod(c_flag[:,t+1:u+1],-2)*torch.prod(gammas[:,t:u],-2)*rewards[:,u+1]
                    returns.append(ret)
                returns = torch.stack(returns,1)
            

            plot_vs = returns.clone().sum(-1)
            
            same_traj = (torch.prod(c_flag[:,:n],-2)).squeeze()
            
            returns = returns[:,0]
            returns = returns + torch.prod(gammas[0,:10],-2).squeeze()*same_traj[:,None]*model.support[None,:]
            returns = returns.squeeze()
            
            next_values = next_values[:,0]

            log_probs = torch.log(q[:,0].gather(-2, action)[:,None] + eps).contiguous()
            
            
            dist = project_distribution(returns, next_values.squeeze(), model.support)
            
            loss = -(dist*(log_probs.squeeze())).sum(-1).view(batch_size,-1).sum(-1)
            dqn_loss = loss.clone().mean()
            td_error = (loss + torch.nan_to_num((dist*torch.log(dist))).sum(-1)).mean()

            
            batched_loss = loss.clone()
            
            
            z = F.normalize(z, 2, dim=-1, eps=1e-5)
            z_pred = F.normalize(z_pred, 2, dim=-1, eps=1e-5)

            
            recon_loss = (mse(z_pred.contiguous().view(-1,2048), z.contiguous().view(-1,2048))).sum(-1)
            recon_loss = 5*(recon_loss.view(batch_size, -1).mean(-1))*same_traj
            
            
            loss += recon_loss
            
            loss = (loss*is_w).mean() # mean across batch axis

        loss.backward()

        param_norm, grad_norm = params_and_grad_norm(model)
        #scaler.scale(loss).backward()
        #scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        #scaler.step(optimizer)
        #scaler.update()
        
        optimizer.step()
        optimizer.zero_grad()
        
        #memory.set_priority(idxs, batched_loss)
        memory.set_priority(idxs, batched_loss, same_traj)
        
        
        lr = optimizer.param_groups[0]['lr']
        wandb.log({'loss': loss, 'dqn_loss': dqn_loss, 'recon_loss': recon_loss.mean(), 'lr': lr, 'returns': plot_vs.mean(),
                'buffer rewards': rewards.mean(0).sum(), 'is_w': is_w.mean(),
                'gamma': gamma_step, 'td_error': td_error, 'param_norm': param_norm.sum(), 'grad_norm': grad_norm.sum()})


    if args.train:
        scores=[]
        memory.free()
        step=0
        grad_step=0
        step=0

        progress_bar = tqdm.tqdm(total=total_steps)

        while step<(10):
            state, info = env.reset()
            state = preprocess(state).unsqueeze(0)

            states = deque(maxlen=4)
            for i in range(4):
                states.append(state)
            
            
            eps_reward=torch.tensor([0], dtype=torch.float)
            
            reward=np.array([0])
            done_flag=np.array([False])
            terminated=np.array([False])

            last_lives=np.array([0])
            life_loss=np.array([0])
            resetted=np.array([0])
            
            last_grad_update=0
            while step<(total_steps):
                progress_bar.update(1)
                model_target.train()
                
                len_memory = len(memory)
                
                #if resetted[0]>0:
                #    states = env.noop_steps(states)
                    
                Q_action = model_target.env_step(torch.cat(list(states),-3).unsqueeze(0))
                
                action = epsilon_greedy(Q_action, n_actions, len_memory).cpu()
                
                memory.push(torch.cat(list(states),-3).detach().cpu(), torch.tensor(reward,dtype=torch.float), action,
                            torch.tensor(np.logical_or(done_flag, life_loss),dtype=torch.bool))
                # print('action', action, action.shape)

                state, reward, terminated, truncated, info = env.step(action.numpy())
                state = preprocess(state).unsqueeze(0)
                states.append(state)
                reward = np.array([reward])
                terminated = np.array([terminated])
                truncated = np.array([truncated])
                
                eps_reward+=reward
                reward = np.clip(reward, -1, 1)

                
                done_flag = np.logical_or(terminated, truncated)
                lives = np.array([info['lives']])
                life_loss = np.clip(last_lives-lives, 0, None)
                resetted = np.clip(lives-last_lives, 0, None)
                last_lives = lives

                
                n = int(initial_n * (final_n/initial_n)**(min(grad_step,schedule_max_step) / schedule_max_step))
                n = np.array(n).item()
                
                memory.priority[len_memory] = memory.max_priority()
                

                if len_memory>2000:
                    for i in range(args.replay_ratio):
                        optimize(step, grad_step, n)
                        target_model_ema(model, model_target)
                        grad_step+=1

                
                if ((step+1)%10000)==0:
                    save_checkpoint(model, model_target, optimizer, step,
                                    'checkpoints/atari_last.pth')

                
                
                    
                
                if grad_step>reset_every:
                    #eval()
                    print('Reseting on step', step, grad_step)
                    
                    #seed_np_torch(random.randint(SEED-1000, SEED+1000)+step)
                    random_model = DQN(n_actions).cuda()
                    model.hard_reset(random_model)
                    
                    #seed_np_torch(random.randint(SEED-1000, SEED+1000)+step)
                    random_model = DQN(n_actions).cuda()
                    model_target.hard_reset(random_model)
                    
                    random_model=None
                    grad_step=0

                    actor_modules=[model.prediction, model.projection, model.a, model.v]
                    params_ac=[]
                    for module in actor_modules:
                        for param in module.parameters():
                            params_ac.append(param)
                            

                    perception_modules=[model.encoder_cnn, model.transition]
                    params_wm=[]
                    for module in perception_modules:
                        for param in module.parameters():
                            params_wm.append(param)
                    
                    optimizer_aux = torch.optim.AdamW(params_wm, lr=lr, weight_decay=0.1, eps=1.5e-4)
                    copy_states(optimizer, optimizer_aux)
                    optimizer = torch.optim.AdamW(chain(params_wm, params_ac),
                                        lr=lr, weight_decay=0.1, eps=1.5e-4)
                    copy_states(optimizer_aux, optimizer)
                
                step+=1
                
                log_t = done_flag.astype(float).nonzero()[0]
                
                if len(log_t)>0:
                    for log in log_t:
                        wandb.log({'eps_reward': eps_reward[log].sum()})
                        scores.append(eps_reward[log].clone())

                    eps_reward[log_t]=0
                    state, info = env.reset()
                    state = preprocess(state).unsqueeze(0)

                    states = deque(maxlen=4)
                    for i in range(4):
                        states.append(state)

        save_checkpoint(model, model_target, optimizer, step, f'checkpoints/{env_name}.pth')


    def eval_phase(eval_runs=50, max_eval_steps=27000, num_envs=1):
        progress_bar = tqdm.tqdm(total=eval_runs)
        
        scores=[]
        
        state, info = env.reset()
        state = preprocess(state).unsqueeze(0)
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
            #seed_np_torch(SEED+eval_run)
            env.seed=eval_run
            model_target.train()
            
            #if resetted[0]>0:
            #    states = env.noop_steps(states)
            
            Q_action = model_target.env_step(torch.cat(list(states),-3).unsqueeze(0))
            action = epsilon_greedy(Q_action.squeeze(), n_actions, 5000, 0.0005, num_envs).cpu()
            
            state, reward, terminated, truncated, info = env.step(action.numpy())
            state = preprocess(state).unsqueeze(0)
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
                state = preprocess(state).unsqueeze(0)

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