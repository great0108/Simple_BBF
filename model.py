import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import renormalize

def init_relu(module):
    if type(module) in (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d):
        nn.init.orthogonal_(module.weight, gain=1.41421)

        if module.bias is not None:
            nn.init.zeros_(module.bias)

def init_xavier(module):
    if type(module) in (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d):
        nn.init.xavier_uniform_(module.weight, gain=1)

        if module.bias is not None:
            nn.init.zeros_(module.bias)

class MLP(nn.Module):
    def __init__(self, in_hiddens=512, med_hiddens=512, out_hiddens=512, layers=1,
                 init=init_relu, in_act=nn.SiLU(), out_act=nn.Identity(),
                 ln_eps=1e-3, last_init=init_xavier, bias=True):
        super().__init__()
        # Special MLP with custom options for non last layer and last layer Linears.

        modules=[]
        self.init=init
        self.last_init=last_init
        
        hiddens=in_hiddens
        _out_hiddens = med_hiddens
        act = in_act
        for l in range(layers):
            last_layer = l==(layers-1)
            if last_layer:
                _out_hiddens = out_hiddens
                act = out_act
            modules.append(nn.Linear(hiddens, _out_hiddens, bias=bias))
            
            modules.append(act)
            hiddens=med_hiddens
        self.mlp=nn.Sequential(*modules)
        #print(self.mlp)

        self.init_weights()

    def turn_off_grads(self):
        for layer in self.mlp:
            if hasattr(layer, 'weight'):
                layer.weight.requires_grad=False
            if hasattr(layer, 'bias'):
                layer.bias.requires_grad=False

    def init_weights(self):
        self.mlp.apply(self.init)
        self.mlp[-2].apply(self.last_init)
        
        
    def forward(self,X):
        return self.mlp(X)

class DQN_Conv(nn.Module):
    def __init__(self, in_hiddens, hiddens, ks, stride, padding=0, max_pool=False, norm=True, init=init_relu, act=nn.SiLU(), bias=True, layers=1):
        super().__init__()

        if layers == 1:
            conv = nn.Conv2d(in_hiddens, hiddens, ks, stride, padding, bias=bias)
        elif layers == 2:
            conv = nn.Sequential(
                nn.Conv2d(in_hiddens, hiddens, ks, stride, padding, bias=bias),
                nn.Conv2d(hiddens, hiddens, ks, stride, padding, bias=bias)
            )

        
        self.conv = nn.Sequential(#nn.Conv2d(in_hiddens, hiddens, ks, stride, padding, padding_mode='replicate'),
                                  conv,
                                  nn.MaxPool2d(3,2,padding=1) if max_pool else nn.Identity(),
                                  nn.BatchNorm2d(hiddens, eps=1e-6) if norm else nn.Identity(),
                                  act,
                                  )
        self.conv.apply(init)
        
    def forward(self, X):
        return self.conv(X)


class DQN(nn.Module):
    def __init__(self, n_actions, hiddens=2048, mlp_layers=1, scale_width=4,
                 num_buckets=51, Vmin=-10, Vmax=10):
        super().__init__()
        self.support = torch.linspace(Vmin, Vmax, num_buckets).cuda()
        
        self.hiddens=hiddens
        self.scale_width=scale_width
        self.act = nn.ReLU()
        self.num_buckets = num_buckets
        self.n_actions = n_actions
        
        # self.encoder_cnn = IMPALA_Resnet(scale_width=scale_width, norm=False, init=init_xavier, act=self.act)
        self.encoder_cnn = nn.Sequential(
            DQN_Conv(12, 32*scale_width, 3, 1, 1, norm=True, max_pool=True, init=init_xavier, act=self.act, layers=2),
            DQN_Conv(32*scale_width, 32*scale_width, 3, 1, 1, norm=True, max_pool=True, init=init_xavier, act=self.act, layers=2),
            DQN_Conv(32*scale_width, 32*scale_width, 3, 1, 1, norm=True, max_pool=True, init=init_xavier, act=self.act, layers=2),
            DQN_Conv(32*scale_width, 32*scale_width, 3, 1, 1, norm=False, init=init_xavier, act=self.act, layers=2)
        )
    
        # Single layer dense that maps the flattened encoded representation into hiddens.
        self.projection = MLP(13824, med_hiddens=hiddens, out_hiddens=hiddens,
                              last_init=init_xavier, layers=1)
        self.prediction = MLP(hiddens, out_hiddens=hiddens, layers=1, last_init=init_xavier)
                                              
        self.transition = nn.Sequential(DQN_Conv(32*scale_width+n_actions, 32*scale_width, 3, 1, 1, norm=False, init=init_xavier, act=self.act),
                                        DQN_Conv(32*scale_width, 32*scale_width, 3, 1, 1, norm=False, init=init_xavier, act=self.act))

        # Single layer dense that maps hiddens into the output dim according to:
        # 1. https://arxiv.org/pdf/1707.06887.pdf -- Distributional Reinforcement Learning
        # 2. https://arxiv.org/pdf/1511.06581.pdf -- Dueling DQN
        self.a = MLP(hiddens, out_hiddens=n_actions*num_buckets, layers=1, in_act=self.act, last_init=init_xavier)
        self.v = MLP(hiddens, out_hiddens=num_buckets, layers=1, in_act=self.act, last_init=init_xavier)

    
    def forward(self, X, y_action):
        X, z = self.encode(X)
        
        q, action = self.q_head(X)
        z_pred = self.get_transition(z, y_action)

        return q, action, X[:,1:].clone().detach(), z_pred
    

    def env_step(self, X):
        with torch.no_grad():
            X, _ = self.encode(X)
            _, action = self.q_head(X)
            
            return action.detach()
    

    def encode(self, X):
        batch, seq = X.shape[:2]
        self.batch = batch
        self.seq = seq
        X = self.encoder_cnn(X.contiguous().view(self.batch*self.seq, *(X.shape[2:]))).contiguous()
        X = renormalize(X).contiguous().view(self.batch, self.seq, *X.shape[-3:])
        X = X.contiguous().view(self.batch, self.seq, *X.shape[-3:])
        z = X.clone()
        X = X.flatten(-3,-1)
        X = self.projection(X)
        return X, z

    def get_transition(self, z, action):
        z = z.contiguous().view(-1, *z.shape[-3:])
        
        action = F.one_hot(action.clone(), self.n_actions).view(-1, self.n_actions)
        action = action.view(-1, 5, self.n_actions, 1, 1).expand(-1, 5, self.n_actions, *z.shape[-2:])

        z_pred = torch.cat( (z, action[:,0]), 1)
        z_pred = self.transition(z_pred)
        z_pred = renormalize(z_pred)
        
        z_preds=[z_pred.clone()]
        

        for k in range(4):
            z_pred = torch.cat( (z_pred, action[:,k+1]), 1)
            z_pred = self.transition(z_pred)
            z_pred = renormalize(z_pred)
            
            z_preds.append(z_pred)
        
        
        z_pred = torch.stack(z_preds,1)

        z_pred = self.projection(z_pred.flatten(-3,-1)).view(self.batch,5,-1)
        z_pred = self.prediction(z_pred)
        
        return z_pred

    
    def q_head(self, X):
        q = self.dueling_dqn(X)
        action = (q*self.support).sum(-1).argmax(-1)
        
        return q, action

    def get_max_action(self, X):
        with torch.no_grad():
            X, _ = self.encode(X)
            q = self.dueling_dqn(X)
            
            action = (q*self.support).sum(-1).argmax(-1)
            return action

    def evaluate(self, X, action):
        with torch.no_grad():
            X, _ = self.encode(X)
            
            q = self.dueling_dqn(X)
            
            action = action[:,:,None,None].expand_as(q)[:,:,0][:,:,None]
            q = q.gather(-2,action)
            
            return q

    def dueling_dqn(self, X):
        X = F.relu(X)
        
        a = self.a(X).view(self.batch, -1, self.n_actions, self.num_buckets)
        v = self.v(X).view(self.batch, -1, 1, self.num_buckets)
        
        q = v + a - a.mean(-2,keepdim=True)
        q = F.softmax(q,-1)
        
        return q
    
    def network_ema(self, rand_network, target_network, alpha=0.5):
        for param, param_target in zip(rand_network.parameters(), target_network.parameters()):
            param_target.data = alpha * param_target.data + (1 - alpha) * param.data.clone()

    def hard_reset(self, random_model, alpha=0.5):
        with torch.no_grad():
            
            self.network_ema(random_model.encoder_cnn, self.encoder_cnn, alpha)
            self.network_ema(random_model.transition, self.transition, alpha)

            self.network_ema(random_model.projection, self.projection, 0)
            self.network_ema(random_model.prediction, self.prediction, 0)

            self.network_ema(random_model.a, self.a, 0)
            self.network_ema(random_model.v, self.v, 0)