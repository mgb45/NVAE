import os
import logging

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from abc import abstractmethod

import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from torchvision.utils import save_image

# from nn.network.sdn_blocks import SDN, PAction
# from nn.network.controllers import SDN_P
# from nn.network.blocks import *
# from nn.utils.misc import collect_variables, collect_by_name, np2tc, tc2np, frame_seq_to_ch
# from nn.utils.math import gumbel_softmax, bvecmat, compute_mmd, softmax2d
from schedulers import LinearScheduler, ConstantScheduler
# from nn.network.hyperspherical_vae.distributions import VonMisesFisher, HypersphericalUniform

device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
logger = logging.getLogger("tc")

def frame_seq_to_ch(frames):
    # frames has shape [B,T,C,W,H]
    assert len(frames.shape) == 5
    return tc.cat(tc.unbind(frames, dim=1), dim=1)


class Embedder(nn.Module):

    @abstractmethod
    def forward(self, inputs, train=False):
        # Compute action and collect other outputs
        pass

    @abstractmethod
    def compute_losses(self, input_data, target_data, outputs):
        pass


class VAE(Embedder):
    def __init__(self, img_shape, action_dim, state_dim, 
                 window=1, kl_reg=1.0, divider=1, decoder="ha"):

        super().__init__()

        self.state_dim = state_dim
        self.img_shape = img_shape # [C,W,H]
        self.img_size = np.prod(self.img_shape)
        self.window = window
        if type(kl_reg) == str:
            self.kl_scheduler = LinearScheduler(kl_reg)
        else:
            self.kl_scheduler = ConstantScheduler(kl_reg)

        self._encoder = nn.Sequential(
                            HaConvNet(3*self.window, divider=divider),
                            nn.Linear(1024//divider, 2*state_dim)
                           )
        if decoder == "ha":
            self._decoder = HaDeconvNet(state_dim, 3*self.window, divider=divider)
        elif decoder == "broadcast":
            self._decoder = BroadcastDecoder(state_dim, 3*self.window, divider=divider)

    def prior(self, shape):
        shape = tuple(shape)
        mean = tc.zeros(shape).to(device)
        std = tc.ones(shape).to(device)
        return Normal(mean, std)

    def posterior(self, frame):
        output = self._encoder(frame)
        mean, logstd = tc.chunk(output, 2, 1)
        return Normal(mean, logstd.exp())

    def forward(self, input_data, train=False):
        # frames: [B,T,C,W,H]
        # actions: [B,T,act_dim]
        frames = input_data["img"]
        actions = input_data["act"]

        # Last index is the current frame
        frame_window = frame_seq_to_ch(frames[:,-self.window:])
        posterior_dist = self.posterior(frame_window)
        if train:
            posterior_sample = posterior_dist.rsample()
        else:
            posterior_sample = posterior_dist.mean

        prior = self.prior([frames.shape[0], self.state_dim])

        recons = self.decode(posterior_sample)

        return {"posterior": posterior_dist,
                "posterior_samples": posterior_sample,
                "prior": prior,
                "recons": recons}

    def decode(self, state):
        return self._decoder(state)

    def compute_losses(self, input_data, outputs, epoch):
        frames = input_data["img"][:,-self.window:]
        frames = frame_seq_to_ch(frames)
        rec_loss = 0.5*((frames-outputs["recons"])**2).sum(dim=[-1,-2,-3]).mean()

        _kl = kl_divergence(outputs["posterior"], outputs["prior"])
        prior_kl = _kl.sum(dim=1).mean() if len(_kl.shape) == 2 else _kl.mean()
        kl_reg = self.kl_scheduler.get_value(epoch)
        return {"rec": rec_loss, "prior_kl": prior_kl}

    def process_outputs(self, input_data, outputs, epoch, path):
        pass


class BaseVRNN(nn.Module):
    # Assumes posterior inference doesn't depend on previous state
    # For now also assumes inference is made based on a single frame

    def __init__(self, img_shape, action_dim, state_dim): 
        super().__init__()
        self.img_shape = img_shape # [C,W,H]
        self.img_size = np.prod(self.img_shape)
        self.state_dim = state_dim
        self.action_dim = action_dim

    def prior(self, state, action):
        raise NotImplementedError

    def posterior(self, frame, state, action):
        raise NotImplementedError

    def decode(self, state):
        raise NotImplementedError

    def forward(self, input_data, train=False):
        # frames: [B,T,C,W,H]
        # actions: [B,T,act_dim]
        frames = input_data["img"]
        actions = input_data["act"]
        seq_len = frames.shape[1]

        outputs = {
            "posterior_dists": [],
            "prior_dists": [],
            "posterior_samples": [],
            "recons": []
        }

        prev_state = None
        prev_action = None
        for t in range(seq_len):
            frame = frames[:,t]
            
            posterior_dist = self.posterior(frame, prev_state, prev_action)
            prior_dist = self.prior(prev_state, prev_action, batch_size=frames.shape[0] if t == 0 else None)

            posterior_sample = posterior_dist.rsample()

            recons = self.decode(posterior_sample)

            outputs["posterior_dists"].append(posterior_dist)
            outputs["prior_dists"].append(prior_dist)
            outputs["posterior_samples"].append(posterior_sample)
            outputs["recons"].append(recons)

            prev_state = posterior_sample
            prev_action = actions[:,t]

        return outputs

    def compute_losses(self, input_data, outputs):
        frames = input_data["img"]
        actions = input_data["act"]
        seq_len = frames.shape[1]

        ll_loss = [tc.pow(outputs["recons"][t]-frames[:,t], 2).sum(dim=[-1,-2,-3]) for t in range(1, seq_len)]
        ll_loss = 0.5*sum(ll_loss).mean()

        kl_loss = [kl_divergence(outputs["posterior_dists"][t], outputs["prior_dists"][t]).sum(dim=1) for t in range(1,seq_len)]
        kl_loss = sum(kl_loss).mean()

        return {"ll_loss": ll_loss, "kl_loss": kl_loss}

    def process_outputs(self, input_data, outputs, epoch, path):
        seq_len = input_data["img"].shape[1]
        rec_frames = tc.stack([outputs["recons"][t][0] for t in range(seq_len)])
        save_image(rec_frames, os.path.join(path, "recs.jpg"), nrow=int(np.sqrt(seq_len)))
        save_image(input_data["img"][0], os.path.join(path, "frames.jpg"), nrow=int(np.sqrt(seq_len)))

        colors = tc.cat(tc.unbind(input_data["state"][:,1:], dim=1), dim=0)
        colors = tc2np(colors)
        zs = tc.cat(outputs["posterior_samples"][1:], dim=0)
        zs = tc2np(zs)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(zs[:,0], zs[:,1], zs[:,2], alpha=0.7, c=colors[:,0])
        plt.savefig(os.path.join(path, "scatter_pos.jpg"))
        plt.cla(); plt.clf(); plt.close()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(zs[:,0], zs[:,1], zs[:,2], alpha=0.7, c=colors[:,1])
        plt.savefig(os.path.join(path, "scatter_vel.jpg"))
        plt.cla(); plt.clf(); plt.close()
    

class LDSVRNN(BaseVRNN):
    def __init__(self, img_shape, action_dim, state_dim, n_modes): 
        super().__init__(img_shape, action_dim, state_dim)
        self.n_modes = n_modes
        self.conv = self.img_shape[1] <= 64

        self.small_img_shape = [img_shape[0], img_shape[1]//2, img_shape[2]//2]
        self.small_img_size = np.prod(self.small_img_shape)

        if self.conv:
            self._image_encoder = HaConvNet(3)
            self._full_encoder = MLP(1024+state_dim+action_dim, 2*state_dim, 2, 100, nn.ReLU())
            self._decoder = HaDeconvNet(state_dim, 3)
        else:
            self._reshaper = nn.Sequential(
                                #nn.Upsample(self.small_img_shape[1:]),
                                Reshape([self.img_size]),
                               )
            self._encoder = MLP(self.img_size, 2*state_dim, 3, 400, nn.ReLU())

            self._decoder = nn.Sequential(
                                MLP(state_dim, self.img_size, 3, 400, nn.ReLU()),
                                Reshape(self.img_shape),
                                #nn.Upsample(self.img_shape[1:]),
                               )

        self._transition_matrices = nn.Linear(n_modes, state_dim*(2*state_dim+action_dim), bias=False)
        self._gating = MLP(state_dim+action_dim, n_modes, 2, 16, nn.ReLU(), nn.Softmax(dim=1))
        self._prior_logstd = nn.Parameter(tc.zeros([state_dim]))

    def prior(self, state, action, batch_size=None):

        if state is None:
            return Normal(tc.zeros([batch_size, self.state_dim]).to(device),
                          tc.ones([batch_size, self.state_dim]).to(device))

        batch_size = state.shape[0]
        gating = self._gating(tc.cat([state, action], dim=1))
        matrices = self._transition_matrices(gating)

        A, B, C = tc.split(matrices, [self.state_dim**2, self.action_dim*self.state_dim, self.state_dim**2], dim=1)
        A = A.reshape([-1, self.state_dim, self.state_dim])
        B = B.reshape([-1, self.action_dim, self.state_dim])
        C = C.reshape([-1, self.state_dim, self.state_dim])

        noise = tc.randn([batch_size, self.state_dim])

        #next_state_mean = (tc.bmm(state.unsqueeze(1), A) + \
        #                   tc.bmm(action.unsqueeze(1), B)).squeeze(1)
        next_state_mean = state*A[:,:,0] + action*B[:,:,0]
        next_state_std = self._prior_logstd[None,:].repeat(batch_size, 1).exp()
        return Normal(next_state_mean, next_state_std)

    def posterior(self, frame, state, action):
        if state is None:
            state = tc.zeros([frame.shape[0], self.state_dim]).to(device)
            action = tc.zeros([frame.shape[0], self.action_dim]).to(device)

        if self.conv:
            frame_enc = self._image_encoder(frame)
            params = self._full_encoder(tc.cat([frame_enc, state, action], dim=1))
            mean, logstd = tc.chunk(params, 2, dim=1)
        else:
            reshaped_frame = self._reshaper(frame)
            params = self._encoder(tc.cat([reshaped_frame, state, action], dim=1))
            mean, logstd = tc.chunk(params, 2, dim=1)

        return Normal(mean, logstd.exp())

    def decode(self, state):
        return self._decoder(state)
    
    def getLDS(self, state, action, batch_size=None):

        if state is None:
            return Normal(tc.zeros([batch_size, self.state_dim]).to(device),
                          tc.ones([batch_size, self.state_dim]).to(device))

        batch_size = state.shape[0]
        gating = self._gating(tc.cat([state, action], dim=1))
        matrices = self._transition_matrices(gating)

        A, B, C = tc.split(matrices, [self.state_dim**2, self.action_dim*self.state_dim, self.state_dim**2], dim=1)
        A = A.reshape([-1, self.state_dim, self.state_dim])
        B = B.reshape([-1, self.action_dim, self.state_dim])
        C = C.reshape([-1, self.state_dim, self.state_dim])
        
        return A,B,C

class BaseWindowE2C(VAE):
    # Assumes posterior inference doesn't depend on previous state
    # For now also assumes inference is made based on a single frame

    def __init__(self, img_shape, action_dim, state_dim, 
                 window=1, kl_reg=1.0, divider=1, decoder="ha"): 
        super().__init__(img_shape, action_dim, state_dim, window, kl_reg, divider, decoder)
        self.action_dim = action_dim

    def prior_transition(self, prev_dist, action):
        raise NotImplementedError

    def forward(self, input_data, train=False):
        # frames: [B,window,C,W,H]
        # actions: [B,window,act_dim]
        frames = input_data["img"]
        actions = input_data["act"]

        vae_outputs = super().forward(input_data, train)

        prev_frames = frame_seq_to_ch(frames[:,-self.window-1:-1])
        prev_posterior_dist = self.posterior(prev_frames)
        trans_prior = self.prior_transition(prev_posterior_dist, actions[:,-2])

        return {"trans_prior": trans_prior, **vae_outputs}

    def compute_losses(self, input_data, outputs, epoch):
        vae_losses = super().compute_losses(input_data, outputs, epoch)
        trans_kl = kl_divergence(outputs["posterior"], outputs["trans_prior"]).sum(dim=1).mean()

        return {"trans_kl": trans_kl, "rec": vae_losses["rec"]}


class ScalarTransE2C(BaseWindowE2C):

    def __init__(self, img_shape, action_dim, state_dim, window, kl_reg=1.0):
        super().__init__(img_shape, action_dim, state_dim, window, kl_reg)
        self.logdt = nn.Parameter(tc.tensor(0.0))
        if action_dim != state_dim:
            self.log_matrix = nn.Parameter(tc.log(tc.ones([action_dim, state_dim])/action_dim))

    def prior_transition(self, prev_state_dist, action):
        if self.action_dim != self.state_dim:
            prior_trans_mean = prev_state_dist.mean + self.logdt.exp()*tc.matmul(action, self.log_matrix.exp())
        else:
            prior_trans_mean = prev_state_dist.mean + self.logdt.exp()*action
        prior_trans_dist = Normal(prior_trans_mean, prev_state_dist.stddev)
        return prior_trans_dist


class MixtureLDSE2C(BaseWindowE2C):
    def __init__(self, img_shape, action_dim, state_dim, n_modes, window, kl_reg=1.0): 
        super().__init__(img_shape, action_dim, state_dim, window, kl_reg)
        self.n_modes = n_modes

        self._transition_matrices = nn.Linear(n_modes, state_dim*(state_dim+action_dim+1), bias=False)
        self._gating = MLP(state_dim+action_dim, n_modes, 2, 16, nn.ReLU(), nn.Softmax(dim=1))
        self._prior_logstd = nn.Parameter(tc.zeros([state_dim]))

    def prior_transition(self, prev_state_dist, action):

        gating = self._gating(tc.cat([prev_state_dist.mean, action], dim=1))
        matrices = self._transition_matrices(gating)

        A, B, C = tc.split(matrices, [self.state_dim**2, self.action_dim*self.state_dim, self.state_dim], dim=1)
        A = A.reshape([-1, self.state_dim, self.state_dim])
        B = B.reshape([-1, self.action_dim, self.state_dim])
        C = C.reshape([-1, self.state_dim])

        next_state_mean = bvecmat(prev_state_dist.mean, A) + bvecmat(action, B) + C
        next_state_std = self._prior_logstd[None,:].repeat(next_state_mean.shape[0], 1).exp()
        return Normal(next_state_mean, next_state_std)


class PhysicsLDSE2C(BaseWindowE2C):
    def __init__(self, img_shape, action_dim, state_dim, 
                 rank="diag", dt=0.1, steps=1, window=1, kl_reg=1.0, divider=1, decoder="ha"): 
        super().__init__(img_shape, action_dim, state_dim, window, kl_reg, divider, decoder)
        self.rank = rank
        self.steps = steps

        self._transition_matrices = MLP(state_dim+action_dim, state_dim*(2*state_dim+action_dim), 2, 16, nn.ReLU())

        self._prior_vel_logstd = nn.Parameter(tc.tensor(0.0))
        self._post_vel_logstd = nn.Parameter(tc.tensor(0.0))
        self._prior_pos_logstd = nn.Parameter(tc.tensor(np.log(0.1)))
        self._logdt = tc.tensor(np.log(dt)).to(device)

    def forward(self, input_data, train=False):
        # frames: [B,window,C,W,H]
        # actions: [B,window,act_dim]
        frames = input_data["img"]
        actions = input_data["act"]
        steps = self.steps

        curr_frame = frames[:,-1]
        curr_pos_post = self.posterior(curr_frame)
        curr_pos_post_sample = curr_pos_post.rsample()

        prev_frame = frames[:,-1-steps]
        prev_pos_post = self.posterior(prev_frame)
        prev_pos_post_sample = prev_pos_post.rsample()

        pprev_frame = frames[:,-2-steps]
        pprev_pos_post = self.posterior(pprev_frame)
        pprev_pos_post_sample = pprev_pos_post.rsample()

        prev_vel = (prev_pos_post_sample - pprev_pos_post_sample)/self._logdt.exp()

        _pos = prev_pos_post_sample
        _vel = prev_vel
        for t in range(steps):
            trans_pos_prior, trans_vel = self.prior_transition(_pos, _vel, actions[:,-1-steps+t])
            _pos = trans_pos_prior.mean
            _vel = trans_vel

        prior = self.prior(curr_pos_post_sample.shape)
        recons = self.decode(prev_pos_post_sample)
        next_recons = self.decode(trans_pos_prior.rsample().to(tc.float32))
        
        st_recon = self.posterior(recons)

        return {"trans_pos_prior": trans_pos_prior,
                "next_pos_posterior": curr_pos_post,
                "pos_posterior": prev_pos_post,
                "prior": prior,
                "recons": recons,
                "next_recons": next_recons,
                "recon_pos": st_recon}

    def prior_transition(self, pos, vel, action):
        
        dt = self._logdt.exp()
        matrices = self._transition_matrices(tc.cat([pos, action], dim=1))

        A, B, C = tc.split(matrices, [self.state_dim**2, self.action_dim*self.state_dim, self.state_dim**2], dim=1)
        A = A.reshape([-1, self.state_dim, self.state_dim])
        B = B.reshape([-1, self.action_dim, self.state_dim])
        C = C.reshape([-1, self.state_dim, self.state_dim])

        if self.rank == "diag":
            next_vel = vel + dt*(-vel*C[:,0].exp()-pos*A[:,0].exp()+action*B[:,0].exp())
        elif self.rank == "full":
            next_vel = vel + dt*(bvecmat(vel,C)+bvecmat(pos,A)+bvecmat(action,B))
        else:
            raise NotImplementedError

        next_pos = pos + dt*next_vel
        return Normal(next_pos, self._prior_pos_logstd.exp()), next_vel

    def compute_losses(self, input_data, outputs, epoch):
        frames = input_data["img"][:,-2]
        next_frames = input_data["img"][:,-1]

        prior_kl = kl_divergence(outputs["pos_posterior"], outputs["prior"]).sum(dim=1).mean()
        trans_pos_kl = kl_divergence(outputs["next_pos_posterior"], outputs["trans_pos_prior"]).sum(dim=1).mean()
        #trans_vel_kl = kl_divergence(outputs["vel_posterior"], outputs["trans_vel_prior"]).sum(dim=1).mean()
        rec_loss = 0.5*((frames-outputs["recons"])**2).sum(dim=[-1,-2,-3]).mean()
        next_rec_loss = 0.5*((next_frames-outputs["next_recons"])**2).sum(dim=[-1,-2,-3]).mean()
        
        consistency_loss = kl_divergence(outputs["recon_pos"],outputs["next_pos_posterior"]).sum(dim=1).mean()

        kl_reg = self.kl_scheduler.get_value(epoch)
        return {"trans_pos_kl": kl_reg*trans_pos_kl, 
                #"trans_vel_kl": trans_vel_kl,
                #"prior_kl": 0.0*prior_kl,
                #"rec": 0.0*rec_loss,
                "next_rec": next_rec_loss,
               "consistency": consistency_loss}

    def process_outputs(self, input_data, outputs, epoch, path):

        pos = outputs["pos_posterior"].mean
        pos = tc2np(pos)
        true_angle = tc2np(input_data["state"][:,-1,:2])
        true_angle = (true_angle-true_angle.min(axis=0, keepdims=True))/(true_angle.max(axis=0, keepdims=True)-true_angle.min(axis=0, keepdims=True))
        colors = np.concatenate([true_angle, np.zeros_like(true_angle)[:,:1]], axis=1)

        plt.scatter(pos[:,0], pos[:,1], c=colors, label="Manifold")
        plt.scatter(pos[:97,0], pos[:97,1], marker="d", c=np.stack([np.arange(97)/97, np.zeros(97), np.ones(97)], axis=1), label="Trajectory")
        plt.legend()
        plt.savefig(os.path.join(path, "pos.jpg"))
        plt.clf(); plt.cla(); plt.close()
        
        #z = tc.tensor([[0.3, 0.1],[0.3,0.4],[0.7,0.1],[0.7,0.3]]).to(device)
        #out = self.decode(z)
        #save_image(out, os.path.join(path, "example_outs.jpg"), nrow=4)


class LDSE2C(BaseWindowE2C):
    def __init__(self, img_shape, action_dim, state_dim, window=1, kl_reg=1.0, divider=1, decoder="ha"): 
        super().__init__(img_shape, action_dim, state_dim, window, kl_reg, divider, decoder)

        self._transition_matrices = MLP(state_dim+action_dim, state_dim*(state_dim+action_dim+1), 2, 16, nn.ReLU())
        self._prior_logstd = nn.Parameter(tc.tensor(np.log(0.1)))

    def forward(self, input_data, train=False):
        # frames: [B,window,C,W,H]
        # actions: [B,window,act_dim]
        frames = input_data["img"]
        actions = input_data["act"]

        curr_frames = frame_seq_to_ch(frames[:,-self.window:])
        curr_post = self.posterior(curr_frames)
        curr_post_sample = curr_post.rsample()

        prev_frames = frame_seq_to_ch(frames[:,-self.window-1:-1])
        prev_post = self.posterior(prev_frames)
        prev_post_sample = prev_post.rsample()

        #pprev_frames = frame_seq_to_ch(frames[:,-self.window-2:-2])
        #pprev_post = self.posterior(pprev_frames)
        #pprev_post_sample = pprev_post.rsample()


        trans_prior = self.prior_transition(prev_post_sample, actions[:,-2])
        #trans_prior = self.prior_transition(pprev_post_sample, actions[:,-3])
        #trans_prior = self.prior_transition(trans_prior.mean, actions[:,-2])

        prior = self.prior(curr_post_sample.shape)
        recons, next_recons = tc.chunk(self.decode(tc.cat([prev_post_sample, trans_prior.rsample()], dim=0)), 2, 0)
        #recons = self.decode(curr_pos_post_sample)
        return {"trans_prior": trans_prior,
                "next_posterior": curr_post,
                "posterior": prev_post,
                "prior": prior,
                "recons": recons,
                "next_recons": next_recons}

    def prior_transition(self, state, action):
        
        matrices = self._transition_matrices(tc.cat([state, action], dim=1))

        A, B, C = tc.split(matrices, [self.state_dim**2, self.action_dim*self.state_dim, self.state_dim], dim=1)
        A = A.reshape([-1, self.state_dim, self.state_dim])
        B = B.reshape([-1, self.action_dim, self.state_dim])
        C = C.reshape([-1, self.state_dim])

        next_state = bvecmat(state,A)+bvecmat(action,B) + C
        return Normal(next_state, self._prior_logstd.exp())

    def compute_losses(self, input_data, outputs, epoch):
        frames = frame_seq_to_ch(input_data["img"][:,-self.window-1:-1])
        next_frames = frame_seq_to_ch(input_data["img"][:,-self.window:])

        prior_kl = kl_divergence(outputs["posterior"], outputs["prior"]).sum(dim=1).mean()
        trans_kl = kl_divergence(outputs["next_posterior"], outputs["trans_prior"]).sum(dim=1).mean()
        rec_loss = 0.5*((frames-outputs["recons"])**2).sum(dim=[-1,-2,-3]).mean()
        next_rec_loss = 0.5*((next_frames-outputs["next_recons"])**2).sum(dim=[-1,-2,-3]).mean()

        kl_reg = self.kl_scheduler.get_value(epoch)
        return {"trans_kl": kl_reg*trans_kl, 
                "prior_kl": kl_reg*prior_kl,
                "rec": rec_loss,
                "next_rec": next_rec_loss}

    def process_outputs(self, input_data, outputs, epoch, path):

        state = outputs["posterior"].mean
        state = tc2np(state)
        true_angle = tc2np(input_data["state"][:,-3,:2])
        true_angle = (true_angle-true_angle.min(axis=0, keepdims=True))/(true_angle.max(axis=0, keepdims=True)-true_angle.min(axis=0, keepdims=True))
        colors = np.concatenate([true_angle, np.zeros_like(true_angle)[:,:1]], axis=1)

        for i, j in [[0,1],[1,2],[2,3],[0,2],[0,3],[1,3]]:
            plt.scatter(state[:,i], state[:,j], c=colors, label="Manifold")
            plt.legend()
            plt.savefig(os.path.join(path, "state%d%d.jpg"%(i,j)))
            plt.clf(); plt.cla(); plt.close()


class AE(Embedder):
    def __init__(self,
                 img_shape,
                 state_dim):

        super().__init__()

        self.state_dim = state_dim
        self.img_shape = img_shape # [C,W,H]
        self.img_size = np.prod(self.img_shape)

        if self.img_shape[-1] >= 64:
            self.small_img_shape = [img_shape[0], img_shape[1]//2, img_shape[2]//2]
            self.small_img_size = np.prod(self.small_img_shape)

            self._encoder = nn.Sequential(
                                nn.Upsample(self.small_img_shape[1:]),
                                Reshape([self.small_img_size]),
                                nn.Linear(self.small_img_size, 400), nn.ReLU(),
                                nn.Linear(400, 400), nn.ReLU(),
                                nn.Linear(400, 400), nn.ReLU(),
                                nn.Linear(400, state_dim)
                               )

            self._decoder = nn.Sequential(
                                nn.Linear(state_dim, 400), nn.ReLU(),
                                nn.Linear(400, 400), nn.ReLU(),
                                nn.Linear(400, 400), nn.ReLU(),
                                nn.Linear(400, self.small_img_size),
                                Reshape(self.small_img_shape),
                                nn.Upsample(self.img_shape[1:]),
                               )

        else:
            self._encoder = nn.Sequential(
                                HaConvNet(3),
                                nn.Linear(1024, state_dim)
                               )

            self._decoder = HaDeconvNet(state_dim, 3)


    def forward(self, input_data, train=False):
        # frames: [B,T,C,W,H]
        frames = input_data["img"]

        # Last index is the current frame
        curr_frame = frames[:,-1]
        recons = self.decode(self.encode(curr_frame))

        return {"recons": recons}

    def encode(self, image):
        return self._encoder(image)

    def decode(self, state):
        return self._decoder(state)

    def compute_losses(self, input_data, outputs, epoch):
        frames = input_data["img"]
        rec_loss = 0.5*((frames[:,-1]-outputs["recons"])**2).sum(dim=[-1,-2,-3]).mean()
        return {"rec": rec_loss}
    
    


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, inp):
        return inp.reshape([inp.shape[0]]+self.shape)

def coord_grid(shape):
    x = tc.linspace(-1, 1, shape[0]).to(device)
    y = tc.linspace(-1, 1, shape[1]).to(device)
    x_grid, y_grid = tc.meshgrid(x, y)
    grid = tc.cat([x_grid[None,...], y_grid[None,...]], dim=0)
    return grid


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 n_hidden_layers,
                 hidden_dim,
                 hidden_act,
                 output_act=None):
        super(MLP, self).__init__()
        if output_act is None:
            output_act = nn.Identity()
        seq = [nn.Linear(input_dim, hidden_dim), hidden_act]+\
            [nn.Linear(hidden_dim, hidden_dim), hidden_act]*(n_hidden_layers-1)+\
            [nn.Linear(hidden_dim, output_dim), output_act]
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)


class ConvNet(nn.Module):
    def __init__(self,
                 input_ch,
                 output_ch,
                 kernel_size,
                 n_hidden_layers,
                 hidden_ch,
                 hidden_act,
                 output_act=None,
                 input_kernel_size=None):
        super(ConvNet, self).__init__()
        if input_kernel_size is None:
            input_kernel_size = kernel_size
        if output_act is None:
            output_act = nn.Identity()
        seq = [nn.Conv2d(input_ch, hidden_ch, input_kernel_size, 1, input_kernel_size//2), hidden_act]+\
            [nn.Conv2d(hidden_ch, hidden_ch, kernel_size, 1, kernel_size//2), hidden_act]*(n_hidden_layers-1)+\
            [nn.Conv2d(hidden_ch, output_ch, kernel_size, 1, kernel_size//2), output_act]
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)


class DeconvDecoder(nn.Module):
    def __init__(self,
                 input_dim,
                 output_ch,
                 kernel_size,
                 n_hidden_layers,
                 hidden_ch,
                 hidden_act,
                 output_act=None,
                 input_kernel_size=None,
                 broadcast=True):
        super(DeconvDecoder, self).__init__()
        if input_kernel_size is None:
            input_kernel_size = kernel_size
        if output_act is None:
            output_act = nn.Identity

        self.broadcast = broadcast
        if self.broadcast:
            self.width = int(2**(n_hidden_layers+3))
            seq = [nn.Conv2d(input_dim+2, hidden_ch, kernel_size, padding=kernel_size//2), hidden_act()]+\
                [nn.Conv2d(hidden_ch, hidden_ch, kernel_size, padding=kernel_size//2), hidden_act()]*(n_hidden_layers-1)+\
                [nn.Conv2d(hidden_ch, output_ch, kernel_size, padding=kernel_size//2), output_act()]
            self.conv_seq = nn.Sequential(*seq)
        else:
            self.fc_seq = nn.Sequential(nn.Linear(input_dim, hidden_ch*4), nn.ReLU())
            seq = [nn.ConvTranspose2d(hidden_ch, hidden_ch, input_kernel_size, 2, padding=1), hidden_act()]+\
                [nn.ConvTranspose2d(hidden_ch, hidden_ch, kernel_size, 2, padding=1), hidden_act()]*n_hidden_layers+\
                [nn.ConvTranspose2d(hidden_ch, output_ch, kernel_size, 2, padding=1), output_act()]
            self.conv_seq = nn.Sequential(*seq)

    def forward(self, inp):
        if self.broadcast:
            x, y = tc.linspace(-1., 1., self.width).to(device), tc.linspace(-1., 1., self.width).to(device)
            x, y = tc.meshgrid(x, y)
            x = x[None,None,:,:]
            y = y[None,None,:,:]
            x = x.repeat(inp.shape[0], 1, 1, 1)
            y = y.repeat(inp.shape[0], 1, 1, 1)
            inp = inp[:,:,None,None]
            inp = inp.repeat(1, 1, self.width, self.width)
            h = tc.cat([inp, x, y], dim=1)
            h = self.conv_seq(h)
        else:
            h = self.fc_seq(inp)
            h = h.reshape([h.shape[0], -1, 2, 2])
            h = self.conv_seq(h)
        return h


class HaConvNet(nn.Module):
    def __init__(self,
                 input_ch,
                 divider=1):
        super(HaConvNet, self).__init__()
        d = divider
        seq = [nn.Conv2d(input_ch+2, 32//d, 4, 2, 0), nn.ReLU(),
               nn.Conv2d(32//d, 64//d, 4, 2, 0), nn.ReLU(),
               nn.Conv2d(64//d, 128//d, 4, 2, 0), nn.ReLU(),
               nn.Conv2d(128//d, 256//d, 4, 2, 0), nn.ReLU(),
               nn.Flatten()]
        self.seq = nn.Sequential(*seq)

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        grid = coord_grid([64,64])
        inputs = tc.cat([inputs, grid[None,...].repeat(batch_size, 1, 1, 1)], dim=1)
        return self.seq(inputs)


class HaDeconvNet(nn.Module):
    def __init__(self,
                 input_dim,
                 output_ch,
                 divider=1):
        super(HaDeconvNet, self).__init__()
        self.d = d = divider
        seq = [nn.ConvTranspose2d(1024//d, 128//d, 5, 2, 0), nn.ReLU(),
               nn.ConvTranspose2d(128//d, 64//d, 5, 2, 0), nn.ReLU(),
               nn.ConvTranspose2d(64//d, 32//d, 6, 2, 0), nn.ReLU(),
               nn.ConvTranspose2d(32//d, output_ch, 6, 2, 0)]
        self.linear = nn.Sequential(nn.Linear(input_dim, 1024//d))
        self.deconv_seq = nn.Sequential(*seq)

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape([x.shape[0], 1024//self.d, 1, 1])
        x = self.deconv_seq(x)
        return x


class JakabConvNet(nn.Module):
    def __init__(self,
                 input_ch,
                 output_ch):
        super().__init__()
        seq = [nn.Conv2d(input_ch, 32, 7, 1, 3), nn.ReLU(),
               nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(),
               nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
               nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(),
               nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
               nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(),
               nn.Conv2d(128, output_ch, 1, 1, 0)]
        self.seq = nn.Sequential(*seq)

    def forward(self, inputs):
        return self.seq(inputs)


class JakabDeconvNet(nn.Module):
    def __init__(self,
                 input_ch,
                 output_ch):
        super().__init__()
        seq = [nn.Conv2d(input_ch, 128, 3, 1, 1), nn.ReLU(),
               nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(),
               nn.Upsample([32,32], mode="bilinear"),
               nn.Conv2d(128, 64, 3, 1, 1), nn.ReLU(),
               nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(),
               nn.Upsample([64,64], mode="bilinear"),
               nn.Conv2d(64, 32, 3, 1, 1), nn.ReLU(),
               nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(),
               nn.Conv2d(32, output_ch, 3, 1, 1)]
        self.seq = nn.Sequential(*seq)

    def forward(self, inputs):
        return self.seq(inputs)


class BroadcastEncoder(nn.Module):
    def __init__(self,
                 input_ch,
                 divider=1):
        super(BroadcastEncoder, self).__init__()
        d = divider
        seq = [nn.Conv2d(input_ch+2, 64//d, 4, 2, 0), nn.ReLU(),
               nn.Conv2d(64//d, 64//d, 4, 2, 0), nn.ReLU(),
               nn.Conv2d(64//d, 64//d, 4, 2, 0), nn.ReLU(),
               nn.Conv2d(64//d, 64//d, 4, 2, 0), nn.ReLU(),
               nn.Flatten()]
        self.seq = nn.Sequential(*seq)

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        grid = coord_grid([64,64])
        inputs = tc.cat([inputs, grid[None,...].repeat(batch_size, 1, 1, 1)], dim=1)
        return self.seq(inputs)


class BroadcastDecoder(nn.Module):
    def __init__(self,
                 input_dim,
                 output_ch,
                 divider=1):
        super(BroadcastDecoder, self).__init__()
        self.d = d = divider
        seq = [nn.Conv2d(input_dim+2, 64//d, 5, 1, 2), nn.ReLU(),
               nn.Conv2d(64//d, 64//d, 5, 1, 2), nn.ReLU(),
               nn.Conv2d(64//d, 64//d, 5, 1, 2), nn.ReLU(),
               nn.Conv2d(64//d, output_ch, 5, 1, 2)]
        self.deconv_seq = nn.Sequential(*seq)

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        inputs = inputs[:,:,None,None]
        inputs = inputs.repeat(1,1,64,64)

        grid = coord_grid([64,64])
        inputs = tc.cat([inputs, grid[None,...].repeat(batch_size, 1, 1, 1)], dim=1)
        return self.deconv_seq(inputs)


class STN(nn.Module):
    def __init__(self, trg_size, ratio):
        super(STN, self).__init__()
        self.trg_size = trg_size
        self.ratio = ratio
        self.src_size = [s//ratio for s in trg_size] 

    def forward(self, x, pos, angle, scale):
        return self.stn(x, pos, angle, scale)

    def stn(self, x, pos, angle, scale):
        if angle.dim() == 2:
            angle = angle[:,0]
        if scale.dim() == 2:
            scale = scale[:,0]

        cos = tc.cos(angle)
        sin = tc.sin(angle)

        theta0 = cos/scale
        theta1 = -sin/scale
        theta2 = (cos*(self.trg_size[0]/2-pos[:,0])-\
            sin*(self.trg_size[0]/2-pos[:,1]))/self.src_size[0]/scale
        theta3 = sin/scale
        theta4 = cos/scale
        theta5 = (sin*(self.trg_size[0]/2-pos[:,0])+\
            cos*(self.trg_size[0]/2-pos[:,1]))/self.src_size[0]/scale
        theta = tc.stack([theta0, theta1, theta2, theta3, theta4, theta5], dim=1)

        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.shape[:2]+tuple(self.trg_size))
        x = F.grid_sample(x, grid)
        return x


class ReadSTN(nn.Module):
    # Angle not supported yet! Might not be correct

    def __init__(self, trg_size, ratio):
        super(ReadSTN, self).__init__()
        self.trg_size = trg_size
        self.ratio = ratio
        self.src_size = [s*ratio for s in trg_size] 

    def forward(self, x, pos, angle, scale):
        return self.stn(x, pos, angle, scale)

    def stn(self, x, pos, angle, scale):
        if angle.dim() == 2:
            angle = angle[:,0]
        if scale.dim() == 2:
            scale = scale[:,0]

        cos = tc.cos(angle)
        sin = tc.sin(angle)

        theta0 = cos*scale
        theta1 = -sin*scale
        theta2 = -(self.src_size[0]/2-pos[:,0])/self.trg_size[0]
        theta3 = sin*scale
        theta4 = cos*scale
        theta5 = -(self.src_size[0]/2-pos[:,1])/self.trg_size[0]
        theta = tc.stack([theta0, theta1, theta2, theta3, theta4, theta5], dim=1)

        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.shape[:2]+tuple(self.trg_size))
        x = F.grid_sample(x, grid)
        return 