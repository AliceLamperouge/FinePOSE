import math
import random
from typing import List
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn


# from common.mixste import *
from common.mixste_finepose import *

__all__ = ["FinePOSE"]

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class FinePOSE(nn.Module):
    """
    Implement FinePOSE
    """

    def __init__(self, args, joints_left, joints_right, is_train=True, num_proposals=1, sampling_timesteps=1):
        super().__init__()

        self.frames = args.number_of_frames
        self.num_proposals = num_proposals
        self.flip = args.test_time_augmentation
        self.joints_left = joints_left
        self.joints_right = joints_right
        self.is_train = is_train

        # build diffusion
        timesteps = args.timestep
        sampling_timesteps = sampling_timesteps
        self.objective = 'pred_x0'
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        self.self_condition = False
        self.scale = args.scale
        self.box_renewal = True
        self.use_ensemble = True

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # Build Dynamic Head.
        drop_path_rate=0
        if is_train:
            drop_path_rate=0.1

        # DPoser-X fusion: read occlusion-aware knobs from args (default: off).
        self.occlusion_aware = bool(getattr(args, 'occlusion_aware', False))
        self.occlusion_ratio = float(getattr(args, 'occlusion_ratio', 0.3))
        self.completion_jitter_steps = int(getattr(args, 'completion_jitter_steps', 250))

        self.pose_estimator = MixSTE2(num_frame=self.frames, num_joints=17, in_chans=2, embed_dim_ratio=args.cs, depth=args.dep,
        num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=drop_path_rate, is_train=is_train,
        occlusion_aware=self.occlusion_aware)


    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, x, inputs_2d, t, input_text, pre_text_tensor, joint_mask=None):
        x_t = torch.clamp(x, min=-1.1 * self.scale, max=1.1*self.scale)
        x_t = x_t / self.scale
        pred_pose = self.pose_estimator(inputs_2d, x_t, t, input_text, pre_text_tensor, joint_mask=joint_mask)

        x_start = pred_pose
        x_start = x_start * self.scale
        x_start = torch.clamp(x_start, min=-1.1 * self.scale, max=1.1*self.scale)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def model_predictions_fliping(self, x, inputs_2d, inputs_2d_flip, t, input_text, pre_text_tensor, joint_mask=None):
        x_t = torch.clamp(x, min=-1.1 * self.scale, max=1.1*self.scale)
        x_t = x_t / self.scale
        x_t_flip = x_t.clone()
        x_t_flip[:, :, :, :, 0] *= -1
        x_t_flip[:, :, :, self.joints_left + self.joints_right] = x_t_flip[:, :, :,
                                                                        self.joints_right + self.joints_left]

        # DPoser-X fusion: mirror the observation-mask under left/right swap.
        joint_mask_flip = None
        if joint_mask is not None:
            joint_mask_flip = joint_mask.clone()
            # joint_mask shape: (b, f, n, 1) at train path, or (b, h, f, n, 1) at eval path.
            if joint_mask_flip.dim() == 4:
                joint_mask_flip[:, :, self.joints_left + self.joints_right] = \
                    joint_mask_flip[:, :, self.joints_right + self.joints_left]
            elif joint_mask_flip.dim() == 5:
                joint_mask_flip[:, :, :, self.joints_left + self.joints_right] = \
                    joint_mask_flip[:, :, :, self.joints_right + self.joints_left]

        pred_pose = self.pose_estimator(inputs_2d, x_t, t, input_text, pre_text_tensor, joint_mask=joint_mask)
        pred_pose_flip = self.pose_estimator(inputs_2d_flip, x_t_flip, t, input_text, pre_text_tensor, joint_mask=joint_mask_flip)

        pred_pose_flip[:, :, :, :, 0] *= -1
        pred_pose_flip[:, :, :, self.joints_left + self.joints_right] = pred_pose_flip[:, :, :,
                                                                      self.joints_right + self.joints_left]
        pred_pose = (pred_pose + pred_pose_flip) / 2

        x_start = pred_pose
        x_start = x_start * self.scale
        x_start = torch.clamp(x_start, min=-1.1 * self.scale, max=1.1*self.scale)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        pred_noise = pred_noise.float()

        return ModelPrediction(pred_noise, x_start)

    @torch.no_grad()
    def ddim_sample(self, inputs_2d, inputs_3d, input_text, pre_text_tensor, clip_denoised=True, do_postprocess=True):
        batch = inputs_2d.shape[0]
        shape = (batch, self.num_proposals, self.frames, 17, 3)
        total_timesteps, sampling_timesteps, eta, objective = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        
        # original random noise
        img = torch.randn(shape, device=self.device)

        ensemble_score, ensemble_label, ensemble_coord = [], [], []
        x_start = None
        preds_all=[]
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=self.device, dtype=torch.long)


            preds = self.model_predictions(img, inputs_2d, time_cond, input_text, pre_text_tensor)
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start
            preds_all.append(x_start)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return preds_all

    @torch.no_grad()
    def ddim_sample_flip(self, inputs_2d, inputs_3d, input_text, pre_text_tensor, clip_denoised=True, do_postprocess=True, input_2d_flip=None):
        batch = inputs_2d.shape[0]
        shape = (batch, self.num_proposals, self.frames, 17, 3)
        total_timesteps, sampling_timesteps, eta, objective = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(shape, device='cuda')

        x_start = None
        preds_all = []
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, dtype=torch.long).cuda()


            preds = self.model_predictions_fliping(img, inputs_2d, input_2d_flip, time_cond, input_text, pre_text_tensor)
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start

            preds_all.append(x_start)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return torch.stack(preds_all, dim=1)


    def _sample_joint_mask(self, batch, frames, num_joints, device, dtype=torch.float32):
        """DPoser-X fusion: sample a per-joint observation mask (1=observed, 0=masked).
        The same mask is shared across frames within a clip so the model learns a
        temporally coherent completion. At least one joint is kept observed per sample
        to avoid a fully-unknown input that collapses conditioning.
        """
        # Bernoulli over joints, broadcast across frames.
        p_keep = max(0.0, 1.0 - float(self.occlusion_ratio))
        mask = torch.bernoulli(torch.full((batch, num_joints), p_keep, device=device))
        # Ensure at least one observed joint per sample without GPU<->CPU sync.
        all_zero = mask.sum(dim=-1) == 0  # (batch,) bool
        rand_idx = torch.randint(0, num_joints, (batch,), device=device)
        rows = torch.arange(batch, device=device)
        ensure = all_zero.to(mask.dtype)
        mask[rows, rand_idx] = torch.maximum(mask[rows, rand_idx], ensure)
        mask = mask.to(dtype).view(batch, 1, num_joints, 1).expand(batch, frames, num_joints, 1).contiguous()
        return mask

    @torch.no_grad()
    def ddim_sample_complete(self, inputs_2d, observed_3d, obs_mask, input_text, pre_text_tensor,
                             input_2d_flip=None):
        """DPoser-X fusion: RePaint-style pose completion.
        - observed_3d: (B, F, J, 3) ground-truth 3D for the observed joints.
        - obs_mask:    (B, F, J, 1) 1 where the joint is observed, 0 where to complete.
        At each reverse DDIM step, observed joints are overwritten with
        q_sample(observed_3d, t) so that only unobserved joints are "painted" by
        the diffusion model.  Noise is initialized from a truncated timestep
        (self.completion_jitter_steps) instead of the full T to exploit the
        observed evidence.
        """
        assert self.occlusion_aware, "ddim_sample_complete requires --occlusion-aware."
        batch = inputs_2d.shape[0]
        device = inputs_2d.device
        shape = (batch, self.num_proposals, self.frames, 17, 3)

        total_timesteps = self.num_timesteps
        # Truncate the starting timestep.
        start_step = min(max(1, self.completion_jitter_steps), total_timesteps)
        sampling_timesteps = min(self.sampling_timesteps, start_step)
        eta = self.ddim_sampling_eta

        times = torch.linspace(-1, start_step - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        # Scale observed ground-truth to the diffusion space, add a proposal axis.
        observed_scaled = (observed_3d * self.scale).unsqueeze(1).expand(
            batch, self.num_proposals, self.frames, 17, 3).contiguous()
        mask_exp = obs_mask.unsqueeze(1).expand(
            batch, self.num_proposals, self.frames, 17, 1).contiguous()

        # Initialize: observed -> q_sample of GT at start_step; unobserved -> pure noise.
        img = torch.randn(shape, device=device) * self.scale
        t0 = torch.full((batch,), start_step - 1, device=device, dtype=torch.long)
        obs_noisy_init = self.q_sample(
            x_start=observed_scaled.view(batch, -1),
            t=t0,
            noise=torch.randn(batch, observed_scaled.numel() // batch, device=device),
        ).view_as(observed_scaled)
        img = mask_exp * obs_noisy_init + (1.0 - mask_exp) * img

        x_start = None
        preds_all = []
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

            if input_2d_flip is not None and self.flip:
                preds = self.model_predictions_fliping(
                    img, inputs_2d, input_2d_flip, time_cond, input_text, pre_text_tensor,
                    joint_mask=obs_mask)
            else:
                preds = self.model_predictions(
                    img, inputs_2d, time_cond, input_text, pre_text_tensor,
                    joint_mask=obs_mask)
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start
            preds_all.append(x_start)

            if time_next < 0:
                # Final step: splice in clean observed GT for the observed joints.
                img = mask_exp * observed_scaled + (1.0 - mask_exp) * x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)

            img_unknown = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

            # RePaint: override observed joints with forward-noised GT at t_next.
            t_next = torch.full((batch,), time_next, device=device, dtype=torch.long)
            obs_noisy = self.q_sample(
                x_start=observed_scaled.reshape(batch, -1),
                t=t_next,
                noise=torch.randn(batch, observed_scaled.numel() // batch, device=device),
            ).view_as(observed_scaled)

            img = mask_exp * obs_noisy + (1.0 - mask_exp) * img_unknown

        return preds_all


    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def forward(self, input_2d, input_3d, input_text, pre_text_tensor, input_2d_flip=None,
                obs_mask=None, observed_3d=None):

        # Prepare Proposals.
        if not self.is_train:
            # DPoser-X fusion: pose-completion inference path.
            if self.occlusion_aware and obs_mask is not None and observed_3d is not None:
                return self.ddim_sample_complete(
                    input_2d, observed_3d, obs_mask, input_text, pre_text_tensor,
                    input_2d_flip=input_2d_flip)
            if self.flip:
                results = self.ddim_sample_flip(input_2d, input_3d, input_text, pre_text_tensor, input_2d_flip=input_2d_flip)
            else:
                results = self.ddim_sample(input_2d, input_3d, input_text, pre_text_tensor)
            return results

        if self.is_train:

            x_poses, noises, t = self.prepare_targets(input_3d)
            x_poses = x_poses.float()
            t = t.squeeze(-1)

            # DPoser-X fusion: joint-level masked training.
            joint_mask = None
            if self.occlusion_aware:
                b, f, n, _ = input_3d.shape
                joint_mask = self._sample_joint_mask(b, f, n, input_3d.device, dtype=x_poses.dtype)
                # Observed joints get the clean (scaled) ground-truth in the diffusion
                # input, so conditioning is informative; unobserved joints keep the
                # standard forward-noised values.
                observed_scaled = (input_3d * self.scale)
                x_poses = joint_mask * observed_scaled + (1.0 - joint_mask) * x_poses

            pred_pose = self.pose_estimator(input_2d, x_poses, t, input_text, pre_text_tensor,
                                            joint_mask=joint_mask)

            return pred_pose


    def prepare_diffusion_concat(self, pose_3d):

        t = torch.randint(0, self.num_timesteps, (1,), device='cuda').long()
        noise = torch.randn(self.frames, 17, 3, device='cuda')

        x_start = pose_3d

        x_start = x_start * self.scale

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min= -1.1 * self.scale, max= 1.1*self.scale)
        x = x / self.scale


        return x, noise, t

    def prepare_targets(self, targets):
        diffused_poses = []
        noises = []
        ts = []
        for i in range(0,targets.shape[0]):
            targets_per_sample = targets[i]

            d_poses, d_noise, d_t = self.prepare_diffusion_concat(targets_per_sample)
            diffused_poses.append(d_poses)
            noises.append(d_noise)
            ts.append(d_t)
        

        return torch.stack(diffused_poses), torch.stack(noises), torch.stack(ts)


