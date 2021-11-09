from functools import partial
from math import inf

import numpy as np
import torch
import time

from ddpg.ddpg import DDPG
from ddpg.utils import CosineAnnealingNormalNoise, ReplayBuffer
from utils.run_utils import evaluate_policy, fill_buffer_randomly, scale


def learn(env_fn,
          epochs,
          buffer_size,
          eval_freq,
          eta_init,
          eta_min,
          reward_scale=5000,
          **algo_kwargs):

    # Build env and get related info
    env = env_fn()
    obs_shp = env.observation_space.shape
    act_shp = env.action_space.shape
    act_dtype = env.action_space.dtype
    assert np.all(env.action_space.high == 1) & np.all(
        env.action_space.low == -1)

    # scalers
    obs_scale = partial(scale, x_rng=env.observation_space.high)

    def rwd_scale(rew):
        return rew / reward_scale

    # explorative noise
    noise = CosineAnnealingNormalNoise(mu=np.zeros(act_shp),
                                       sigma=eta_init,
                                       sigma_min=eta_min,
                                       T_max=epochs)

    # construct model and buffer
    model = DDPG(obs_shp[0], act_shp[0], epochs=epochs, **algo_kwargs)

    buffer = ReplayBuffer(obs_shp[0],
                          act_shp[0],
                          maxlen=buffer_size,
                          act_dtype=act_dtype)

    fill_buffer_randomly(env_fn, buffer, obs_scale)
    assert buffer.is_full()

    # init recorder
    q_opt = np.inf
    pwc_opt = np.inf
    opc_opt = np.inf
    start = time.time()
    pop_opt = []
    pop_opt1 = []
    state_opt =[]
    try:
        # main loop
        for epoch in range(epochs):
            ret_ep, act_ep = 0., []
            act_ep1 = []
            pwc_ep = 0
            opc_ep = 0
            state_ep = []
            obs, done = env.reset()
            obs = obs_scale(obs)

            for step in range(env.max_svs):
                act = model.act(torch.as_tensor(obs, dtype=torch.float32))
                act = np.clip(act + noise(), -1, 1)

                # env steps
                act1, obs2, pwc, opc, done = env.step(act)
                rew = pwc + opc
                state_ep.append(obs2)
                obs2 = obs_scale(obs2)

                # push experience into the buffer
                buffer.store(obs, act, rew, done, obs2)

                # record
                act_ep.append(act)
                act_ep1.append(act1)
                
                ret_ep += rew
                pwc_ep += pwc
                opc_ep += opc
                

                obs = obs2

                # update the model
                model.update(buffer, rwd_scale)
                if done:
                    # update the global optima
                    if q_opt >= ret_ep:
                        q_opt = ret_ep
                        pwc_opt =pwc_ep
                        opc_opt = opc_ep
                        pop_opt = np.asarray(act_ep)
                        pop_opt1 = np.asarray(act_ep1)
                        state_opt = state_ep

                    if epoch % 500 == 0:
                        print(f'EP: {epoch}|EpR: {ret_ep:.0f}| Q*: {q_opt:.0f}| T: {time.time()-start:.0f}|N:{noise.sigma:.3f}')

                    noise.step()
                    model.lr_step()
                    break
    except KeyboardInterrupt:
        pass
    finally:
        print("done")
        print("evaluation")
        print(f' Q*: {q_opt:.0f}')
        print(f' pwc*: {pwc_opt:.0f}')
        print(f' opc*: {opc_opt:.0f}')

        print("action")
        print(len(pop_opt1))
        print(len(pop_opt1[0]))
        for i in range(len(pop_opt1)):
            print(pop_opt1[i])
        
        print("state_opt")
        print(len(state_opt[0]))
        for i in range(len(state_opt)):
            print(state_opt[i])
