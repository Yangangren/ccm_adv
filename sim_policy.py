
import os, shutil
import os.path as osp
import pickle
import json
import numpy as np
import click
import torch

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv, CameraWrapper
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder
from rlkit.torch.sac.agent import PEARLAgent
from configs.default import default_config
from launch_experiment import deep_update_dict
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.samplers.util import rollout
import matplotlib.pyplot as plt


def sim_policy(variant, path_to_exp, num_trajs=1, deterministic=False, save_video=False):
    '''
    simulate a trained policy adapting to a new task
    optionally save videos of the trajectories - requires ffmpeg
    :variant: experiment configuration dict
    :path_to_exp: path to exp folder
    :num_trajs: number of trajectories to simulate per task (default 1)
    :deterministic: if the policy is deterministic (default stochastic)
    :save_video: whether to generate and save a video (default False)
    '''

    # create multi-task environment and sample tasks
    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params2']))
    tasks = env.get_all_task_idx()
    print(tasks)
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    eval_tasks=list(tasks[-variant['n_eval_tasks']:])
    print('testing on {} test tasks, {} trajectories each'.format(len(eval_tasks), num_trajs))

    # instantiate networks
    latent_dim = variant['latent_size']
    context_encoder1 = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
    reward_dim = 1
    net_size = variant['net_size']
    recurrent = variant['algo_params']['recurrent']
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder

    context_encoder = encoder_model(
        hidden_sizes=[400, 400, 400],
        input_size=obs_dim * 2  + action_dim + reward_dim,
        output_size=context_encoder1,
    )
    context_encoder_target = encoder_model(
        hidden_sizes=[400, 400, 400],
        input_size=obs_dim + action_dim + reward_dim + obs_dim,
        output_size=context_encoder1,

    )

    forwardenc = encoder_model(
        hidden_sizes=[200, 200, 200],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=obs_dim + reward_dim,

    )
    backwardenc = encoder_model(
        hidden_sizes=[200, 200, 200],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=obs_dim,

    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )
    agent = PEARLAgent(
        latent_dim,
        context_encoder,
        context_encoder_target,
        forwardenc,
        backwardenc,
        policy,
        **variant['algo_params']
    )
    # deterministic eval
    if deterministic:
        agent = MakeDeterministic(agent)

    # load trained weights (otherwise simulate random policy)
    context_encoder.load_state_dict(torch.load(os.path.join(path_to_exp, 'context_encoder.pth'), map_location='cpu'))
    policy.load_state_dict(torch.load(os.path.join(path_to_exp, 'policy.pth'), map_location='cpu'))

    # loop through tasks collecting rollouts
    all_rets = []
    all_mass = []
    video_frames = []
    zs = dict([(i, []) for i in range(30)])

    for idx in tasks:
        env.reset_task(idx)
        all_mass.append(env._wrapped_env._mass)
        agent.clear_z()
        paths = []
        for n in range(num_trajs):
            zs[idx].append(agent.z_means)
            path = rollout(env, agent, max_path_length=variant['algo_params']['max_path_length'], accum_context=True, save_frames=save_video)
            paths.append(path)
            if save_video:
                video_frames += [t['frame'] for t in path['env_infos']]
            if n >= variant['algo_params']['num_exp_traj_eval']:
                agent.infer_posterior(agent.context)
        all_rets.append([sum(p['rewards']) for p in paths])

    with open('z.pkl', 'wb') as f:
        pickle.dump(zs, f)



    if save_video:
        # save frames to file temporarily
        temp_dir = os.path.join(path_to_exp, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        for i, frm in enumerate(video_frames):
            frm.save(os.path.join(temp_dir, '%06d.jpg' % i))

        video_filename=os.path.join(path_to_exp, 'video.mp4'.format(idx))
        # run ffmpeg to make the video
        os.system('ffmpeg -i {}/%06d.jpg -vcodec mpeg4 {}'.format(temp_dir, video_filename))
        # delete the frames
        shutil.rmtree(temp_dir)

    # compute average returns across tasks
    n = min([len(a) for a in all_rets])
    rets = [a[:n] for a in all_rets]
    rets = np.mean(np.stack(rets), axis=1)
    path_name = path_to_exp.split('/')[1]
    np.save(path_name, rets)
    for i, ret in enumerate(rets):
        print('task {:.3f}, avg return: {}'.format(all_mass[i], ret))
    print('Total avg return: {}'.format(np.mean(rets)))

@click.command()
@click.argument('config', default='configs/cheetah-mass.json')
@click.argument('path', default='output/ccm_adv/cheetah-mass/1')
@click.option('--num_trajs', default=1)
@click.option('--deterministic', is_flag=True, default=False)
@click.option('--video', is_flag=True, default=False)
def main(config, path, num_trajs, deterministic, video):
    variant = default_config
    if config:
        with open(osp.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    sim_policy(variant, path, num_trajs, deterministic, video)

def plot():
    ccm_results = np.load('ccm.npy')
    ccm_adv_results = np.load('ccm_adv.npy')
    all_data = np.concatenate((ccm_results, ccm_adv_results), axis=1)

    masses = np.linspace(0.1, 3.1, np.size(ccm_results), endpoint=False)
    fig = plt.figure(figsize=(10, 3.2))
    plt.subplots_adjust(left=0.10, bottom=0.16, right=0.98, top=0.90,
                    wspace=0, hspace=0)

    ax1 = fig.add_subplot(111)
    ax1.plot(masses, all_data[:, 0])
    ax1.plot(masses, all_data[:, 1])
    ax1.set_xlim([0., 3.])
    from matplotlib import patches
    ylim = ax1.get_ylim()
    ax1.add_patch(patches.Rectangle((0., ylim[0]), 1.0, ylim[1]-ylim[0], facecolor='green', alpha=0.2))
    ax1.add_patch(patches.Rectangle((1.0, ylim[0]), 1.0, ylim[1]-ylim[0], facecolor='orange', alpha=0.2))
    ax1.add_patch(patches.Rectangle((2.0, ylim[0]), 1.0, ylim[1]-ylim[0], facecolor='green', alpha=0.2))
    ax1.set_xlabel('Mass', fontsize=16)
    ax1.set_ylabel('Average Return', fontsize=16)
    ax1.tick_params(labelsize=12)
    plt.legend(['ccm', 'ccm_adv'], loc='best', ncol=1, fontsize=12)
    plt.savefig('robust_test.png')


if __name__ == "__main__":
    main()
    plot()
