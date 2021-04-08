# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPG_Agent
import torch
from utilities import soft_update, transpose_to_tensor, transpose_list
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from constants import *  

class MADDPG:
    def __init__(self, state_size, action_size, random_seed=32, num_agent = 2):
        super(MADDPG, self).__init__()

        # critic input = obs_full + actions = 14+2+2+2=20
        self.maddpg_agent = [DDPG_Agent(state_size, action_size, random_seed, actor_hidden=[128, 64], critic_hidden=[128, 64], id=i) for i in range(num_agent)]

        
        #state_size, action_size, random_seed, actor_hidden= [400, 300], critic_hidden = [400, 300]
        #def __init__(self, in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic, hidden_in_critic, hidden_out_critic, lr_actor=1.0e-2, lr_critic=1.0e-2)
        
        self.discount_factor = GAMMA
        self.tau = TAU
        self.iter = 0

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor_local for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.actor_target for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(obs, noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions

    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [ddpg_agent.target_act(obs, noise) for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return target_actions

    def update(self, samples, agent_number, logger):
        """update the critics and actors of all the agents """

        # ---                   TRANSFORM INPUT                  --- #
        """
            need to transpose each element of the samples
            to flip obs[parallel_agent][agent_number] to
            obs[agent_number][parallel_agent]
        """
        obs, obs_full, action, reward, next_obs, next_obs_full, done = map(transpose_to_tensor, samples)

        obs_full = torch.stack(obs_full)
        next_obs_full = torch.stack(next_obs_full)



        # ---                   GET AGENT                  --- #
        agent = self.maddpg_agent[agent_number]
        agent.critic_opt.zero_grad()



        # ---                   UPDATE CRITIC (TD)                   --- #
        """
            critic loss = batch mean of (y- Q(s,a) from target network)^2 
            y = reward of this timestep + discount * Q(st+1,at+1) from target network
        """

        target_actions = self.target_act(next_obs)
        target_actions = torch.cat(target_actions, dim=1)
        
        target_critic_input = torch.cat((next_obs_full.t(),target_actions), dim=1).to(DEVICE)
        
        with torch.no_grad():
            q_next = agent.critic_target(target_critic_input)
        
        y = reward[agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - done[agent_number].view(-1, 1))
        action = torch.cat(action, dim=1)
        critic_input = torch.cat((obs_full.t(), action), dim=1).to(DEVICE)
        q = agent.critic_local(critic_input)

        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y.detach())
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_opt.step()


        # ---                   UPDATE ACTOR (PPO)                   --- #
        """
            make input to agent
            detach the other agents to save computation
            saves some time for computing derivative
        """
        agent.actor_opt.zero_grad()

        q_input = [ self.maddpg_agent[i].actor_local(ob) if i == agent_number \
                   else self.maddpg_agent[i].actor_local(ob).detach()
                   for i, ob in enumerate(obs) ]
                
        q_input = torch.cat(q_input, dim=1)
        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        q_input2 = torch.cat((obs_full.t(), q_input), dim=1)
        
        # get the policy gradient
        actor_loss = -agent.critic_local(q_input2).mean()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_opt.step()

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        logger.add_scalars('agent%i/losses' % agent_number,
                           {'critic loss': cl,
                            'actor_loss': al},
                           self.iter)

    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.actor_target, ddpg_agent.actor_local, self.tau)
            soft_update(ddpg_agent.critic_target, ddpg_agent.critic_local, self.tau)
            
            
            




