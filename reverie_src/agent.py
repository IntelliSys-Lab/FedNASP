import json
import os
import sys
import numpy as np
import random
import math
import time
import copy
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import logging
from env import R2RBatch
import utils
from utils import padding_idx, add_idx, Tokenizer, print_progress
from param import args
from collections import defaultdict
import gc
import model_CA
from gating_policy import GatePolicyTwoTower
logger = logging.getLogger()

class BaseAgent(object):
    ''' Base class for an R2R agent to generate and save trajectories. '''

    def __init__(self, env, results_path):
        self.env = env
        self.results_path = results_path
        # random.seed(1)
        self.results = {}
        self.losses = [] # For learning agents

    def write_results(self):
        output = [{'instr_id': k, 'trajectory': v, 'predObjId': r} for k, (v,r) in self.results.items()]
        with open(self.results_path, 'w') as f:
            json.dump(output, f)

    def get_results(self):
        output = [{'instr_id': k, 'trajectory': v, 'predObjId': r} for k, (v,r) in self.results.items()]
        return output

    def rollout(self, **args):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self, iters=None, **kwargs):
        self.env.reset_epoch(shuffle=(iters is not None))   # If iters is not none, shuffle the env batch
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        looped = False
        self.loss = 0
        with torch.no_grad():
            if iters is not None:
                # For each time, it will run the first 'iters' iterations. (It was shuffled before)
                for i in range(iters):
                    for traj in self.rollout(**kwargs):
                        self.loss = 0
                        self.results[traj['instr_id']] = (traj['path'], traj['predObjId'])
            else:   # Do a full round
                while True:
                    for traj in self.rollout(**kwargs):
                        if traj['instr_id'] in self.results:
                            looped = True
                        else:
                            self.loss = 0
                            self.results[traj['instr_id']] = (traj['path'], traj['predObjId'])
                    if looped:
                        break


class Seq2SeqAgent(BaseAgent):
    ''' An agent based on an LSTM seq2seq model with attention. '''

    # For now, the agent can't pick which forward move to make - just the one in the middle
    env_actions = {
      'left': ([0], [-1], [0]), # left
      'right': ([0], [1], [0]), # right
      'up': ([0], [0], [1]), # up
      'down': ([0], [0], [-1]), # down
      'forward': ([1], [0], [0]), # forward
      '<end>': ([0], [0], [0]), # <end>
      '<start>': ([0], [0], [0]), # <start>
      '<ignore>': ([0], [0], [0])  # <ignore>
    }

    def __init__(self, env, results_path, tok, episode_len=20):
        super(Seq2SeqAgent, self).__init__(env, results_path)
        self.tok = tok
        self.episode_len = episode_len
        self.feature_size = self.env.feature_size

        # VilBERT is the only supported listener backbone.
        self.vln_bert = model_CA.VLNBERT(
            feature_size=self.feature_size + args.angle_feat_size).cuda()
        self.critic = model_CA.Critic().cuda()

        # Optimizers
        self.vln_bert_optimizer = args.optimizer(self.vln_bert.parameters(), lr=args.lr)
        self.critic_optimizer = args.optimizer(self.critic.parameters(), lr=args.lr)
        self.optimizers = (self.vln_bert_optimizer, self.critic_optimizer)

        # Evaluations
        self.losses = []
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignoreid, size_average=False)
        self.criterion_REF = nn.CrossEntropyLoss(ignore_index=args.ignoreid, size_average=False)
        # self.ndtw_criterion = utils.ndtw_initialize()
        self.objProposals, self.obj2viewpoint = utils.loadObjProposals()

        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)

    def _sort_batch(self, obs, sorted_instr=True):
        ''' Extract instructions from a list of observations and sort by descending
            sequence length (to enable PyTorch packing). '''

        seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
        seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1]     # Full length

        seq_tensor = torch.from_numpy(seq_tensor)
        seq_lengths = torch.from_numpy(seq_lengths)

        # Sort sequences by lengths
        if sorted_instr:
            seq_lengths, perm_idx = seq_lengths.sort(0, True)       # True -> descending
            sorted_tensor = seq_tensor[perm_idx]
            perm_idx = list(perm_idx)
        else:
            sorted_tensor = seq_tensor
            perm_idx = None

        mask = (sorted_tensor != padding_idx)    # seq_lengths[0] is the Maximum length

        token_type_ids = torch.zeros_like(mask)

        visual_mask = torch.ones(args.directions).bool()
        visual_mask = visual_mask.unsqueeze(0).repeat(mask.size(0),1)
        visual_mask = torch.cat((mask, visual_mask), -1)

        return sorted_tensor.long().cuda(), \
               mask.bool().cuda(),  token_type_ids.long().cuda(), \
               visual_mask.bool().cuda(), \
               list(seq_lengths), perm_idx

    def _feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        features = np.empty((len(obs), args.directions, self.feature_size + args.angle_feat_size), dtype=np.float32)

        for i, ob in enumerate(obs):
            features[i, :, :] = ob['feature']   # Image feat

        return torch.from_numpy(features).cuda()

    def _candidate_variable(self, obs):
        candidate_leng = [len(ob['candidate']) for ob in obs]
        candidate_feat = np.zeros((len(obs), max(candidate_leng), self.feature_size + args.angle_feat_size), dtype=np.float32)
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            for j, cc in enumerate(ob['candidate']):
                candidate_feat[i, j, :] = cc['feature']

        return torch.from_numpy(candidate_feat).cuda(), candidate_leng

    def _object_variable(self, obs):
        cand_obj_leng = [len(ob['candidate_obj'][2]) + 1 for ob in obs] # +1 is for no REF
        # VilBERT object features append the four-angle representation.
        cand_obj_feat = np.zeros((len(obs), max(cand_obj_leng), self.feature_size + 4), dtype=np.float32)
        cand_obj_pos = np.zeros((len(obs), max(cand_obj_leng), 5), dtype=np.float32)

        for i, ob in enumerate(obs):
            obj_local_pos, obj_features, candidate_objId = ob['candidate_obj']
            for j, cc in enumerate(candidate_objId):
                cand_obj_feat[i, j, :] = obj_features[j]
                cand_obj_pos[i, j, :] = obj_local_pos[j]

        return torch.from_numpy(cand_obj_feat).cuda(), torch.from_numpy(cand_obj_pos).cuda(), cand_obj_leng

    def get_input_feat(self, obs):
        input_a_t = np.zeros((len(obs), args.angle_feat_size), np.float32)
        for i, ob in enumerate(obs):
            input_a_t[i] = utils.angle_feature(ob['heading'], ob['elevation'])
        input_a_t = torch.from_numpy(input_a_t).cuda()

        # f_t = self._feature_variable(obs)      # Image features from obs
        f_t = None
        candidate_feat, candidate_leng = self._candidate_variable(obs)

        obj_feat, obj_pos, obj_leng = self._object_variable(obs)

        return input_a_t, f_t, candidate_feat, candidate_leng, obj_feat, obj_pos, obj_leng

    def _teacher_action(self, obs, ended, cand_size):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = args.ignoreid
            else:
                for k, candidate in enumerate(ob['candidate']):
                    if candidate['viewpointId'] == ob['teacher']:   # Next view point
                        a[i] = k
                        break
                else:   # Stop here
                    assert ob['teacher'] == ob['viewpoint']         # The teacher action should be "STAY HERE"
                    a[i] = cand_size - 1
        return torch.from_numpy(a).cuda()

    def _teacher_REF(self, obs, just_ended):
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if not just_ended[i]:                                            # Just ignore this index
                a[i] = args.ignoreid
            else:
                candidate_objs = ob['candidate_obj'][2]
                for k, kid in enumerate(candidate_objs):
                    if kid == ob['objId']:
                        a[i] = k
                        break
                else:
                    a[i] = args.ignoreid
        return torch.from_numpy(a).cuda()

    def make_equiv_action(self, a_t, perm_obs, perm_idx=None, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        def take_action(i, idx, name):
            if type(name) is int:       # Go to the next view
                self.env.env.sims[idx].makeAction([name], [0], [0])
            else:                       # Adjust
                self.env.env.sims[idx].makeAction(*self.env_actions[name])
            state = self.env.env.sims[idx].getState()[0]
            if traj is not None:
                traj[i]['path'].append((state.location.viewpointId, state.heading, state.elevation))

        if perm_idx is None:
            perm_idx = range(len(perm_obs))

        for i, idx in enumerate(perm_idx):
            action = a_t[i]
            if action != -1:            # -1 is the <stop> action
                select_candidate = perm_obs[i]['candidate'][action]
                src_point = perm_obs[i]['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point ) // 12   # The point idx started from 0
                trg_level = (trg_point ) // 12
                while src_level < trg_level:    # Tune up
                    take_action(i, idx, 'up')
                    src_level += 1
                while src_level > trg_level:    # Tune down
                    take_action(i, idx, 'down')
                    src_level -= 1
                while self.env.env.sims[idx].getState()[0].viewIndex != trg_point:    # Turn right until the target
                    take_action(i, idx, 'right')
                assert select_candidate['viewpointId'] == \
                       self.env.env.sims[idx].getState()[0].navigableLocations[select_candidate['idx']].viewpointId
                take_action(i, idx, select_candidate['idx'])

                state = self.env.env.sims[idx].getState()[0]
                if traj is not None:
                    traj[i]['path'].append((state.location.viewpointId, state.heading, state.elevation))

    def rollout(self, train_ml=None, train_rl=True, reset=True, speaker=None):
        """
        :param train_ml:    The weight to train with maximum likelihood
        :param train_rl:    whether use RL in training
        :param reset:       Reset the environment
        :param speaker:     Speaker used in back translation.
                            If the speaker is not None, use back translation.
                            O.w., normal training
        :return:
        """
        if self.feedback == 'teacher' or self.feedback == 'argmax':
            train_rl = False

        if reset: # Reset env
            obs = np.array(self.env.reset())
        else:
            obs = np.array(self.env._get_obs())

        batch_size = len(obs)

        # Reorder the language input for the encoder (do not ruin the original code)
        sentence, language_attention_mask, token_type_ids, \
            visual_attention_mask, seq_lengths, perm_idx = self._sort_batch(obs)
        perm_obs = obs[perm_idx]

        ''' Language BERT: VilBERT is the only supported backbone. '''
        language_inputs = {'mode':        'language',
                        'sentence':       sentence,
                        'token_type_ids': token_type_ids}
        # (batch_size, seq_len, hidden_size)
        language_inputs['lang_masks'] = language_attention_mask
        h_t, language_features = self.vln_bert(**language_inputs)
        language_attention_mask = language_attention_mask[:, 1:]

        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
            'predObjId': None
        } for ob in perm_obs]

        # Init the reward shaping
        last_dist = np.zeros(batch_size, np.float32)
        # last_ndtw = np.zeros(batch_size, np.float32)
        for i, ob in enumerate(perm_obs):   # The init distance from the view point to the target
            last_dist[i] = ob['distance']
            path_act = [vp[0] for vp in traj[i]['path']]
            # last_ndtw[i] = self.ndtw_criterion[ob['scan']](path_act, ob['gt_path'], metric='ndtw')

        # Initialization the tracking state
        ended = np.array([False] * batch_size)  # Indices match permuation of the model, not env
        just_ended = np.array([False] * batch_size)

        # Init the logs
        rewards = []
        hidden_states = []
        policy_log_probs = []
        masks = []
        stop_mask = torch.tensor([False] * batch_size).cuda().unsqueeze(1)
        entropys = []
        ml_loss = 0.
        ref_loss = 0.

        # For test result submission: no backtracking
        visited = [set() for _ in range(batch_size)]

        for t in range(self.episode_len):

            input_a_t, f_t, candidate_feat, candidate_leng, obj_feat, obj_pos, obj_leng = self.get_input_feat(perm_obs)

            # the first [CLS] token, initialized by the language BERT, servers
            # as the agent's state passing through time steps
            visual_temp_mask = (utils.length2mask(candidate_leng) == 0).bool()
            obj_temp_mask = (utils.length2mask(obj_leng) == 0).bool()
            visual_attention_mask = torch.cat((language_attention_mask, visual_temp_mask, obj_temp_mask), dim=-1)

            self.vln_bert.vln_bert.config.directions = max(candidate_leng)
            self.vln_bert.vln_bert.config.obj_directions = max(obj_leng)
            ''' Visual BERT '''
            visual_inputs = {'mode':              'visual',
                            'sentence':           language_features,
                            'token_type_ids':     token_type_ids,
                            'action_feats':       input_a_t,
                            'pano_feats':         f_t,
                            'cand_feats':         candidate_feat,
                            'obj_feats':          obj_feat,
                            'obj_pos':            obj_pos,
                            'already_dropfeat':   (speaker is not None)}
            visual_inputs.update({
                'h_t': h_t,
                'lang_masks': language_attention_mask,
                'cand_masks': visual_temp_mask,
                'obj_masks': obj_temp_mask,
                'act_t': t,
            })
            h_t, logit, logit_REF = self.vln_bert(**visual_inputs)
            hidden_states.append(h_t)

            # Mask outputs where agent can't move forward
            # Here the logit is [b, max_candidate]
            candidate_mask = utils.length2mask(candidate_leng)
            candidate_mask = torch.cat((candidate_mask, stop_mask), dim=-1)
            # print("logit.shape:", tuple(logit.shape))
            # print("visual_temp_mask.shape:", tuple(visual_temp_mask.shape))  # [B, max_candidate]
            # print('candidate.shape:', tuple(candidate_mask.shape))
            # print("stop_index (expected):", visual_temp_mask.size(1))
            # print("args.ignoreid:", args.ignoreid)
            logit.masked_fill_(candidate_mask, -float('inf'))
                
            candidate_mask_obj = utils.length2mask(obj_leng)
            logit_REF.masked_fill_(candidate_mask_obj, -float('inf'))

            if train_ml is not None:
                # Supervised training
                target = self._teacher_action(perm_obs, ended, candidate_mask.size(1))
                ml_loss += self.criterion(logit, target)

            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t = target                # teacher forcing
            elif self.feedback == 'argmax':
                _, a_t = logit.max(1)        # student forcing - argmax
                a_t = a_t.detach()
                log_probs = F.log_softmax(logit, 1)                              # Calculate the log_prob here
                policy_log_probs.append(log_probs.gather(1, a_t.unsqueeze(1)))   # Gather the log_prob for each batch
            elif self.feedback == 'sample':
                probs = F.softmax(logit, 1)    # sampling an action from model
                c = torch.distributions.Categorical(probs)
                self.logs['entropy'].append(c.entropy().sum().item())      # For log
                entropys.append(c.entropy())                                # For optimization
                a_t = c.sample().detach()
                policy_log_probs.append(c.log_prob(a_t))
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            # Prepare environment action
            # NOTE: Env action is in the perm_obs space
            cpu_a_t = a_t.cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                if ((next_id == visual_temp_mask.size(1)) or (t == self.episode_len-1)) and (not ended[i]):  # just stopped and forced stopped
                    just_ended[i] = True
                    if self.feedback == 'argmax':
                        _, ref_t = logit_REF[i].max(0)
                        if ref_t != obj_leng[i]-1:  # decide not to do REF
                            traj[i]['predObjId'] = perm_obs[i]['candidate_obj'][2][ref_t]

                    # REMOVE: submit-only object fallback belongs to the
                    # excluded test-submission workflow.
                else:
                    just_ended[i] = False

                if (next_id == visual_temp_mask.size(1)) or (next_id == args.ignoreid) or (ended[i]):    # The last action is <end>
                    cpu_a_t[i] = -1             # Change the <end> and ignore action to -1

            ''' Supervised training for REF '''
            if train_ml is not None:
                target_obj = self._teacher_REF(perm_obs, just_ended)
                ref_loss += self.criterion_REF(logit_REF, target_obj)

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, perm_obs, perm_idx, traj)
            obs = np.array(self.env._get_obs())
            perm_obs = obs[perm_idx]                    # Perm the obs for the resu

            if train_rl:
                # Calculate the mask and reward
                dist = np.zeros(batch_size, np.float32)
                # ndtw_score = np.zeros(batch_size, np.float32)
                reward = np.zeros(batch_size, np.float32)
                mask = np.ones(batch_size, np.float32)
                for i, ob in enumerate(perm_obs):
                    dist[i] = ob['distance']
                    # path_act = [vp[0] for vp in traj[i]['path']]
                    # ndtw_score[i] = self.ndtw_criterion[ob['scan']](path_act, ob['gt_path'], metric='ndtw')
                    if ended[i]:
                        reward[i] = 0.0
                        mask[i] = 0.0
                    else:
                        action_idx = cpu_a_t[i]
                        if action_idx == -1:                              # If the action now is end
                            # navigation success if the target object is visible when STOP
                            # end_viewpoint_id = ob['scan'] + '_' + ob['viewpoint']
                            # if self.objProposals.__contains__(end_viewpoint_id):
                            #     if ob['objId'] in self.objProposals[end_viewpoint_id]['objId']:
                            #         reward[i] = 2.0 + ndtw_score[i] * 2.0
                            #     else:
                            #         reward[i] = -2.0
                            # else:
                            #     reward[i] = -2.0
                            if dist[i] < 1.0:                             # Correct
                                reward[i] = 2.0  # + ndtw_score[i] * 2.0
                            else:                                         # Incorrect
                                reward[i] = -2.0
                        else:                                             # The action is not end
                            # Change of distance and nDTW reward
                            reward[i] = - (dist[i] - last_dist[i])
                            # ndtw_reward = ndtw_score[i] - last_ndtw[i]
                            if reward[i] > 0.0:                           # Quantification
                                reward[i] = 1.0  # + ndtw_reward
                            elif reward[i] < 0.0:
                                reward[i] = -1.0  # + ndtw_reward
                            else:
                                raise NameError("The action doesn't change the move")
                            # miss the target penalty
                            if (last_dist[i] <= 1.0) and (dist[i]-last_dist[i] > 0.0):
                                reward[i] -= (1.0 - last_dist[i]) * 2.0
                rewards.append(reward)
                masks.append(mask)
                last_dist[:] = dist
                # last_ndtw[:] = ndtw_score

            # Update the finished actions
            # -1 means ended or ignored (already ended)
            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            # Early exit if all ended
            if ended.all():
                break

        if train_rl:
            # Last action in A2C
            input_a_t, f_t, candidate_feat, candidate_leng, obj_feat, obj_pos, obj_leng = self.get_input_feat(perm_obs)
            visual_temp_mask = (utils.length2mask(candidate_leng) == 0).bool()
            obj_temp_mask = (utils.length2mask(obj_leng) == 0).bool()
            visual_attention_mask = torch.cat((language_attention_mask, visual_temp_mask, obj_temp_mask), dim=-1)

            self.vln_bert.vln_bert.config.directions = max(candidate_leng)
            self.vln_bert.vln_bert.config.obj_directions = max(obj_leng)
            ''' Visual BERT '''
            visual_inputs = {'mode':              'visual',
                            'sentence':           language_features,
                            'token_type_ids':     token_type_ids,
                            'action_feats':       input_a_t,
                            'pano_feats':         f_t,
                            'cand_feats':         candidate_feat,
                            'obj_feats':          obj_feat,
                            'obj_pos':            obj_pos,
                            'already_dropfeat':   (speaker is not None)}
            visual_inputs.update({
                'h_t': h_t,
                'lang_masks': language_attention_mask,
                'cand_masks': visual_temp_mask,
                'obj_masks': obj_temp_mask,
                'act_t': len(hidden_states),
            })
            last_h_, _, _ = self.vln_bert(**visual_inputs)

            rl_loss = 0.

            # NOW, A2C!!!
            # Calculate the final discounted reward
            last_value__ = self.critic(last_h_).detach()    # The value esti of the last state, remove the grad for safety
            discount_reward = np.zeros(batch_size, np.float32)  # The inital reward is zero
            for i in range(batch_size):
                if not ended[i]:        # If the action is not ended, use the value function as the last reward
                    discount_reward[i] = last_value__[i]

            length = len(rewards)
            total = 0
            for t in range(length-1, -1, -1):
                discount_reward = discount_reward * args.gamma + rewards[t]   # If it ended, the reward will be 0
                mask_ = torch.from_numpy(masks[t]).cuda()
                clip_reward = discount_reward.copy()
                r_ = torch.from_numpy(clip_reward).cuda()
                v_ = self.critic(hidden_states[t])
                a_ = (r_ - v_).detach()

                # r_: The higher, the better. -ln(p(action)) * (discount_reward - value)
                rl_loss += (-policy_log_probs[t] * a_ * mask_).sum()
                rl_loss += (((r_ - v_) ** 2) * mask_).sum() * 0.5     # 1/2 L2 loss
                if self.feedback == 'sample':
                    rl_loss += (- 0.01 * entropys[t] * mask_).sum()
                self.logs['critic_loss'].append((((r_ - v_) ** 2) * mask_).sum().item())

                total = total + np.sum(masks[t])
            self.logs['total'].append(total)

            # Normalize the loss function
            if args.normalize_loss == 'total':
                rl_loss /= total
            elif args.normalize_loss == 'batch':
                rl_loss /= batch_size
            else:
                assert args.normalize_loss == 'none'

            self.loss += rl_loss
            self.logs['RL_loss'].append(rl_loss.item())

        if train_ml is not None:
            self.loss += ml_loss * train_ml / batch_size
            self.logs['IL_loss'].append((ml_loss * train_ml / batch_size).item())
            self.loss += ref_loss * args.ref_loss_weight / batch_size
            self.logs['REF_loss'].append(ref_loss.item() * args.ref_loss_weight / batch_size)

        if type(self.loss) is int:  # For safety, it will be activated if no losses are added
            self.losses.append(0.)
        else:
            self.losses.append(self.loss.item() / self.episode_len)    # This argument is useless.

        # import pdb; pdb.set_trace()

        return traj

    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False, iters=None):
        ''' Evaluate once on each instruction in the current environment '''
        self.feedback = feedback
        if use_dropout:
            self.vln_bert.train()
            self.critic.train()
        else:
            self.vln_bert.eval()
            self.critic.eval()
        super(Seq2SeqAgent, self).test(iters)

    def zero_grad(self):
        self.loss = 0.
        self.losses = []
        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            optimizer.zero_grad(set_to_none=True)

    def accumulate_gradient(self, feedback='teacher', **kwargs):
        if feedback == 'teacher':
            self.feedback = 'teacher'
            self.rollout(train_ml=args.teacher_weight, train_rl=False, **kwargs)
        elif feedback == 'sample':
            self.feedback = 'teacher'
            self.rollout(train_ml=args.ml_weight, train_rl=False, **kwargs)
            self.feedback = 'sample'
            self.rollout(train_ml=None, train_rl=True, **kwargs)
        else:
            assert False

    def optim_step(self):
        self.loss.backward()

        torch.nn.utils.clip_grad_norm_(self.vln_bert.parameters(), 40.)

        self.vln_bert_optimizer.step()
        self.critic_optimizer.step()

    def train(self, n_iters, feedback='teacher', **kwargs):
        ''' Train for a given number of iterations '''
        self.feedback = feedback

        self.vln_bert.train()
        self.critic.train()

        self.losses = []
        for iter in range(1, n_iters + 1):

            self.vln_bert_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)

            self.loss = 0

            if feedback == 'teacher':
                self.feedback = 'teacher'
                self.rollout(train_ml=args.teacher_weight, train_rl=False, **kwargs)
            elif feedback == 'sample': # agents in IL and RL separately
                if args.ml_weight != 0:
                    self.feedback = 'teacher'
                    self.rollout(train_ml=args.ml_weight, train_rl=False, **kwargs)
                self.feedback = 'sample'
                self.rollout(train_ml=None, train_rl=True, **kwargs)
            else:
                assert False
            # l = copy.deepcopy(self.loss)
            self.loss.backward()
            # logger.info(f'iter: {iter} : Loss: {str(self.loss)}')

            torch.nn.utils.clip_grad_norm_(self.vln_bert.parameters(), 40.)

            self.vln_bert_optimizer.step()
            self.critic_optimizer.step()

            print_progress(iter, n_iters, prefix='Progress:', suffix='Complete', bar_length=50)

        # Release grad buffers between training phases/clients.
        self.vln_bert_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)

    

    def save(self, epoch, path):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}
        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
        all_tuple = [("vln_bert", self.vln_bert, self.vln_bert_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        states = torch.load(path)

        def recover_state(name, model, optimizer):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]['state_dict'].keys())
            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
            state.update(states[name]['state_dict'])
            model.load_state_dict(state)
           
        all_tuple = [("vln_bert", self.vln_bert, self.vln_bert_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]
        for param in all_tuple:
            recover_state(*param)
        return states['vln_bert']['epoch'] - 1

    def load_pretrain(self, path):
        ''' Loads parameters from pretrained network '''
        load_states = torch.load(path)
        # print(self.vln_bert.state_dict()['candidate_att_layer.linear_in.weight'])
        # print(self.vln_bert.state_dict()['visual_bert.bert.encoder.layer.9.intermediate.dense.weight'])

        def recover_state(name, model):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(load_states[name]['state_dict'].keys())
            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS FOUND IN MODEL")
                for ikey in model_keys:
                    if ikey not in load_keys:
                        print('key not in model: ', ikey)
                for ikey in load_keys:
                    if ikey not in model_keys:
                        print('key not in loaded states: ', ikey)

            state.update(load_states[name]['state_dict'])
            model.load_state_dict(state)

        all_tuple = [("vln_bert", self.vln_bert)]
        for param in all_tuple:
            recover_state(*param)

        return load_states['vln_bert']['epoch'] - 1


# ==============================================================
# PrefixSeq2SeqAgent – step-wise conditional prefix personalization
# ==============================================================

class PrefixSeq2SeqAgent(BaseAgent):
    """Agent with step-wise conditional prefix personalization.

    Uses ``PrefixVLNBERT`` from ``model_prefix`` as the backbone, with
    per-block local prefixes and an external two-tower gate policy.

    The ``rollout`` uses a **1-pass pre-forward gate**:
      g_t is computed before the backbone visual forward from
      [h_{t-1}, candidate_count_norm, step_norm], then applied once.
    """

    # share the class-level env_actions from Seq2SeqAgent
    env_actions = Seq2SeqAgent.env_actions

    def __init__(self, env, results_path, tok, episode_len=20,
                 prefix_len=8, prefix_modules='infer',
                 gate_hidden=256, freeze_backbone=True):
        super().__init__(env, results_path)
        if args.vlnbert != 'vilbert':
            raise ValueError('PrefixSeq2SeqAgent requires --vlnbert vilbert.')
        self.tok = tok
        self.episode_len = episode_len
        self.feature_size = env.feature_size
        self.freeze_backbone = freeze_backbone

        # ---- model ----
        # REMOVE: model_prefix_prevalent belongs to the excluded Prevalent path.
        import model_prefix as mpfx
        # Disable language-prefix personalization in this variant.
        effective_prefix_modules = prefix_modules
        if isinstance(prefix_modules, str) and prefix_modules.strip().lower() not in ('', 'infer', 'auto'):
            modules = [m.strip() for m in prefix_modules.split(',') if m.strip() and m.strip() != 'lang_last']
            effective_prefix_modules = ','.join(modules) if modules else 'infer'
        old_enable_lang_prefix = bool(getattr(args, 'enable_lang_prefix', True))
        args.enable_lang_prefix = False
        self.vln_bert = mpfx.PrefixVLNBERT(
            feature_size=self.feature_size + args.angle_feat_size,
            prefix_len=prefix_len,
            prefix_modules=effective_prefix_modules,
            gate_hidden=gate_hidden).cuda()
        args.enable_lang_prefix = old_enable_lang_prefix
        if hasattr(self.vln_bert, 'enable_lang_prefix'):
            self.vln_bert.enable_lang_prefix = False
        if hasattr(self.vln_bert.vln_bert, 'enable_lang_prefix'):
            self.vln_bert.vln_bert.enable_lang_prefix = False
        self.critic = mpfx.Critic().cuda()
        gate_hidden_size = int(getattr(
            self.vln_bert.vln_bert.config,
            'bi_hidden_size',
            self.vln_bert.vln_bert.config.hidden_size,
        ))
        gate_num_blocks = int(self.vln_bert.num_prefix_modules)
        self.gate_policy = GatePolicyTwoTower(
            hidden_size=gate_hidden_size,
            num_blocks=gate_num_blocks,
            stats_dim=2,
            obs_dim=int(self.feature_size),
        ).cuda()
        self.models = (self.vln_bert, self.critic, self.gate_policy)

        # ---- freeze backbone if requested ----
        if freeze_backbone:
            logger.info("Freezing backbone VLNBERT parameters, only training local prefixes and gate policy.")
            for p in self.vln_bert.vln_bert.parameters():
                p.requires_grad = False
            for p in self.vln_bert.action_state_project.parameters():
                p.requires_grad = False
            for p in self.vln_bert.action_LayerNorm.parameters():
                p.requires_grad = False
            # keep local prefix parameters trainable
            for name, p in self.vln_bert.named_parameters():
                if name.startswith('prefix_layers.') or ('attn_prefix_' in name):
                    p.requires_grad = True

        # ---- optimizers ----
        pfx_lr = getattr(args, 'prefix_lr', 1e-4)
        vln_trainable = [p for p in self.vln_bert.parameters() if p.requires_grad]
        self.vln_bert_optimizer = args.optimizer(vln_trainable, lr=pfx_lr)
        self.critic_optimizer = args.optimizer(
            self.critic.parameters(), lr=args.lr)
        gate_lr = getattr(args, 'gate_lr', None)
        if gate_lr is None:
            gate_lr = args.lr
        self.gate_optimizer = torch.optim.Adam(self.gate_policy.parameters(), lr=gate_lr)
        self.optimizers = (self.vln_bert_optimizer, self.critic_optimizer, self.gate_optimizer)

        # ---- loss ----
        self.losses = []
        self.criterion = torch.nn.CrossEntropyLoss(
            ignore_index=args.ignoreid, size_average=False)
        self.criterion_REF = torch.nn.CrossEntropyLoss(
            ignore_index=args.ignoreid, size_average=False)
        self.objProposals, self.obj2viewpoint = utils.loadObjProposals()

        # Logs
        self.logs = defaultdict(list)
        sys.stdout.flush()

   

    # ----- re-use helpers from Seq2SeqAgent -----
    _sort_batch        = Seq2SeqAgent._sort_batch
    _feature_variable  = Seq2SeqAgent._feature_variable
    _candidate_variable = Seq2SeqAgent._candidate_variable
    _object_variable   = Seq2SeqAgent._object_variable
    get_input_feat     = Seq2SeqAgent.get_input_feat
    _teacher_action    = Seq2SeqAgent._teacher_action
    _teacher_REF       = Seq2SeqAgent._teacher_REF
    make_equiv_action  = Seq2SeqAgent.make_equiv_action

    def test(self, use_dropout=False, feedback='argmax',
             allow_cheat=False, iters=None):
        self.feedback = feedback
        if use_dropout:
            self.vln_bert.train()
            self.critic.train()
            self.gate_policy.train()
        else:
            self.vln_bert.eval()
            self.critic.eval()
            self.gate_policy.eval()
        super().test(iters)

    def save(self, epoch, path):
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}

        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

        all_tuple = [
            ("vln_bert", self.vln_bert, self.vln_bert_optimizer),
            ("critic", self.critic, self.critic_optimizer),
            ("gate_policy", self.gate_policy, self.gate_optimizer),
        ]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        states = torch.load(path)

        def recover_state(name, model, optimizer):
            if name not in states:
                return
            state = model.state_dict()
            state.update(states[name]['state_dict'])
            model.load_state_dict(state)
            # REMOVE: --loadOptim belonged to the removed standalone local
            # workflow. Ours persists its optimizer state per client instead.

        all_tuple = [
            ("vln_bert", self.vln_bert, self.vln_bert_optimizer),
            ("critic", self.critic, self.critic_optimizer),
            ("gate_policy", self.gate_policy, self.gate_optimizer),
        ]
        for param in all_tuple:
            recover_state(*param)
        return states['vln_bert']['epoch'] - 1

    # ------------------------------------------------------------------ #
    #  load_backbone_from_ckpt – load global weights into prefix model
    # ------------------------------------------------------------------ #
    def load_backbone_from_ckpt(self, ckpt_path):
        """Load backbone + critic weights from a standard VLNBERT ckpt.

        New parameters (prefix_layers, gate_net) are left at their
        random initialisation.
        """
        states = torch.load(ckpt_path, map_location='cpu')

        if isinstance(states, dict) and 'vln_bert' in states:
            vln_sd = states['vln_bert'].get('state_dict', states['vln_bert'])
            cri_sd = states.get('critic', {}).get('state_dict', states.get('critic', {}))
        elif isinstance(states, dict) and 'state_dict' in states:
            vln_sd = states['state_dict']
            cri_sd = {}
        elif isinstance(states, dict):
            vln_sd = states
            cri_sd = {}
        else:
            raise ValueError(f"Unsupported checkpoint type: {type(states)}")

        model_sd = self.vln_bert.state_dict()
        matched, new_keys = {}, []
        for k in model_sd:
            if k in vln_sd:
                matched[k] = vln_sd[k]
            else:
                new_keys.append(k)

        model_sd.update(matched)
        self.vln_bert.load_state_dict(model_sd, strict=True)
        if isinstance(cri_sd, dict) and len(cri_sd) > 0:
            self.critic.load_state_dict(cri_sd, strict=False)

        logger.info(f"[prefix] Loaded {len(matched)} params from ckpt, "
                    f"{len(new_keys)} new (prefix/gating extras)")
        for k in new_keys:
            logger.info(f"  [NEW] {k}")

    # ------------------------------------------------------------------ #
    #  rollout  – 1-pass SPM pre-forward gate (baseline-aligned logic)
    # ------------------------------------------------------------------ #
    def rollout(self, train_ml=None, train_rl=True, reset=True, speaker=None):
        if self.feedback == 'teacher' or self.feedback == 'argmax':
            train_rl = False

        if reset:
            obs = np.array(self.env.reset())
        else:
            obs = np.array(self.env._get_obs())

        batch_size = len(obs)
        sentence, language_attention_mask, token_type_ids, \
            visual_attention_mask, seq_lengths, perm_idx = self._sort_batch(obs)
        perm_obs = obs[perm_idx]

        device = sentence.device
        # Ours always starts with zero prefix scale before the first learned gate.
        prev_g = torch.zeros(
            batch_size, self.vln_bert.num_prefix_modules,
            device=device, dtype=torch.float32,
        )

        h_t, language_features = self.vln_bert(
            mode='language',
            sentence=sentence,
            token_type_ids=token_type_ids,
            lang_masks=language_attention_mask,
            prefix_scales=torch.zeros_like(prev_g),
        )
        # VilBERT language features exclude the leading CLS mask during visual fusion.
        language_attention_mask = language_attention_mask[:, 1:]

        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
            'predObjId': None,
        } for ob in perm_obs]

        last_dist = np.zeros(batch_size, np.float32)
        for i, ob in enumerate(perm_obs):
            last_dist[i] = ob['distance']

        ended = np.array([False] * batch_size)
        just_ended = np.array([False] * batch_size)

        rewards, hidden_states, policy_log_probs, masks, entropys = [], [], [], [], []
        stop_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
        ml_loss = 0.0
        ref_loss = 0.0
        gate_smooth_loss = 0.0   # L2 between consecutive gate outputs
        gate_smooth_steps = 0

        for t in range(self.episode_len):
            input_a_t, f_t, candidate_feat, candidate_leng, \
                obj_feat, obj_pos, obj_leng = self.get_input_feat(perm_obs)

            visual_temp_mask = (utils.length2mask(candidate_leng) == 0).bool()
            obj_temp_mask = (utils.length2mask(obj_leng) == 0).bool()

            self.vln_bert.vln_bert.config.directions = max(candidate_leng)
            self.vln_bert.vln_bert.config.obj_directions = max(obj_leng)

            cand_drop = candidate_feat.clone()
            obj_drop = obj_feat.clone()
            if self.vln_bert.training:
                if args.angle_feat_size > 0:
                    cand_drop[..., :-args.angle_feat_size] = self.vln_bert.drop_env(
                        cand_drop[..., :-args.angle_feat_size]
                    )
                else:
                    cand_drop = self.vln_bert.drop_env(cand_drop)
                obj_drop[..., :-4] = self.vln_bert.drop_env(obj_drop[..., :-4])

            cand_pad_mask = utils.length2mask(candidate_leng).bool()
            if args.angle_feat_size > 0:
                cand_vis = cand_drop[..., :-args.angle_feat_size]
            else:
                cand_vis = cand_drop
            valid = (~cand_pad_mask).unsqueeze(-1).float()
            cand_sum = (cand_vis * valid).sum(dim=1)
            denom = valid.sum(dim=1).clamp(min=1.0)
            cand_summary = cand_sum / denom

            cand_count = torch.tensor(candidate_leng, dtype=torch.float32, device=device).unsqueeze(1)
            cand_count_norm = cand_count / float(max(1, max(candidate_leng)))
            step_norm = torch.full((batch_size, 1), float(t) / float(max(1, self.episode_len)), device=device)
            stats_t = torch.cat([cand_count_norm, step_norm], dim=1)

            # Learned gate is part of the retained ours method, not an ablation.
            g_t = self.gate_policy(h_t, stats_t, cand_summary).float()

            # Gate temporal smoothness: L2(g_t - g_{t-1})
            if t >= 1 and torch.is_tensor(g_t) and torch.is_tensor(prev_g):
                gate_smooth_loss = gate_smooth_loss + ((g_t - prev_g) ** 2).mean()
                gate_smooth_steps += 1

            self.logs['gate_mean'].append(g_t.mean().item())
            for bi in range(g_t.size(1)):
                self.logs[f'gate_block_{bi}'].append(g_t[:, bi].mean().item())

            h_t, logit, logit_REF = self.vln_bert(
                mode='visual',
                sentence=language_features,
                token_type_ids=token_type_ids,
                h_t=h_t,
                action_feats=input_a_t,
                pano_feats=f_t,
                cand_feats=cand_drop,
                obj_feats=obj_drop,
                obj_pos=obj_pos,
                lang_masks=language_attention_mask,
                cand_masks=visual_temp_mask,
                obj_masks=obj_temp_mask,
                act_t=t,
                already_dropfeat=True,
                prefix_scales=g_t,
            )
            hidden_states.append(h_t)

            candidate_mask = utils.length2mask(candidate_leng)
            candidate_mask = torch.cat((candidate_mask, stop_mask), dim=-1)
            logit.masked_fill_(candidate_mask, -float('inf'))

            candidate_mask_obj = utils.length2mask(obj_leng)
            logit_REF.masked_fill_(candidate_mask_obj, -float('inf'))

            if train_ml is not None:
                target = self._teacher_action(perm_obs, ended, candidate_mask.size(1))
                ml_loss += self.criterion(logit, target)

            if self.feedback == 'teacher':
                a_t = target
            elif self.feedback == 'argmax':
                _, a_t = logit.max(1)
                a_t = a_t.detach()
                log_probs = F.log_softmax(logit, 1)
                policy_log_probs.append(log_probs.gather(1, a_t.unsqueeze(1)))
            elif self.feedback == 'sample':
                probs = F.softmax(logit, 1)
                c = torch.distributions.Categorical(probs)
                self.logs['entropy'].append(c.entropy().sum().item())
                entropys.append(c.entropy())
                a_t = c.sample().detach()
                policy_log_probs.append(c.log_prob(a_t))
            else:
                sys.exit('Invalid feedback option')

            cpu_a_t = a_t.cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                if ((next_id == visual_temp_mask.size(1)) or (t == self.episode_len - 1)) and (not ended[i]):
                    just_ended[i] = True
                    if self.feedback == 'argmax':
                        _, ref_t = logit_REF[i].max(0)
                        if ref_t != obj_leng[i] - 1:
                            traj[i]['predObjId'] = perm_obs[i]['candidate_obj'][2][ref_t]
                  
                else:
                    just_ended[i] = False

                if (next_id == visual_temp_mask.size(1)) or (next_id == args.ignoreid) or ended[i]:
                    cpu_a_t[i] = -1

            if train_ml is not None:
                target_obj = self._teacher_REF(perm_obs, just_ended)
                ref_loss += self.criterion_REF(logit_REF, target_obj)

            self.make_equiv_action(cpu_a_t, perm_obs, perm_idx, traj)
            obs = np.array(self.env._get_obs())
            perm_obs = obs[perm_idx]

            if train_rl:
                dist = np.zeros(batch_size, np.float32)
                reward = np.zeros(batch_size, np.float32)
                mask = np.ones(batch_size, np.float32)
                for i, ob in enumerate(perm_obs):
                    dist[i] = ob['distance']
                    if ended[i]:
                        reward[i] = 0.0
                        mask[i] = 0.0
                    else:
                        action_idx = cpu_a_t[i]
                        if action_idx == -1:
                            if dist[i] < 1.0:
                                reward[i] = 2.0
                            else:
                                reward[i] = -2.0
                        else:
                            reward[i] = -(dist[i] - last_dist[i])
                            if reward[i] > 0.0:
                                reward[i] = 1.0
                            elif reward[i] < 0.0:
                                reward[i] = -1.0
                            else:
                                raise NameError("The action doesn't change the move")
                            if (last_dist[i] <= 1.0) and (dist[i] - last_dist[i] > 0.0):
                                reward[i] -= (1.0 - last_dist[i]) * 2.0
                rewards.append(reward)
                masks.append(mask)
                last_dist[:] = dist

            ended[:] = np.logical_or(ended, (cpu_a_t == -1))
            prev_g = g_t.detach()

            if ended.all():
                break

        if train_rl:
            input_a_t, f_t, candidate_feat, candidate_leng, \
                obj_feat, obj_pos, obj_leng = self.get_input_feat(perm_obs)

            visual_temp_mask = (utils.length2mask(candidate_leng) == 0).bool()
            obj_temp_mask = (utils.length2mask(obj_leng) == 0).bool()
            self.vln_bert.vln_bert.config.directions = max(candidate_leng)
            self.vln_bert.vln_bert.config.obj_directions = max(obj_leng)

            last_h_, _, _ = self.vln_bert(
                mode='visual',
                sentence=language_features,
                token_type_ids=token_type_ids,
                h_t=h_t,
                action_feats=input_a_t,
                pano_feats=f_t,
                cand_feats=candidate_feat,
                obj_feats=obj_feat,
                obj_pos=obj_pos,
                lang_masks=language_attention_mask,
                cand_masks=visual_temp_mask,
                obj_masks=obj_temp_mask,
                act_t=len(hidden_states),
                already_dropfeat=False,
                prefix_scales=prev_g,
            )

            rl_loss = 0.0
            last_value__ = self.critic(last_h_).detach()
            discount_reward = np.zeros(batch_size, np.float32)
            for i in range(batch_size):
                if not ended[i]:
                    discount_reward[i] = last_value__[i]

            length = len(rewards)
            total = 0
            for tstep in range(length - 1, -1, -1):
                discount_reward = discount_reward * args.gamma + rewards[tstep]
                mask_ = torch.from_numpy(masks[tstep]).cuda()
                r_ = torch.from_numpy(discount_reward.copy()).cuda()
                v_ = self.critic(hidden_states[tstep])
                a_ = (r_ - v_).detach()

                rl_loss += (-policy_log_probs[tstep] * a_ * mask_).sum()
                rl_loss += (((r_ - v_) ** 2) * mask_).sum() * 0.5
                if self.feedback == 'sample':
                    rl_loss += (-0.01 * entropys[tstep] * mask_).sum()
                self.logs['critic_loss'].append((((r_ - v_) ** 2) * mask_).sum().item())
                total = total + np.sum(masks[tstep])

            self.logs['total'].append(total)

            if args.normalize_loss == 'total':
                rl_loss /= total
            elif args.normalize_loss == 'batch':
                rl_loss /= batch_size
            else:
                assert args.normalize_loss == 'none'

            self.loss += rl_loss
            self.logs['RL_loss'].append(rl_loss.item())

        if train_ml is not None:
            self.loss += ml_loss * train_ml / batch_size
            self.logs['IL_loss'].append((ml_loss * train_ml / batch_size).item())
            self.loss += ref_loss * args.ref_loss_weight / batch_size
            self.logs['REF_loss'].append(ref_loss.item() * args.ref_loss_weight / batch_size)

        # Gate temporal smoothness loss
        lambda_smooth = getattr(args, 'lambda_smooth', 0.01)
        if gate_smooth_steps > 0 and lambda_smooth > 0:
            gate_smooth_avg = gate_smooth_loss / gate_smooth_steps
            self.loss += lambda_smooth * gate_smooth_avg
            self.logs['gate_smooth_loss'].append(gate_smooth_avg.item())
        else:
            self.logs['gate_smooth_loss'].append(0.0)

        self.logs['gate_entropy_corr'].append(0.0)
        if type(self.loss) is int:
            self.losses.append(0.0)
        else:
            self.losses.append(self.loss.item() / self.episode_len)
        return traj

    # ------------------------------------------------------------------ #
    #  train  – same pattern as Seq2SeqAgent.train
    # ------------------------------------------------------------------ #
    def train(self, n_iters, feedback='teacher', **kwargs):
        self.feedback = feedback
        self.vln_bert.train()
        self.critic.train()
        self.gate_policy.train()
        self.losses = []

        for iter_i in range(1, n_iters + 1):
            self.vln_bert_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            self.gate_optimizer.zero_grad(set_to_none=True)
            self.loss = 0

            if feedback == 'teacher':
                self.feedback = 'teacher'
                self.rollout(
                    train_ml=args.teacher_weight, train_rl=False, **kwargs)
            elif feedback == 'sample':
                if args.ml_weight != 0:
                    self.feedback = 'teacher'
                    self.rollout(
                        train_ml=args.ml_weight, train_rl=False, **kwargs)
                self.feedback = 'sample'
                self.rollout(train_ml=None, train_rl=True, **kwargs)
            else:
                assert False

            self.loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.vln_bert.parameters(), 40.)

            self.vln_bert_optimizer.step()
            self.critic_optimizer.step()
            self.gate_optimizer.step()

            print_progress(
                iter_i, n_iters,
                prefix='Progress:', suffix='Complete', bar_length=50)

        # Release grad buffers between clients/rounds.
        self.vln_bert_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.gate_optimizer.zero_grad(set_to_none=True)
