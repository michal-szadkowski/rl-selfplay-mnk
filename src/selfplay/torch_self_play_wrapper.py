import torch
from typing import Dict, Tuple, Optional, Any
from env.constants import PLAYER_BLACK, PLAYER_WHITE


class TorchSelfPlayWrapper:
    def __init__(self, env):
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs

        self.opponent_policy = None
        self.agent_side = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.pending_resets = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def set_opponent(self, policy):
        self.opponent_policy = policy

    def reset(self, seed=None, options=None):
        self.env.reset()
        self.pending_resets.fill_(False)

        if options and "agent_side" in options:
            self.agent_side[:] = torch.as_tensor(options["agent_side"], device=self.device)
        else:
            self.agent_side = torch.randint(0, 2, (self.num_envs,), device=self.device)

        self._opponent_move_if_needed(torch.arange(self.num_envs, device=self.device))

        return self._get_canonical_obs(), {}

    def step(self, actions: torch.Tensor):
        reset_mask = self.pending_resets.clone()
        play_mask = ~reset_mask

        rewards = torch.zeros(self.num_envs, device=self.device)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        if reset_mask.any():
            reset_idxs = torch.nonzero(reset_mask).squeeze(1)
            self.env.reset(reset_idxs)

            self.agent_side[reset_idxs] = torch.randint(
                0, 2, (len(reset_idxs),), device=self.device
            )
            self._opponent_move_if_needed(reset_idxs)

        if play_mask.any():
            play_idxs = torch.nonzero(play_mask).squeeze(1)

            _, r_ag, t_ag = self.env.step_subset(actions[play_idxs], play_idxs)

            rewards[play_idxs] = r_ag[play_idxs]
            terminated[play_idxs] = t_ag[play_idxs]

            still_playing_global = play_idxs[~terminated[play_idxs]]

            if len(still_playing_global) > 0:
                opp_r, opp_t = self._opponent_move_if_needed(still_playing_global)

                if opp_r is not None:
                    rewards[still_playing_global] -= opp_r[still_playing_global]
                    terminated[still_playing_global] = opp_t[still_playing_global]

        self.pending_resets = terminated.clone()

        return self._get_canonical_obs(), rewards, terminated, torch.zeros_like(terminated), {}

    def _opponent_move_if_needed(self, env_idxs: torch.Tensor):
        """Wykonuje ruch przeciwnika, jeśli jest jego tura. Zwraca pełne tensory rewards/dones."""
        if len(env_idxs) == 0:
            return None, None

        current_player = self.env.current_player[env_idxs]
        my_side = self.agent_side[env_idxs]

        opp_turn_mask = current_player != my_side
        if not opp_turn_mask.any():
            return None, None

        active_opp_idxs = env_idxs[opp_turn_mask]

        full_obs = self.env.observe()
        obs_subset = full_obs["observation"][active_opp_idxs]
        mask_subset = full_obs["action_mask"][active_opp_idxs]

        opp_is_white = self.env.current_player[active_opp_idxs] == PLAYER_WHITE
        if opp_is_white.any():
            obs_subset[opp_is_white] = torch.flip(obs_subset[opp_is_white], dims=(1,))

        with torch.no_grad():
            opp_actions = self.opponent_policy.act(
                {"observation": obs_subset, "action_mask": mask_subset}
            )

        _, r, t = self.env.step_subset(opp_actions, active_opp_idxs)
        return r, t

    def _get_canonical_obs(self):
        raw_obs = self.env.observe()
        obs = raw_obs["observation"].clone()
        mask = raw_obs["action_mask"]

        white_agent_mask = self.agent_side == PLAYER_WHITE
        if white_agent_mask.any():
            obs[white_agent_mask] = torch.flip(obs[white_agent_mask], dims=(1,))

        invalid = mask.sum(dim=1) == 0
        if invalid.any():
            mask[invalid, 0] = True

        return {"observation": obs, "action_mask": mask}

    def get_agent_obs(self):
        return self._get_canonical_obs()
