import torch
from typing import Dict, Tuple, Optional
from env.torch_vector_mnk_env import TorchVectorMnkEnv
from selfplay.policy import Policy


class TorchSelfPlayWrapper:
    """
    Vectorized Self-Play Wrapper for the MNK game on GPU.

    Implements the `AutoresetMode.NEXT_STEP` logic:
    - When an episode finishes (Done=True), the observation returned contains the
      terminal state (e.g., the winning line).
    - The actual reset happens at the *beginning* of the next `step()` call.
    - It manages the opponent's turns automatically, ensuring the external Agent
      always perceives the environment as "My Turn".
    """

    def __init__(self, env: TorchVectorMnkEnv):
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs

        # 0 = Agent plays Black, 1 = Agent plays White
        # Initialized randomly
        self.agent_side = torch.randint(0, 2, (self.num_envs,), device=self.device)
        self.opponent_policy = None

        # Autoreset mask: Tracks environments that finished in the previous step
        # and are waiting to be reset at the start of the current step.
        self.autoreset_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def set_opponent(self, policy: Policy):
        self.opponent_policy = policy

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Resets all environments immediately.

        Args:
            options: Optional dict. If "agent_side" is present, it forces the agent's
                     color (0 or 1) for deterministic testing.
        """
        # 1. Physical environment reset (clears boards)
        self.env.reset()

        # 2. Determine sides (Deterministic via options or Random)
        if options is not None and "agent_side" in options:
            side_val = options["agent_side"]
            if isinstance(side_val, int):
                self.agent_side = torch.full(
                    (self.env.num_envs,), side_val, device=self.device, dtype=torch.long
                )
            else:
                self.agent_side = torch.as_tensor(
                    side_val, device=self.device, dtype=torch.long
                )
        else:
            self.agent_side = torch.randint(0, 2, (self.env.num_envs,), device=self.device)

        # 3. Clear autoreset mask (we are starting fresh)
        self.autoreset_mask[:] = False

        # 4. Handle Opponent's First Move
        # If Agent is White (1), Opponent (Black) must move first.
        self._resolve_opponent_turns()

        return self.get_agent_obs()

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict]:
        """
        Executes a step in the vectorized environment following NEXT_STEP logic.

        Flow:
        1. Validate turns (Fail-Fast).
        2. Reset environments that finished in the previous step (ignoring inputs for them).
        3. Execute Agent moves in active environments.
        4. Execute Opponent moves immediately after, if the game continues.
        5. Return observations from the Agent's perspective.
        """

        # --- VALIDATION: Strict Turn Order ---
        # Ensure the Agent only acts in environments where it is actually its turn.
        # We skip validation for environments pending reset (as their state is transient).
        active_mask = ~self.autoreset_mask

        if active_mask.any():
            active_indices = torch.nonzero(active_mask).squeeze(1)

            current_players = self.env.current_player[active_indices]
            agent_sides = self.agent_side[active_indices]
            is_agent_turn = current_players == agent_sides

            if not is_agent_turn.all():
                # Map local index back to global index for error reporting
                bad_local_idx = int(torch.nonzero(~is_agent_turn)[0].item())
                bad_global_idx = int(active_indices[bad_local_idx].item())
                raise ValueError(
                    f"Turn Mismatch: Agent tried to act in Env {bad_global_idx}, but it is Opponent's turn there."
                )

        # --- 1. IDENTIFY WORKLOAD ---
        # Environments waiting for reset (finished in prev step) -> RESET
        envs_to_reset_mask = self.autoreset_mask
        # Environments active -> ACT
        envs_to_act_mask = ~self.autoreset_mask

        # Output buffers
        total_rewards = torch.zeros(self.num_envs, device=self.device)
        total_dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # --- 2. HANDLE AUTORESET ---
        # Reset environments that finished previously. Agent actions here are ignored.
        if envs_to_reset_mask.any():
            reset_indices = torch.nonzero(envs_to_reset_mask).squeeze(1)

            # A. Physical Reset
            self.env.reset(reset_indices)

            # B. Randomize sides for the new game
            self.agent_side[reset_indices] = torch.randint(
                0, 2, (len(reset_indices),), device=self.device
            )

            # C. Opponent moves first (if needed)
            self._resolve_opponent_turns(reset_indices)

        # --- 3. HANDLE ACTIVE STEPS ---
        # Execute Agent actions and subsequent Opponent responses.
        if envs_to_act_mask.any():
            active_indices = torch.nonzero(envs_to_act_mask).squeeze(1)

            # A. Agent Move
            _, subset_rewards, subset_dones = self.env.step_subset(
                actions[active_indices], active_indices
            )

            # Store Agent results
            total_rewards[active_indices] = subset_rewards
            total_dones[active_indices] = subset_dones

            # B. Opponent Move (Only where the game is still ongoing)
            subset_still_playing = ~subset_dones

            if subset_still_playing.any():
                global_opp_indices = active_indices[subset_still_playing]

                opp_rewards, opp_dones = self._resolve_opponent_turns(global_opp_indices)

                if opp_dones is not None:
                    # If Opponent wins, Agent gets -1.0 reward.
                    total_rewards[global_opp_indices] -= opp_rewards
                    # Game ends if Opponent wins.
                    total_dones[global_opp_indices] = (
                        total_dones[global_opp_indices] | opp_dones
                    )

        # --- 4. STATE UPDATE ---
        # Mark environments that finished NOW to be reset in the NEXT step.
        self.autoreset_mask = total_dones.clone()

        # --- 5. OBSERVATION ---
        # Returns canonical observations (Channel 0 = Self).
        # - Reset envs: Start of a new game.
        # - Active envs: State after Agent+Opp moves (or terminal state if Agent won).
        return self.get_agent_obs(), total_rewards, total_dones, {}

    def get_agent_obs(self, env_indices=None) -> Dict[str, torch.Tensor]:
        """
        Returns observations transformed to the Agent's Canonical View.

        Canonical View:
        - Channel 0: Agent's pieces ("Me")
        - Channel 1: Opponent's pieces ("Enemy")
        """
        if env_indices is None:
            env_indices = torch.arange(self.env.num_envs, device=self.device)

        raw_obs = self.env.observe()

        obs = raw_obs["observation"][env_indices]
        mask = raw_obs["action_mask"][env_indices]

        my_side = self.agent_side[env_indices]
        i_am_white = my_side == 1

        # If Agent is White (1), its pieces are in Channel 1.
        # We flip channels [0, 1] -> [1, 0] so pieces move to Channel 0.
        if i_am_white.any():
            obs = obs.clone()
            obs[i_am_white] = torch.flip(obs[i_am_white], dims=(1,))

        return {"observation": obs, "action_mask": mask}

    def _resolve_opponent_turns(self, env_indices: Optional[torch.Tensor] = None):
        """
        Executes the opponent's policy in the specified environments.

        Returns:
            (rewards, dones): Tensors aligned with `env_indices` containing
                              results of the opponent's moves.
        """
        if self.opponent_policy is None:
            return None, None

        if env_indices is None:
            env_indices = torch.arange(self.num_envs, device=self.device)

        # 1. Identify turns
        current_players = self.env.current_player[env_indices]
        my_sides = self.agent_side[env_indices]
        is_opp_turn = current_players != my_sides

        if not is_opp_turn.any():
            return None, None

        global_opp_indices = env_indices[is_opp_turn]

        # 2. Prepare Observation for Opponent (Canonical View)
        full_obs = self.env.observe()
        opp_obs = {
            "observation": full_obs["observation"][global_opp_indices],
            "action_mask": full_obs["action_mask"][global_opp_indices],
        }

        # Flip view if Opponent is playing White (meaning Agent is Black/0)
        opp_is_white = self.agent_side[global_opp_indices] == 0
        if opp_is_white.any():
            opp_obs["observation"] = opp_obs["observation"].clone()
            opp_obs["observation"][opp_is_white] = torch.flip(
                opp_obs["observation"][opp_is_white], dims=(1,)
            )

        # 3. Inference
        with torch.no_grad():
            opp_actions = self.opponent_policy.act(opp_obs)
            # Ensure action shape consistency
            if isinstance(opp_actions, torch.Tensor) and opp_actions.dim() == 0:
                opp_actions = opp_actions.unsqueeze(0)

        # 4. Execute Move
        if opp_actions is not None:
            _, full_env_rewards, full_env_dones = self.env.step_subset(
                opp_actions, global_opp_indices
            )
        else:
            return None, None

        # 5. Map results back to the requested `env_indices` shape
        ret_rewards = torch.zeros(len(env_indices), device=self.device)
        ret_dones = torch.zeros(len(env_indices), dtype=torch.bool, device=self.device)

        # Scatter results based on the turn mask
        ret_rewards[is_opp_turn] = full_env_rewards[global_opp_indices]
        ret_dones[is_opp_turn] = full_env_dones[global_opp_indices]

        return ret_rewards, ret_dones
