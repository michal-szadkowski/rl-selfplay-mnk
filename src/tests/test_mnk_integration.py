import torch
import pytest
from src.env.torch_vector_mnk_env import TorchVectorMnkEnv
from src.selfplay.torch_self_play_wrapper import TorchSelfPlayWrapper

# -------------------------------------------------------------------------
# 1. NARZĘDZIA POMOCNICZE (FIXTURES & MOCKS)
# -------------------------------------------------------------------------


class ScriptedPolicy:
    """
    Polityka, która zawsze zagrywa w zdefiniowane pole.
    Eliminuje losowość z testów.
    """

    def __init__(self, action_idx: int):
        self.action_idx = action_idx

    def act(self, obs_dict):
        # Ignorujemy obserwację, zawsze gramy w to samo miejsce
        batch_size = obs_dict["action_mask"].shape[0]
        device = obs_dict["action_mask"].device
        return torch.full((batch_size,), self.action_idx, device=device, dtype=torch.long)


@pytest.fixture
def wrapper_factory():
    """
    Tworzy wrapper z 1 środowiskiem i zadanym przeciwnikiem.
    Używamy num_envs=1, aby testować logikę, a nie broadcasting tensorów.
    """

    def _create(opponent_action_idx=0):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Plansza 3x3, k=3
        env = TorchVectorMnkEnv(m=3, n=3, k=3, num_envs=1, device=device)
        wrapper = TorchSelfPlayWrapper(env)
        wrapper.set_opponent(ScriptedPolicy(opponent_action_idx))
        return wrapper

    return _create


# -------------------------------------------------------------------------
# 2. TESTY FIZYKI (ENV)
# -------------------------------------------------------------------------


def test_env_mechanics_win(wrapper_factory):
    """Czy silnik wykrywa wygraną (poziomą)?"""
    wrapper = wrapper_factory()
    env = wrapper.env
    env.reset()

    # Ustawiamy X X _ na wierszu 0
    env.boards[0, 0, 0, 0] = 1
    env.boards[0, 0, 0, 1] = 1

    # Wykonujemy ruch w (0,2) -> index 2
    actions = torch.tensor([2], device=env.device)
    _, rewards, dones = env.step(actions)

    assert dones[0].item() is True
    assert rewards[0].item() == 1.0


def test_env_illegal_move(wrapper_factory):
    """Czy silnik rzuca błąd przy ruchu na zajęte pole?"""
    wrapper = wrapper_factory()
    env = wrapper.env
    env.reset()

    # Zajmujemy pole (0,0)
    env.boards[0, 0, 0, 0] = 1

    # Próbujemy zagrać w (0,0) -> index 0
    actions = torch.tensor([0], device=env.device)

    with pytest.raises(ValueError, match="Illegal Move"):
        env.step(actions)


# -------------------------------------------------------------------------
# 3. TESTY WRAPPERA (LOGIKA GRY)
# -------------------------------------------------------------------------


def test_canonical_view(wrapper_factory):
    """Czy Agent zawsze widzi swoje kamienie na Kanale 0 (Me)?"""
    wrapper = wrapper_factory()

    # A. Agent jest Czarny (0)
    wrapper.reset(options={"agent_side": 0})
    wrapper.env.boards[0, 0, 0, 0] = 1.0  # Czarny kamień na planszy
    obs = wrapper.get_agent_obs()
    assert obs["observation"][0, 0, 0, 0] == 1.0  # Widzi na Kanale 0

    # B. Agent jest Biały (1)
    # Resetujemy i wymuszamy bycie Białym
    # Uwaga: przy resecie na Białego, Opponent (Scripted) zagra w pole 0.
    wrapper.set_opponent(
        ScriptedPolicy(8)
    )  # Niech Opp zagra w (2,2) żeby nie psuć testu (0,0)
    wrapper.reset(options={"agent_side": 1})

    # Stawiamy ręcznie Biały kamień na (0,0)
    wrapper.env.boards[0, 1, 0, 0] = 1.0

    obs = wrapper.get_agent_obs()
    # Agent (Biały) powinien widzieć swój biały kamień na Kanale 0
    assert obs["observation"][0, 0, 0, 0] == 1.0
    # A kamień przeciwnika (Czarnego na 2,2) powinien być na Kanale 1
    assert obs["observation"][0, 1, 2, 2] == 1.0


def test_agent_win_reward(wrapper_factory):
    """Czy wygrana Agenta daje +1 i done=True?"""
    wrapper = wrapper_factory()
    wrapper.reset(options={"agent_side": 0})

    # Agent ma już 2 kamienie
    wrapper.env.boards[0, 0, 0, 0] = 1
    wrapper.env.boards[0, 0, 0, 1] = 1

    # Agent wygrywa ruchem w (0,2) -> index 2
    obs, rewards, dones, _ = wrapper.step(torch.tensor([2], device=wrapper.device))

    assert rewards[0].item() == 1.0
    assert dones[0].item() is True
    # Agent widzi wygraną planszę
    assert obs["observation"][0, 0].sum() == 3.0


def test_opponent_win_penalty(wrapper_factory):
    """
    Czy wygrana Przeciwnika daje Agentowi karę -1 i done=True?
    (Kluczowy test logiki resolve_opponent_turns)
    """
    # Przeciwnik jest zaprogramowany, żeby zagrać w pole 5 (1,2)
    wrapper = wrapper_factory(opponent_action_idx=5)
    wrapper.reset(options={"agent_side": 0})  # Agent zaczyna

    # SYTUACJA NA PLANSZY:
    # Agent (Black): Bezpieczne kamienie na (0,0), (0,1)
    wrapper.env.boards[0, 0, 0, 0] = 1
    wrapper.env.boards[0, 0, 0, 1] = 1

    # Opponent (White): Ma kamienie na (1,0), (1,1). Brakuje mu (1,2) [idx 5].
    wrapper.env.boards[0, 1, 1, 0] = 1
    wrapper.env.boards[0, 1, 1, 1] = 1

    # Agent wykonuje ruch w (2,0) [idx 6]. Nie wygrywa, nie blokuje.
    # Sekwencja: Agent Move -> Opponent Move (w 5) -> Opponent Win.
    obs, rewards, dones, _ = wrapper.step(torch.tensor([6], device=wrapper.device))

    assert dones[0].item() is True, "Gra powinna się skończyć"
    assert rewards[0].item() == -1.0, "Agent powinien dostać karę"

    # Sprawdzenie wizualne: Opponent (Channel 1) powinien mieć 3 kamienie w rzędzie 1
    assert obs["observation"][0, 1, 1, :].sum() == 3.0


def test_autoreset_next_step(wrapper_factory):
    """Czy autoreset działa w trybie NEXT_STEP (opóźniony)?"""
    wrapper = wrapper_factory()
    wrapper.reset(options={"agent_side": 0})

    # Setup: Agent wygrywa w tym kroku
    wrapper.env.boards[0, 0, 0, 0] = 1
    wrapper.env.boards[0, 0, 0, 1] = 1

    # KROK 1: Wygrana
    obs, rewards, dones, _ = wrapper.step(torch.tensor([2], device=wrapper.device))
    assert dones[0].item() is True
    assert rewards[0].item() == 1.0
    # Obserwacja to nadal stara gra (wygrana)
    assert obs["observation"][0, 0].sum() == 3.0

    # KROK 2: Autoreset
    # Dowolna akcja (zostanie zignorowana, bo env jest w stanie autoreset)
    obs_new, rewards_new, dones_new, _ = wrapper.step(torch.tensor([0], device=wrapper.device))

    assert dones_new[0].item() is False
    assert rewards_new[0].item() == 0.0
    # Obserwacja to NOWA gra (pusta)
    assert obs_new["observation"][0, 0].sum() == 0.0


def test_opponent_starts_after_reset(wrapper_factory):
    """
    Czy jeśli wylosujemy Agenta jako Białego, Opponent gra pierwszy?
    """
    # Przeciwnik ma zagrać w środek (1,1) -> index 4
    wrapper = wrapper_factory(opponent_action_idx=4)

    # Resetujemy i wymuszamy, że Agent jest Biały (1)
    # To oznacza, że Przeciwnik (Czarny) zaczyna gre.
    obs = wrapper.reset(options={"agent_side": 1})

    # Agent (Biały) powinien widzieć:
    # Kanał 0 (Me): Pusto
    # Kanał 1 (Enemy): Kamień na środku
    assert obs["observation"][0, 0].sum() == 0.0
    assert obs["observation"][0, 1, 1, 1] == 1.0
