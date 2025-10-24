import numpy as np
from src.env.vector_mnk_env import VectorMnkEnv


def print_state(env, step_name):
    """Wypisz stan środowiska dla debugowania."""
    print(f"\n=== {step_name} ===")
    print(f"Agent selection: {env.agent_selection}")
    print(f"Terminations: {env.terminations}")
    print(f"Rewards:\n{env.rewards}")
    print("Boards:")
    for i in range(env.num_envs):
        print(f"\nEnv {i}:")
        print(f"Black:\n {env.boards[i, 0]}")
        print(f"White:\n {env.boards[i, 1]}")

    obs, rewards, terminations, truncations, infos = env.last()
    print(f"Last() rewards: {rewards}")
    print(f"Last() terminations: {terminations}")


def test_basic_functionality():
    """Test podstawowej funkcjonalności z 3 środowiskami."""
    print("=== TEST 1: Podstawowa funkcjonalność ===")

    # Tworzymy środowisko 3x3x3 z 3 równoległymi środowiskami
    env = VectorMnkEnv(m=3, n=3, k=3, parallel=3)

    # Reset wszystkich środowisk
    env.reset(np.arange(3))
    print_state(env, "Po resecie")

    # Krok 1: Ruch tylko w środowisku 0 i 2
    actions = np.array([0, None, 4], dtype=object)  # Env 0: (0,0), Env 2: (1,1)
    env.step(actions)
    print_state(env, "Po kroku 1 (akcje w env 0, 2)")

    # Krok 2: Ruch tylko w środowisku 1
    actions = np.array([None, 1, None], dtype=object)  # Env 1: (0,1)
    env.step(actions)
    print_state(env, "Po kroku 2 (akcja w env 1)")

    # Krok 3: Kontynuacja gry w środowisku 0 i 2
    actions = np.array([3, None, 8], dtype=object)  # Env 0: (1,0), Env 2: (2,2)
    env.step(actions)
    print_state(env, "Po kroku 3 (akcje w env 0, 2)")


def test_win_detection_bug():
    """Test wykrywania błędu w _check_wins - krytyczny błąd indeksowania."""
    print("\n\n=== TEST 2: Wykrywanie błędu w _check_wins ===")

    # Środowisko 3x3x3 z 3 środowiskami
    env = VectorMnkEnv(m=3, n=3, k=3, parallel=3)

    # Reset wszystkich środowisk
    env.reset(np.arange(3))

    # Symuluj wygraną w środowisku 2 (indeks 2), ale nie w 0 i 1
    # Czarne wygrywają w środowisku 2: (0,0), (0,1), (0,2)

    # Krok 1: Czarne w env 2 na (0,0)
    actions = np.array([None, None, 0], dtype=object)
    env.step(actions)
    print_state(env, "Env 2: Czarne (0,0)")

    # Krok 2: Białe w env 2 na (1,0)
    actions = np.array([None, None, 3], dtype=object)
    env.step(actions)
    print_state(env, "Env 2: Białe (1,0)")

    # Krok 3: Czarne w env 2 na (0,1)
    actions = np.array([None, None, 1], dtype=object)
    env.step(actions)
    print_state(env, "Env 2: Czarne (0,1)")

    # Krok 4: Białe w env 2 na (1,1)
    actions = np.array([None, None, 4], dtype=object)
    env.step(actions)
    print_state(env, "Env 2: Białe (1,1)")

    # Krok 5: Czarne w env 2 na (0,2) - POWINNA BYĆ WYGRANA!
    print("\n*** OCZEKIWANA WYGRANA W ŚRODOWISKU 2 ***")
    actions = np.array([None, None, 2], dtype=object)
    env.step(actions)
    print_state(env, "Env 2: Czarne (0,2) - POWINNA BYĆ WYGRANA!")

    # Sprawdź czy wykryto wygraną
    if env.terminations[2]:
        print("✓ WYGRANA WYKRYTA PRAWIDŁOWO w środowisku 2")
    else:
        print("✗ BŁĄD: Wygrana NIE została wykryta w środowisku 2")
        print("To wskazuje na błąd w _check_wins() - linia 104")


def test_reward_management():
    """Test zarządzania nagrodami."""
    print("\n\n=== TEST 3: Zarządzanie nagrodami ===")

    env = VectorMnkEnv(m=3, n=3, k=3, parallel=3)
    env.reset(np.arange(3))

    # Ustaw różne nagrody w różnych środowiskach
    env.rewards[0] = [0.5, -0.5]  # Env 0: czarne wygrywa
    env.rewards[1] = [-1.0, 1.0]  # Env 1: białe wygrywa
    env.rewards[2] = [0.0, 0.0]  # Env 2: remis

    print("Przed krokiem:")
    print(f"Rewards:\n{env.rewards}")

    # Wykonaj krok tylko w środowisku 1
    actions = np.array([None, 4, None], dtype=object)
    env.step(actions)

    print("\nPo kroku (tylko env 1):")
    print(f"Rewards:\n{env.rewards}")

    # Sprawdź czy nagrody w env 0 i 2 zostały zachowane
    if np.allclose(env.rewards[0], [0.0, 0.0]) and np.allclose(env.rewards[2], [0.0, 0.0]):
        print("✗ BŁĄD: Nagrody w env 0 i 2 zostały zresetowane!")
        print("Powinny być zachowane, bo nie było w nich akcji")
    else:
        print("✓ Nagrody w nieaktywnych środowiskach zachowane")


def test_mixed_termination():
    """Test mieszanych stanów termination w różnych środowiskach."""
    print("\n\n=== TEST 4: Mieszane stany termination ===")

    env = VectorMnkEnv(m=3, n=3, k=3, parallel=3)
    env.reset(np.arange(3))

    # Symuluj zakończone środowisko 1
    env.terminations[1] = True

    print("Przed próbą akcji w zakończonym środowisku:")
    print(f"Terminations: {env.terminations}")

    # Spróbuj wykonać akcję w środowisku 1 (zakończonym)
    try:
        actions = np.array([0, 4, 8], dtype=object)  # Including action in terminated env 1
        env.step(actions)
        print("✗ BŁĄD: Powinien być rzucony ValueError!")
    except ValueError as e:
        print(f"✓ Prawidłowo rzucono ValueError: {e}")


if __name__ == "__main__":
    print("DEBUG VECTOR MNK ENV")
    print("=" * 50)

    test_basic_functionality()
    test_win_detection_bug()
    test_reward_management()
    test_mixed_termination()

    print("\n\n" + "=" * 50)
    print("Zakończono testy. Sprawdź wyniki powyżej.")
