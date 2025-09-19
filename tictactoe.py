import random
from collections import defaultdict
import numpy as np
from tqdm import tqdm

# Fix random seeds for reproducibility
random.seed(42)
np.random.seed(42)

X, O, E = 'X', 'O', ' '  # players: agent=X, opponent=O, empty cell=E

def empty_board():
    # Represent board as an immutable tuple of 9 positions
    return tuple([E]*9)

# All winning combinations (rows, columns, diagonals)
WIN_LINES = [
    (0,1,2),(3,4,5),(6,7,8),  # rows
    (0,3,6),(1,4,7),(2,5,8),  # cols
    (0,4,8),(2,4,6)           # diagonals
]

def check_winner(state):
    """Check if X or O has won, or if the board is a draw.
       Returns 'X', 'O', 'DRAW', or None if game is still ongoing.
    """
    for a,b,c in WIN_LINES:
        line = (state[a], state[b], state[c])
        if line == (X,X,X): return X
        if line == (O,O,O): return O
    if E not in state: return 'DRAW'
    return None

def legal_actions(state):
    """Return all indices (0–8) that are empty and can be played."""
    return [i for i,v in enumerate(state) if v == E]

def place(state, idx, p):
    """Return new state after placing player p at index idx."""
    s = list(state)
    s[idx] = p
    return tuple(s)

def opponent_move(state):
    """Opponent O plays uniformly random legal move. 
       If the game is already terminal, return state unchanged.
    """
    if check_winner(state): return state
    acts = legal_actions(state)
    if not acts: return state
    return place(state, np.random.choice(acts), O)

def step(state, action, player):
    """Take one step in the environment.
       X = agent, O = opponent.
       1. Apply agent's action.
       2. Check for terminal (win/loss/draw).
       3. If not terminal, environment leaves it as-is (opponent handled separately).
       Returns (next_state, reward, done) from X’s perspective.
       TODO: This is where transition + reward logic is implemented.
    """
    assert action in legal_actions(state), "Illegal action"
    s1 = place(state, action, player)
    w = check_winner(s1)
    if w == X:  return s1, +1.0, True   # win
    if w == O:  return s1, -1.0, True   # loss
    if w == 'DRAW': return s1, 0.0, True  # draw
    return s1, 0.0, False  # game continues

def random_policy(state):
    """Uniform random policy: choose one of the legal actions at random."""
    acts = legal_actions(state)
    return random.choice(acts)

def generate_episode(policy):
    """Simulate one full game episode starting from empty board.
       Alternate turns between X (agent) and O (opponent).
       TODO: This is the self-play simulator used for Monte Carlo.
    """
    s = empty_board()
    done=False
    states_in_episode = [s]
    rewards = []
    player = X
    while not done:
        a = policy(s)
        s, r, done = step(s, a, player)
        # Switch turns
        player = X if player == O else O
        states_in_episode.append(s)
        rewards.append(r)
    return states_in_episode, rewards

def mc_first_visit_V(num_episodes=50000):
    """Monte Carlo first-visit policy evaluation.
       Simulate many episodes under random policy to estimate V(s).
       TODO: Collect returns and compute averages for each state.
    """
    returns = dict()
    V = dict()
    for _ in tqdm(range(num_episodes), desc="First Visit MC"):
        states_in_episode, rewards = generate_episode(random_policy)
        for i, state in enumerate(states_in_episode[:-1]):
            # G = return from state = sum of rewards from this timestep onward
            G = sum(rewards[i:])
            if state in returns:
                total, count = returns[state]
                returns[state] = (total + G, count + 1)
            else:
                returns[state] = (G, 1)
            total, count = returns[state]
            V[state] = total/count
    return V

def get_all_states():
    """Enumerate all reachable states from the empty board."""
    all_states = set()
    def bfs(state, player):
        if state in all_states:
            return
        all_states.add(state)
        for action in legal_actions(state):
            next_player = X if player==O else O
            next_state, _ , _ = step(state, action, player)
            bfs(next_state, next_player)
    bfs(empty_board(), X)
    return all_states

def iterative_policy_evaluation(num_episodes=50000):
    """Iterative Bellman policy evaluation for the random policy.
       TODO: Compute V(s) by sweeping through all states until convergence.
    """
    V = defaultdict(float)
    all_states = get_all_states()
    eps = 1e-5  # convergence tolerance
    done = False
    num_iters = 0
    while not done:
        delta = 0
        for state in all_states:
            _legal_actions = legal_actions(state)
            N = len(_legal_actions)
            if N == 0:
                V[state] = 0
                continue
            p_a = 1 / N  # uniform random policy
            previous_value = V[state]
            sum_value = 0
            for action in _legal_actions:
                # Transition after agent move
                next_state, r, is_terminal = step(state, action, X)
                o_legal_actions = legal_actions(next_state)
                o_sum_value = 0
                if not is_terminal:
                    # Average over opponent random moves
                    for o_action in o_legal_actions:
                        o_next_state, o_r, _ = step(next_state, o_action, O)
                        o_sum_value += (o_r + V[o_next_state])
                    o_sum_value /= len(o_legal_actions)
                else:
                    o_sum_value += r
                sum_value += p_a * o_sum_value
            V[state] = sum_value
            delta = max(delta, abs(previous_value - sum_value))
        done = (delta < eps)
        num_iters += 1
        print(f"Delta: {delta} / {eps}")
    return V

def pretty_print_state(state):
    """Helper: format board nicely for writing to file."""
    s = str(state[0:3]) + "\n" + str(state[3:6]) + "\n" + str(state[6:9]) + "\n\n"
    return s

def greedy_one_step_with_V(state, V):
    """One-step greedy policy improvement:
       Pick the action leading to successor state with highest V(s').
       TODO: This is the "policy improvement" step.
    """
    _legal_actions = legal_actions(state)
    best_action = np.random.choice(_legal_actions)
    best_value_next_state = -np.inf
    for action in _legal_actions:
        next_state, _, _ = step(state, action, X)
        value = V.get(next_state, -np.inf)
        if value > best_value_next_state:
            best_value_next_state = value
            best_action = action
    return best_action

def improved_policy(V):
    """Return improved policy function π(s) = greedy wrt V(s)."""
    def pi(s):
        return greedy_one_step_with_V(s, V)
    return pi

# ---------- Evaluation ----------
def play_many(policy, n=10000):
    """Play n games under given policy vs random opponent.
       Collect win/draw/loss stats.
    """
    wins=draws=losses=0
    for _ in range(n):
        s = empty_board()
        done=False
        player=X
        while not done:
            a = policy(s)
            s, r, done = step(s, a, player)
            player = X if player == O else O
        if r > 0: wins += 1
        elif r < 0: losses += 1
        else: draws += 1
    return wins/n, draws/n, losses/n

if __name__ == "__main__":
    # Ask user which method to use
    while (r := input("Which method? 'm' for monte carlo first visit 'p' for iterative policy evaluation. [m/p] ")) not in ['m','p']:
        print()

    if r == 'm':
        print("Estimating V under random policy...")
        V = mc_first_visit_V(50000)
        v_empty = V.get(empty_board(), 0.0)
        print("V_random(empty) ≈", round(v_empty, 4))
        # Save values to file
        with open('monte_carlo.txt', 'w') as f:
            for state, value in V.items():
                f.write(f"Value: {value} \n" + pretty_print_state(state))
    elif r == 'p':
        print("Estimating V under random policy with policy iteration...")
        V = iterative_policy_evaluation(50000)
        v_empty = V.get(empty_board(), 0.0)
        print("V_random(empty) ≈", round(v_empty, 4))
        with open('policy_iteration.txt', 'w') as f:
            for state, value in V.items():
                f.write(f"Value: {value} \n" + pretty_print_state(state))

    print("Evaluating random vs random:")
    w,d,l = play_many(random_policy, 10000)
    print(f"Random policy as X vs random O: win={w:.3f}, draw={d:.3f}, loss={l:.3f}")

    pi1 = improved_policy(V)
    print("Evaluating improved policy:")
    w,d,l = play_many(pi1, 10000)
    print(f"Improved policy as X vs random O: win={w:.3f}, draw={d:.3f}, loss={l:.3f}")
