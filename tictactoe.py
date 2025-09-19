import random
from collections import defaultdict
import numpy as np
from tqdm import tqdm

random.seed(42)
np.random.seed(42)

X, O, E = 'X', 'O', ' '  # players and empty

def empty_board():
    return tuple([E]*9)  # immutable state

WIN_LINES = [
    (0,1,2),(3,4,5),(6,7,8),  # rows
    (0,3,6),(1,4,7),(2,5,8),  # cols
    (0,4,8),(2,4,6)           # diagonals
]

def check_winner(state):
    for a,b,c in WIN_LINES:
        line = (state[a], state[b], state[c])
        if line == (X,X,X): return X
        if line == (O,O,O): return O
    if E not in state: return 'DRAW'
    return None

def legal_actions(state):
    return [i for i,v in enumerate(state) if v == E]

def place(state, idx, p):
    s = list(state)
    s[idx] = p
    return tuple(s)

def opponent_move(state):
    """Opponent O plays uniformly random legal move. If terminal, returns state."""
    if check_winner(state): return state
    acts = legal_actions(state)
    if not acts: return state
    return place(state, random.choice(acts), O)

def step(state, action, player):
    """Agent X acts; environment responds with O's random move.
       Returns (next_state, reward, done) from X's perspective.
    """
    assert action in legal_actions(state), "Illegal action"
    s1 = place(state, action, player)
    w = check_winner(s1)
    if w == X:  return s1, +1.0, True
    if w == O:  return s1, -1.0, True
    if w == 'DRAW': return s1, 0.0, True

    return s1, 0.0, False  # ongoing

def random_policy(state):
    acts = legal_actions(state)
    return random.choice(acts)

def generate_episode(policy):
    s = empty_board()
    done=False
    states_in_episode = [s]
    rewards = []
    player = X
    while not done:
        a = policy(s)
        s, r, done = step(s, a, player)
        player = X if player == O else O
        states_in_episode.append(s)
        rewards.append(r)

    return states_in_episode, rewards

def mc_first_visit_V(num_episodes=50000):
    returns = dict()
    V = dict()
    for _ in tqdm(range(num_episodes), desc="First Visit MC"):
        states_in_episode, rewards = generate_episode(random_policy)
        for i, state in enumerate(states_in_episode[:-1]):
            # i represents the reward of the next state in rewards
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
    all_states = set()

    def bfs(state, player):
        if state in all_states:
            return
        all_states.add(state)
        _legal_actions = legal_actions(state)
        for action in _legal_actions:
            next_player = X if player==O else O
            next_state, _ , _ = step(state, action, player)
            bfs(next_state, next_player)
    
    bfs(empty_board(), X)
    return all_states


def iterative_policy_evaluation(num_episodes=50000):
    V = defaultdict(float)
    all_states = get_all_states()
    eps = 1e-5
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
            p_a = 1 / N
            previous_value = V[state]
            sum_value = 0
            for action in _legal_actions:
                next_state, r, is_terminal = step(state, action, X)
                o_legal_actions = legal_actions(next_state)
                o_sum_value = 0
                if not is_terminal:
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
    s = str(state[0:3]) + "\n" + str(state[3:6]) + "\n" + str(state[6:9]) + "\n\n"
    return s

def greedy_one_step_with_V(state, V):
    _legal_actions = legal_actions(state)
    best_action = random.choice(_legal_actions)
    best_value_next_state = -np.inf
    for action in _legal_actions:
        next_state, _, _ = step(state, action, X)
        value = V.get(next_state, -np.inf)
        if value > best_value_next_state:
            best_value_next_state = value
            best_action = action
    
    return best_action

def improved_policy(V):
    def pi(s):
        return greedy_one_step_with_V(s, V)
    return pi

# ---------- Evaluation ----------
def play_many(policy, n=10000):
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
    

    while (r := input("Which method? 'm' for monte carlo first visit 'p' for iterative policy evaluation. [m/p] ")) not in ['m','p']:
        print()

    if r == 'm':
        print("Estimating V under random policy...")
        V = mc_first_visit_V(50000)
        v_empty = V.get(empty_board(), 0.0)
        print("V_random(empty) ≈", round(v_empty, 4))
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
