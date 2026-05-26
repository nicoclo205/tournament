import numpy as np

from connect4.connect_state import ConnectState
from connect4.policy import Policy


class NewPolicy(Policy):

    def __init__(self, N_iterations=200, exploration_constant=1.41) -> None:
        self.rng: np.random.Generator = np.random.default_rng()
        self.N_iterations = N_iterations
        self.exploration_constant = exploration_constant

    def mount(self) -> None:
        """
        Resets the enviroment with random values
        """
        self.rng = np.random.default_rng()

    def act(self, s: np.ndarray) -> int:
        """
        Define the way the algorithm will act
        """
        current_player = self._infer_current_player(s)
        state = ConnectState(s, current_player)
        free_cols = state.get_free_cols()

        # Validation that the game is not over
        if len(free_cols) == 0 or state.is_final():
            return int(self.rng.choice(free_cols)) if free_cols else 0

        # Heuristic immediate win or immediate block
        winning_move = self._find_immediate_move(state, current_player)
        block_move = self._find_immediate_move(state, -current_player)

        if winning_move is not None:
            return winning_move
        
        if block_move is not None:
            return block_move
        
        return self._ucb1(state, current_player)
        
    def _find_immediate_move(self, state: ConnectState, player: int) -> int | None:
        """
        Find immediate moves for situations where the agent can win immediatly
        block for preventing the opponent to win
        """
        temp_state = ConnectState(state.board, player)

        if temp_state.is_final():
            return None
        
        for col in temp_state.get_free_cols():
            next_state = temp_state.transition(col)
            if next_state.get_winner() == player:
                return int(col)
        
        return None
    
    def _infer_current_player(self, board: np.ndarray) -> int:
        """
        Infer what is the current player
        """
        reds = int(np.sum(board == -1))
        yellows = int(np.sum(board == 1))

        return -1 if reds == yellows else 1

    def _rollout(self, state: ConnectState, root_player: int) -> int:
        """
        Simulates a random game from a given state
        """
        while not state.is_final():
            free_cols = state.get_free_cols()

            win = self._find_immediate_move(state, state.player)
            if win is not None:
                col = win
            else:
                block = self._find_immediate_move(state, -state.player)
                if block is not None:
                    col = block
                else:
                    base_weights = [1, 2, 3, 4, 3, 2, 1]
                    weights = np.array([base_weights[c] for c in free_cols], dtype=float)
                    weights /= weights.sum()
                    col = int(self.rng.choice(free_cols, p=weights))

            state = state.transition(col)
            
        winner = state.get_winner()

        if winner == root_player:
            return 1
        elif winner == 0:
            return 0
        else:
            return -1
        
    def _ucb1(self, state: ConnectState, current_player: int) -> int:
        
        free_cols = state.get_free_cols()
        Q = {col: 0.0 for col in free_cols}
        N = {col: 0 for col in free_cols}

        for t in range(1, self.N_iterations + 1):

            best_col = free_cols[0]
            best_score = -float('inf')

            for col in free_cols:
                if N[col] == 0:
                    score = float('inf')
                else:
                    score = Q[col] / N[col] + self.exploration_constant * np.sqrt(np.log(t) / N[col])

                if score > best_score:
                    best_score = score
                    best_col = col

            next_state = state.transition(best_col)
            result = self._rollout(next_state, current_player)

            Q[best_col] += result
            N[best_col] += 1

        return max(free_cols, key=lambda col: Q[col] / N[col])

        



