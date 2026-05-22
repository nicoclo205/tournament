import numpy as np

from connect4.connect_state import ConnectState
from connect4.policy import Policy


class MCTSPolicy(Policy):
    """Agente MCTS para Connect-4.

    Usa Monte Carlo Tree Search con rollout aleatorio y detección de jugadas
    inmediatas ganadoras/bloqueadoras para mejorar la fuerza frente a un agente
    aleatorio.
    """

    def __init__(self) -> None:
        self.rng: np.random.Generator = np.random.default_rng()
        self.n_simulations: int = 400
        self.exploration_constant: float = 1.4

    def mount(self, timeout: float | None = None) -> None:
        self.rng = np.random.default_rng()

    def act(self, s: np.ndarray) -> int:
        current_player = self._infer_current_player(s)
        state = ConnectState(s, current_player)
        free_cols = state.get_free_cols()

        if len(free_cols) == 0 or state.is_final():
            return int(self.rng.choice(free_cols)) if free_cols else 0

        # Jugada ganadora inmediata
        winning_move = self._find_immediate_win(state, current_player)
        if winning_move is not None:
            return winning_move

        # Bloquear victoria inmediata del rival
        blocking_move = self._find_immediate_win(state, -current_player)
        if blocking_move is not None:
            return blocking_move

        return self._run_mcts(state)

    def _infer_current_player(self, board: np.ndarray) -> int:
        reds = int(np.sum(board == -1))
        yellows = int(np.sum(board == 1))
        return -1 if reds == yellows else 1

    def _find_immediate_win(self, state: ConnectState, player: int) -> int | None:
        temp_state = ConnectState(state.board, player)
        if temp_state.is_final():
            return None
        for col in temp_state.get_free_cols():
            next_state = temp_state.transition(col)
            if next_state.get_winner() == player:
                return int(col)
        return None

    def _run_mcts(self, state: ConnectState) -> int:
        assert self.rng is not None
        root_player = state.player
        root = self._Node(state.board, -root_player, parent=None, action=None)

        for _ in range(self.n_simulations):
            node = root

            # Selection
            while not node.is_terminal and not node.has_untried_actions():
                node = node.best_child(self.exploration_constant)

            # Expansion
            if not node.is_terminal:
                node = node.expand(self.rng)

            # Simulation
            reward = self._simulate(node, root_player)

            # Backpropagation
            node.backpropagate(reward)

        best_child = max(root.children, key=lambda child: child.N)
        assert best_child.action is not None
        return int(best_child.action)

    def _simulate(self, node: "MCTSPolicy._Node", root_player: int) -> int:
        assert self.rng is not None
        state = ConnectState(node.board, -node.player)

        while not state.is_final():
            free_cols = state.get_free_cols()
            action = self._rollout_policy(free_cols)
            state = state.transition(action)

        winner = state.get_winner()
        if winner == 0:
            return 0
        return 1 if winner == root_player else -1

    def _rollout_policy(self, free_cols: list[int]) -> int:
        assert self.rng is not None
        return int(self.rng.choice(free_cols))

    class _Node:
        def __init__(
            self,
            board: np.ndarray,
            player: int,
            parent: "MCTSPolicy._Node | None",
            action: int | None,
        ) -> None:
            self.board = board.copy()
            self.player = player
            self.parent = parent
            self.action = action
            self.children: list[MCTSPolicy._Node] = []
            self.N = 0
            self.Q = 0.0

            current_player = -self.player
            state = ConnectState(self.board, current_player)
            self.is_terminal = state.is_final()
            self.untried_actions = state.get_free_cols() if not self.is_terminal else []

        def has_untried_actions(self) -> bool:
            return len(self.untried_actions) > 0

        def expand(self, rng: np.random.Generator) -> "MCTSPolicy._Node":
            action = int(rng.choice(self.untried_actions))
            self.untried_actions.remove(action)

            current_player = -self.player
            next_state = ConnectState(self.board, current_player).transition(action)
            child = MCTSPolicy._Node(
                next_state.board,
                current_player,
                parent=self,
                action=action,
            )
            self.children.append(child)
            return child

        def best_child(self, c: float) -> "MCTSPolicy._Node":
            best_score = -np.inf
            best_child: MCTSPolicy._Node | None = None

            for child in self.children:
                if child.N == 0:
                    score = np.inf
                else:
                    score = child.Q / child.N + c * np.sqrt(np.log(self.N) / child.N)
                if score > best_score:
                    best_score = score
                    best_child = child

            assert best_child is not None
            return best_child

        def backpropagate(self, reward: int) -> None:
            visited: set[int] = set()
            node: MCTSPolicy._Node | None = self
            while node is not None:
                if id(node) not in visited:
                    visited.add(id(node))
                    node.N += 1
                    node.Q += reward
                reward = -reward
                node = node.parent
