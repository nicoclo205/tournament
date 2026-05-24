import numpy as np
from connect4.connect_state import ConnectState
from connect4.policy import Policy

import time


# MCTS NODE
class MCTSNode:

    # Constructor
    def __init__(self, state: ConnectState, parent=None, action=None):

        # Atributos básicos
        self.state = state
        self.parent = parent
        self.action = action

        self.children = {}  # Diccionario para asociar con acciones

        self.W = 0.0    # Suma de recompensas
        self.N = 0      # Número de visitas

        # Iniciamos con todas las acciones disponibles para expandir
        self.untried_actions = state.get_free_cols()

    # UCB1
    def ucb1(self, c=1.41):

        # Si no se ha visitado, priorizamos la exploración
        if self.N == 0:
            return float("inf")

        # Si no tiene padre, es la raíz, y no queremos explorarla
        if self.parent is None:
            return float("inf")

        # Cálculo de UCB1
        exploitation = self.W / self.N                              # Promedio de recompensas
        exploration = c * np.sqrt(np.log(self.parent.N) / self.N)   # Término de exploración

        # Retornamos la suma de ambos términos
        return exploitation + exploration


# AGENT
class MaoAgent(Policy):

    iterations : int

    # Configuraciones Especiales para el agente

    # Solo concidera jugadas seguras
    safe_moves_only = True

    # Just random
    just_random = False

    # Estrategia de expansión inteligente
    smart_expand_strategy = True

    # Estrategia de simulación inteligente (Priority)
    smart_simulation_strategy = True

    # Be careful you are runnig out of time
    flag_time_warning = False

    # Invertir backpropagation (castigo por perder en vez de recompensa por ganar)
    inverted_backprop = False


    # Metricas Internas Almacenables
    
    # Tiempo Promedio por movimiento
    avg_time = 0

    # Numero de movimientos realizados (para calcular el promedio)
    move_count = 0

    # Tiempo total de juego para ganar métricas al final de la partida
    total_time = 0 # Guardamos en segundos

    # c de UCB1 (puede ser ajustado o incluso adaptativo)
    c = 1.41


    # Inicialización
    def __init__(self, iterations=1000, **kwargs):

        # Random generator para desempates y simulaciones
        self.rng = np.random.default_rng()

        # IMPORTANT:
        # fixed iterations instead of timeout
        self.iterations = iterations

        # Otras configuraciones pueden ser pasadas por kwargs y almacenadas como atributos si es necesario
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Root del árbol MCTS
        self.root = None

    # Montaje del agente (se llama antes de cada partida)
    def mount(self, timeout=None):

        # Inicializamos el generador de números aleatorios y el nodo raíz
        self.rng = np.random.default_rng()
        self.root = None

    # MAIN POLICY
    def act(self, s: np.ndarray) -> int:

        # Iniciamos el conteo de tiempo para métricas
        start_time = time.time() # Tiempo en segundos al inicio del movimiento

        # Identificar el jugador actual (-1 o 1)
        player = -1 if np.count_nonzero(s) % 2 == 0 else 1

        # Casillas libres segun el estado actual
        free = [c for c in range(7) if s[0, c] == 0]

        if self.just_random:
            return int(self.rng.choice(free))

        '''
        Estrategia de preprocesamiento antes de MCTS:
        Lo que hace diferente a mi solucion de cualquier MCTS basico.
        '''

        # LEVEL 1: WIN IMMEDIATELY
        for col in free:

            # Copiamos el board
            board = s.copy()

            # Aplicamos la jugada
            row = self._drop(board, col, player)

            # Verificamos si ganamos
            if self._check_win(board, row, col, player):
                # En caso de ganar, devolvemos esa columna inmediatamente
                return int(col)

        # LEVEL 2: BLOCK IMMEDIATE LOSS
        for col in free:

            # Copiamos el board
            board = s.copy()

            # Aplicamos la jugada del oponente
            row = self._drop(board, col, -player)

            # Verificamos si el oponente ganaría con esa jugada
            if self._check_win(board, row, col, -player):
                # No podemos dejar que gane ahi. Devolvemos esa columna para bloquearlo.
                return int(col)

        # LEVEL 3: SAFE MOVES ONLY
        # avoid giving immediate wins
        safe_moves = []

        # Nuevmaente, iteramos sobre las columnas libres para evaluar su seguridad
        for col in free:

            # Copiamos el board
            board = s.copy()

            # Simulamos nuestra jugada en esa columna
            self._drop(board, col, player)

            # Establecemos flag de seguridad para esa jugada
            opponent_wins = False

            # Evaluamos todas las posibles respuestas del oponente
            opp_free = [c for c in range(7) if board[0, c] == 0]

            # Ciclo de posibles
            for oc in opp_free:

                # Board temporal para simular la respuesta del oponente
                temp = board.copy()

                # Aplicamos la jugada del oponente en esa columna
                r = self._drop(temp, oc, -player)

                # Verificamos si el oponente ganaría con esa jugada
                if self._check_win(temp, r, oc, -player):

                    # No es segura la jugada
                    opponent_wins = True
                    break

            # Si ninguna respuesta del oponente resulta en una victoria, es una jugada segura
            if not opponent_wins:
                safe_moves.append(col)

        # Si hay jugadas seguras, limitamos el espacio de búsqueda a esas columnas
        if safe_moves and self.safe_moves_only:
            free = safe_moves

        # Jugamos rapido si solo hay una opción segura
        if len(free) == 1:
            return int(free[0])
        
        # Jugamos rapido si se va a quedar sin tiempo
        if self.flag_time_warning and self.avg_time > 0 and (self.total_time + self.avg_time) > 120:
            return int(self.rng.choice(free))

        # CREATE ROOT
        state = ConnectState(board=s, player=player)

        self.root = MCTSNode(state)

        # MCTS LOOP
        # ====================================================
        for _ in range(self.iterations):

            node = self._selection(self.root)

            if not node.state.is_final() and node.untried_actions:
                node = self._expand(node)

            result = self._simulate(node.state)

            self._backprop(node, result)
        # ====================================================
        

        # Actualizamos la métrica de tiempo promedio
        end_time = time.time()
        self.avg_time += (end_time - start_time) - self.avg_time / (self.move_count + 1)
        self.move_count += 1

        # Actualizamos el tiempo total de juego
        self.total_time += (end_time - start_time)

        # FALLBACK (En caso de error, devolvemos una columna aleatoria de las libres)
        if not self.root.children:
            return int(self.rng.choice(free))

        # BEST MOVE = MOST VISITS
        best_action = max(
            self.root.children,
            key=lambda a: self.root.children[a].N
        )

        # devolvemos como int la mejor acción encontrada
        return int(best_action)


    # MCTS PHASES

    # Selección: Descendemos por el árbol usando UCB1 hasta llegar a un nodo no terminal que tenga acciones sin explorar
    def _selection(self, node: MCTSNode) -> MCTSNode:

        # Mientras el nodo no sea terminal, seguimos descendiendo por el árbol
        while not node.state.is_final():

            # Si el nodo tiene acciones sin explorar, lo devolvemos para expandirlo
            if node.untried_actions:
                return node

            # Si no tiene acciones sin explorar, seleccionamos el hijo con el mayor valor de UCB1
            node = max(
                node.children.values(),
                key=lambda n: n.ucb1(self.c)
            )

        # Si llegamos a un nodo terminal, lo devolvemos para simular desde ahí
        return node

    # Expansión: Elegimos una acción no explorada del nodo seleccionado
    def _expand(self, node: MCTSNode) -> MCTSNode:

        # Elegimos una acción no explorada usando la estrategia de expansión inteligente
        # De esta manera no evaluamos cosas muy lejos de lo que puede ser peligroso o beneficioso para el jugador actual
        action = self._smart_expand_choice(
            node.state.board,
            node.untried_actions,
            node.state.player
        ) if self.smart_expand_strategy else int(self.rng.choice(node.untried_actions))

        # Marcamos esa acción como explorada
        node.untried_actions.remove(action)

        # Creamos el estado del nodo hijo aplicando la acción elegida
        child_state = node.state.transition(int(action))

        # Creamos el nodo hijo
        child = MCTSNode(
            state=child_state,
            parent=node,
            action=action
        )

        # Asociamos al padre
        node.children[action] = child

        # Devolvemos el nodo hijo para simular desde ahí
        return child

    # Simulación: Desde el nodo expandido, simulamos una partida aleatoria (con heurísticas) hasta llegar a un estado terminal.
    def _simulate(self, state: ConnectState) -> int:

        # Copiamos el board para no modificar el estado original durante la simulación
        # Eficiencia segun mis pruebas cuesta mucho crear un objeto ConnectState
        board = state.board.copy()

        # Jugador actual en la simulación
        player = state.player

        # Guardamos el jugador raíz para determinar el resultado al final de la simulación
        root_player = player

        # Priorizar columnas centrales
        priority = [3, 2, 4, 1, 5, 0, 6]

        # Iterar buscando un estado terminal
        for _ in range(42):

            # Obtenemos las columnas libres en el estado actual del board
            free = [c for c in range(7) if board[0, c] == 0]

            if not free:
                return 0


            # WIN IMMEDIATELY
            # ====================================================
            chosen = None

            for col in free:

                r = self._drop(board, col, player)

                won = self._check_win(board, r, col, player)

                board[r, col] = 0

                if won:
                    chosen = col
                    break
            # ====================================================


            # BLOCK OPPONENT
            # ====================================================
            if chosen is None:

                for col in free:

                    r = self._drop(board, col, -player)

                    won = self._check_win(board, r, col, -player)

                    board[r, col] = 0

                    if won:
                        chosen = col
                        break
            # ====================================================


            # SAFE CENTRAL MOVE
            # ====================================================
            if chosen is None:

                safe = []

                for col in free:

                    r = self._drop(board, col, player)

                    losing = False

                    opp_free = [c for c in range(7) if board[0, c] == 0]

                    for oc in opp_free:

                        rr = self._drop(board, oc, -player)

                        opp_wins = self._check_win(
                            board,
                            rr,
                            oc,
                            -player
                        )

                        board[rr, oc] = 0

                        if opp_wins:
                            losing = True
                            break

                    board[r, col] = 0

                    if not losing:
                        safe.append(col)

                candidates = safe if safe else free

                if self.smart_simulation_strategy:
                    for pcol in priority:
                        if pcol in candidates:
                            chosen = pcol
                            break
                else:
                    chosen = int(self.rng.choice(candidates))
                
            # ====================================================


            # PLAY MOVE
            # ====================================================

            row = self._drop(board, chosen, player)

            # validacion si se gana...
            if self._check_win(board, row, chosen, player):

                if player == root_player:
                    return 1
                else:
                    return -1

            player = -player

        return 0

    # Backpropagation: Actualizamos las estadísticas de los nodos en el camino desde el nodo simulado
    # hasta la raíz con el resultado de la simulación.
    def _backprop(self, node: MCTSNode, result: int):

        while node is not None:

            node.N += 1
            node.W += result

            if self.inverted_backprop:
                result = -result

            node = node.parent


    # SMART EXPANSION
    def _smart_expand_choice(
        self,
        board: np.ndarray,
        actions: list,
        player: int
    ) -> int:
        '''
        Estrategia de expansión inteligente para elegir la próxima acción a expandir.
        Basada en mi experiencia jugando ajedrez uno no puede hacer la vista gorda a lo que de verdad importa en el juego.
        '''

        priority = [3, 2, 4, 1, 5, 0, 6]

        # immediate win

        for col in actions:

            temp = board.copy()

            r = self._drop(temp, col, player)

            if self._check_win(temp, r, col, player):
                return int(col)

        # immediate block

        for col in actions:

            temp = board.copy()

            r = self._drop(temp, col, -player)

            if self._check_win(temp, r, col, -player):
                return int(col)

        # center priority

        for pcol in priority:
            if pcol in actions:
                return int(pcol)

        return int(self.rng.choice(actions))

    # BOARD UTILS
    @staticmethod
    def _drop(board: np.ndarray, col: int, player: int) -> int:
        '''Simula dejar caer una ficha del jugador en la columna especificada.'''

        for r in range(5, -1, -1):

            if board[r, col] == 0:

                board[r, col] = player

                return r

        raise ValueError(f"Column {col} is full")

    @staticmethod
    def _check_win(
        board: np.ndarray,
        row: int,
        col: int,
        player: int
    ) -> bool:
        '''
        Verifica si el jugador ha ganado después de colocar una ficha en (row, col).
        Para esto, revisa las 4 direcciones posibles (horizontal, vertical, diagonal \ y diagonal /)
        '''

        directions = [
            (0, 1),
            (1, 0),
            (1, 1),
            (1, -1)
        ]

        for dr, dc in directions:

            count = 1

            for sign in [1, -1]:

                nr = row + sign * dr
                nc = col + sign * dc

                while (
                    0 <= nr < 6 and
                    0 <= nc < 7 and
                    board[nr, nc] == player
                ):

                    count += 1

                    nr += sign * dr
                    nc += sign * dc

            if count >= 4:
                return True

        return False