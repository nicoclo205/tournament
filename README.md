# Connect Four AI Tournament — Por qué ganó NicoAgent

## Agentes comparados

| | **NicoAgent** (Ganador) | **MaoAgent** (Perdedor) |
|---|---|---|
| Iteraciones | 200 | 1000 |
| Estructura MCTS | Bandit plano (solo raíz) | Árbol completo |
| Rollout | Ponderado por columna central | Safe-moves check en cada paso |
| Backprop | N/A (bandit) | Incorrecta por defecto |

---

## Razones concretas de la derrota de MaoAgent

### 1. Backpropagation incorrecta (bug crítico)
En MCTS con árbol completo, el resultado debe **negarse en cada nivel** al subir,
porque lo que es bueno para un jugador es malo para el otro.
MaoAgent **no hace esto** (`inverted_backprop=False` por defecto), por lo que
los nodos del oponente acumulan estadísticas con el signo equivocado.
NicoAgent evita este problema usando un bandit plano sobre la raíz.

### 2. Simulación excesivamente costosa
Dentro de cada rollout, MaoAgent evalúa `safe_moves` comprobando **todas las
respuestas posibles del oponente** para cada columna (~49 operaciones por paso).
Esto hace cada iteración ~5–7× más lenta, neutralizando la ventaja de tener
5× más iteraciones (1000 vs 200).

### 3. Expansión determinista en simulación
MaoAgent elige siempre la columna central más alta disponible como movimiento
"seguro", lo que vuelve los rollouts **demasiado predecibles y sesgados**.
NicoAgent usa pesos probabilísticos `[1,2,3,4,3,2,1]`, que generan simulaciones
más representativas del espacio real de juego.

### 4. Complejidad innecesaria sin beneficio real
El filtro de jugadas seguras (`safe_moves_only`) en el nivel de decisión puede
**descartar la única jugada ganadora** si todas las columnas libres regalan un
triunfo al oponente en el turno siguiente — situación frecuente al final de partida.
NicoAgent confía en el MCTS para manejar ese trade-off.

---

## Lección

> Más iteraciones no compensan una backpropagation incorrecta.
> Un MCTS simple y correcto supera a uno complejo y con bugs.
