# Agente Connect-4: UCB1 + Rollouts con Heurística y Bias al Centro

**Autor:** Nicolás  
**Curso:** Fundamentos de Inteligencia Artificial — Corte 3  
**Universidad de La Sabana, 2026.1**

---

## Descripción

Agente para Connect-4 que combina tres estrategias:

1. **Heurística inmediata** — detecta y ejecuta jugadas ganadoras o bloqueantes en el turno actual antes de cualquier búsqueda.
2. **UCB1** (Tema 10) — selecciona qué columna explorar balanceando explotación y exploración mediante la fórmula:

$$\text{UCB1}(col) = \frac{Q(col)}{N(col)} + c \cdot \sqrt{\frac{\ln(t)}{N(col)}}$$

3. **Rollouts guiados** — simulaciones aleatorias con heurística interna (ganar/bloquear) y bias hacia columnas centrales, basado en el principio de simulaciones del Tema 11 y 13.

---

## Estructura del repositorio

```
Nico_agent_v2/
├── agent.py          # Código del agente (clase NewPolicy)
├── entrega.ipynb     # Notebook con análisis, gráficas y conclusiones
└── readme.md         # Este archivo
```

---

## Requisitos

- Python 3.10+
- `numpy`
- `matplotlib` (para el notebook)

El agente no requiere dependencias externas más allá de las ya usadas en el proyecto del torneo.

---

## Uso

### En el torneo

El agente se registra automáticamente al colocarse en la carpeta `groups/Nico_agent_v2/`. El archivo `connect4/utils.py` lo detecta por herencia de `Policy`.

Para ejecutar el torneo completo:

```bash
cd tournament/
python main.py
```

### Prueba individual del agente

```python
from groups.Nico_agent_v2.agent import NewPolicy
from connect4.connect_state import ConnectState

agent = NewPolicy()
agent.mount()

state = ConnectState()          # tablero vacío, jugador rojo empieza
col = agent.act(state.board)    # devuelve columna (0–6)
print(f"Agente elige columna: {col}")
```

### Ajuste de parámetros

Los parámetros se configuran en `__init__` de `NewPolicy`:

| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| `N_iterations` | `200` | Iteraciones UCB1 por turno. Más = mejor pero más lento. |
| `exploration_constant` | `1.41` | Constante $c$ del UCB1 ($\approx \sqrt{2}$). Mayor = más exploración. |

```python
agent = NewPolicy()
agent.N_iterations = 100        # reducir para mayor velocidad
agent.exploration_constant = 2.0  # aumentar para más exploración
agent.mount()
```

---

## Algoritmo

```
act(estado):
    1. Inferir jugador actual por conteo de fichas
    2. ¿Jugada ganadora inmediata?   → retornar columna
    3. ¿Jugada bloqueante inmediata? → retornar columna
    4. Ejecutar UCB1 por N_iterations:
         a. Elegir columna con mayor UCB1(col)
         b. Correr rollout desde esa columna
         c. Actualizar Q[col] y N[col]
    5. Retornar columna con mayor Q[col]/N[col]

rollout(estado):
    Mientras no sea estado final:
        1. ¿Jugada ganadora inmediata? → jugarla
        2. ¿Jugada bloqueante inmediata? → jugarla
        3. Si no → elegir columna con pesos [1,2,3,4,3,2,1]
    Retornar +1 (victoria), -1 (derrota), 0 (empate)
```

---

## Resultados

| Métrica | Valor |
|---|---|
| Win rate vs aleatorio (Rojo) | ≥ 95% |
| Win rate vs aleatorio (Amarillo) | ≥ 95% |
| Jugadas ilegales | 0% |

Ver análisis completo en [`entrega.ipynb`](entrega.ipynb).

---

## Diferencia respecto a otros agentes del grupo

A diferencia de un agente **MCTS completo** (que construye un árbol con nodos, backpropagation y UCT), este agente aplica UCB1 **directamente sobre las 7 columnas** sin mantener árbol entre iteraciones. Esto lo hace más simple conceptualmente, directamente conectado a Multi-Armed Bandits, y más fácil de interpretar y ajustar.
