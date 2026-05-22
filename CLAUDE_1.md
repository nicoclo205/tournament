# Agente Connect-4 — MCTS con First-Visit Monte Carlo

## Estructura del archivo

Un solo archivo `policy.py` dentro de la carpeta del grupo en `tournament/groups/TuGrupo/policy.py`.
Hereda de `connect4.policy.Policy` e implementa `mount()` y `act(s)`.

---

## Interfaz obligatoria

```python
class MCTSPolicy(Policy):
    def mount(self) -> None:
        # inicialización antes de cada partida

    def act(self, s: np.ndarray) -> int:
        # recibe el tablero, retorna columna (0–6)
```

El tablero `s` es un `np.ndarray` de 6x7 donde:
- `0` = vacío
- `-1` = Rojo (mueve primero)
- `1` = Amarillo (mueve segundo)

---

## 1. Inferencia del color propio

`act(s)` no recibe explícitamente el color del agente. Se infiere contando fichas:

- Si `count(-1) == count(1)` → le toca a **Rojo** (-1)
- Si `count(-1) > count(1)` → le toca a **Amarillo** (1)

Esto funciona porque Rojo siempre mueve primero, entonces si hay igual cantidad de fichas es porque nadie ha movido aún o ambos movieron la misma cantidad de veces.

```python
def infer_current_player(board):
    reds = np.sum(board == -1)
    yellows = np.sum(board == 1)
    return -1 if reds == yellows else 1
```

El agente llama esto al inicio de cada `act()` para saber quién es.

---

## 2. Nodo del árbol

Cada nodo guarda:

- `board` — tablero en ese estado
- `player` — jugador que **acaba de mover** para llegar aquí
- `parent` — nodo padre (None si es raíz)
- `action` — columna que llevó del padre a este nodo
- `children` — hijos ya expandidos
- `untried_actions` — columnas válidas aún no expandidas
- `N` — número de visitas → `N[s, a]` del temario
- `Q` — suma acumulada de recompensas → para estimar `q_hat(s, a)`

---

## 3. Las 4 fases de cada simulación

### Fase 1 — Selection (UCB)

Baja por el árbol mientras todos los hijos estén expandidos y el nodo no sea terminal, eligiendo siempre el hijo con mayor UCB:

```
UCB(s, a) = Q/N + sqrt( log(N_padre) / N )
```

- `Q/N` → explotación: prefiere nodos con alta tasa de victoria
- `sqrt(...)` → exploración: prefiere nodos poco visitados
- Si `N == 0` → score infinito (fuerza explorar nodos no visitados)

Viene de UCB1 (Bandits, Clase 10) y Exploration Based on Winning Probabilities (Clase 12).

### Fase 2 — Expansion

Cuando se llega a un nodo con acciones sin explorar:
- Se elige una acción no intentada con `rng`
- Se crea el nodo hijo con el tablero resultante
- Se agrega al árbol y se retorna para simular desde él

### Fase 3 — Simulation (default policy aleatoria)

Desde el nodo expandido, ambos jugadores eligen columnas aleatorias con `rng` hasta que el juego termine. Retorna la recompensa desde la perspectiva del agente:

- `+1` si ganamos
- `-1` si perdemos
- `0` si empate

Este es el "inner trial / trash trial" del temario (Clase 13). Su calidad individual no importa; lo que importa es el promedio acumulado.

### Fase 4 — Backpropagation con First-Visit Monte Carlo

Sube el resultado desde el nodo hoja hasta la raíz actualizando `N` y `Q`.

Aplica **First-Visit Monte Carlo (FVMC, Clase 11)**:
- Cada estado-acción se actualiza **solo una vez** por simulación
- Se garantiza con un `set` de nodos ya visitados en el camino

Al subir un nivel se invierte el reward (`* -1`) porque es un juego de suma cero: lo bueno para uno es malo para el otro (Clase 12).

---

## 4. Decisión final

Después de `n_simulations` iteraciones, se elige la columna del hijo con mayor `N` (más visitado), no el de mayor `Q/N`. Esto es más robusto estadísticamente con presupuestos finitos.

---

## 5. RandomState

```python
self.rng: np.random.RandomState
```

Se inicializa en `mount()` y se usa en expansion y simulation para toda aleatoriedad del agente.

---

## Justificación del temario

| Componente | Clase |
|---|---|
| Alternating Markov Game | Clase 12 |
| Recompensa solo al final, γ=1 | Clase 12 |
| Trial-Based Online Policy Improvement | Clase 13 |
| UCB para selección | Clase 10 + Clase 12 |
| First-Visit Monte Carlo en backprop | Clase 11 |
| Default policy aleatoria en simulación | Clase 13 |

---

## Variables para el análisis (rúbrica)

- `n_simulations` → graficar win rate vs N contra el agente aleatorio
- Color (rojo / amarillo) → comparar desempeño inferido en ambos casos
- Contra sí mismo → debería converger a ~50/50 con el mismo N

---

## Propuesta de mejora

Con pocas simulaciones los rollouts aleatorios evalúan mal los estados lejanos. La mejora natural sería reemplazar la default policy aleatoria por una heurística ligera que detecte amenazas inmediatas, reduciendo el número de simulaciones necesarias para alcanzar buen desempeño.
