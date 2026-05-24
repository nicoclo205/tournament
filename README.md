# MaoAgent — Un jugador de Cuatro en Línea basado en MCTS

## ¿De qué va esto?

El agente que desarrollé se llama **MaoAgent** y es el foco de este documento.

La idea detrás del agente no es solo hacer un MCTS estándar. Está inspirada en cómo yo juego ajedrez: primero ves lo obvio, lo que te gana o te mata en este turno; y solo si no hay nada urgente te pones a pensar en profundidad. Eso es exactamente lo que hace el agente.

---

## Cómo correr el torneo

```powershell
conda run -n ia-env python main.py
```

`Al menos asi funciona en mi compu`

Eso es todo. El framework encuentra los agentes solo.

---

## Arquitectura del agente

El agente vive en `groups/MaoAgent/policy.py` y usa únicamente **NumPy** como librería externa. No hay ninguna dependencia adicional de IA, no hay modelos preentrenados, no hay nada externo. Todo el razonamiento lo construye el agente en tiempo real a partir del estado del tablero.

### La lógica antes de pensar

Antes de entrar al árbol MCTS, el agente pasa por tres niveles de análisis inmediato. Si alguno resuelve la situación, devuelve la jugada al instante sin gastar ni una iteración del árbol.

**Nivel 1 — Ganar ya:** Si hay una columna que me da la victoria en este turno, la juego. Sin más.

**Nivel 2 — No dejar ganar:** Si el oponente tiene una amenaza inmediata, la bloqueo. Tampoco hay que pensar.

**Nivel 3 — Jugadas seguras:** Esto es más sutil. Antes de dejarle el turno al oponente, me fijo si alguna de mis jugadas posibles le abre una victoria inmediata a él. Si es así, esa jugada no es "segura" y la filtro del espacio de búsqueda. El MCTS solo analiza columnas que no le regalan el juego al contrario.

Si después de estos tres filtros queda una sola opción válida, se juega directamente sin encender el árbol.

### La bandera de tiempo (al estilo ajedrez)

En ajedrez existe la expresión "caérsele la bandera" cuando a un jugador se le acaba el tiempo. Pensando en eso, el agente lleva un registro de su tiempo promedio por movimiento. Si detecta que, al ritmo que va, va a quedarse sin tiempo antes de terminar la partida, activa un modo de emergencia: elige aleatoriamente entre las jugadas seguras disponibles para ir rápido. No es lo ideal, pero puede que el oponente esté igual de presionado.

---

## Las cuatro fases del MCTS

### 1. Selección

Desde la raíz, el agente desciende por el árbol eligiendo siempre el hijo con mayor valor UCB1, hasta encontrar un nodo que tenga acciones sin explorar o que sea terminal.

La fórmula UCB1 que usa el agente es:

```
UCB1 = W/N + c * sqrt(ln(N_padre) / N)
```

Donde `W` es la suma de recompensas acumuladas, `N` es el número de visitas al nodo, y `c` es una constante de exploración ajustable (por defecto `1.41`, que corresponde a `√2`). El balance entre explotar lo que ya funciona y explorar lo desconocido lo controla ese parámetro.

### 2. Expansión

Al llegar a un nodo con acciones sin explorar, el agente elige cuál expandir primero usando `_smart_expand_choice`. Esta función replica la misma lógica de los niveles previos: primero intenta expandir una jugada ganadora, luego una jugada de bloqueo, y si no hay ninguna de esas, prioriza las columnas centrales `[3, 2, 4, 1, 5, 0, 6]` porque el centro del tablero da más conectividad.

Esto hace que el árbol crezca primero hacia las ramas que de verdad importan, en lugar de desperdiciar iteraciones en esquinas irrelevantes.

### 3. Simulación

Desde el nodo recién expandido, se simula una partida completa hasta llegar a un estado terminal. Esta simulación no es aleatoria pura: aplica exactamente la misma lógica de ganar/bloquear/jugar-seguro dentro del rollout. Es decir, durante la simulación también se evitan los movimientos suicidas y se capturan las victorias inmediatas.

El resultado de la simulación es `+1` si gana el jugador raíz, `-1` si pierde, y `0` si empata.

### 4. Backpropagación

El resultado sube por todos los nodos del camino hasta la raíz, actualizando `W` (recompensas) y `N` (visitas). Aquí el agente tiene dos modos:

- **Modo normal:** el resultado se propaga tal cual hacia arriba. Funciona bien cuando la perspectiva de W/N se entiende de forma relativa al jugador que creó el nodo.
- **Modo invertido (`inverted_backprop`):** se multiplica el resultado por -1 en cada nivel. Útil para estudiar cómo cambia la búsqueda cuando el castigo por perder es lo que guía el árbol en vez del premio por ganar.

Al final del bucle MCTS, el agente elige la acción del hijo con mayor número de visitas `N`, que es la medida más robusta de confianza en el árbol.

---

## Flags de configuración

El agente está diseñado para ser estudiable. Se pueden pasar parámetros al constructor para cambiar su comportamiento sin tocar el código:

| Parámetro | Tipo | Defecto | Efecto |
|---|---|---|---|
| `iterations` | int | 1000 | Número de iteraciones MCTS por movimiento |
| `c` | float | 1.41 | Constante de exploración UCB1 |
| `safe_moves_only` | bool | True | Filtrar jugadas que regalan victorias al oponente |
| `smart_expand_strategy` | bool | True | Expansión guiada por heurísticas |
| `smart_simulation_strategy` | bool | True | Simulación guiada por heurísticas |
| `inverted_backprop` | bool | False | Cambio de perspectiva en backpropagación |
| `flag_time_warning` | bool | False | Activar modo de emergencia por tiempo |
| `just_random` | bool | False | Modo totalmente aleatorio (baseline) |

Por defecto, el agente viene con la configuración que mejores resultados ha dado en las pruebas. Para usarlo en el torneo solo hay que conectarlo; no requiere configuración adicional.

---

## Métricas internas

Para poder estudiar el agente y comparar configuraciones, lleva un registro de:

- `avg_time` — tiempo promedio por movimiento (en segundos)
- `move_count` — número de movimientos realizados en la partida
- `total_time` — tiempo total acumulado en la partida

Estos valores se pueden leer al final de una partida para comparar rendimiento entre variantes del agente.

---

## Estructura del repositorio

```
proyecto_ia/
├── connect4/
│   ├── policy.py           # Clase base abstracta Policy
│   ├── connect_state.py    # ConnectState: board 6×7, lógica de transición
│   ├── environment_state.py
│   ├── dtos.py             # Tipos: Participant, Match, Game
│   └── utils.py            # Auto-descubrimiento de políticas
├── groups/
│   └── MaoAgent/
│       └── policy.py       # El agente — no modificar salvo petición explícita
├── tournament.py           # Lógica del torneo y función play()
├── main.py                 # Punto de entrada
└── Entrega.ipynb           # Benchmarks y análisis
```

---

---

## Cosas que se pueden mejorar

El agente funciona bien pero tiene varias limitaciones que son conscientes. La primera y más obvia es que el árbol se reconstruye desde cero en cada turno: en cada movimiento se tira todo lo que se pensó antes y se empieza de nuevo. Lo correcto sería reutilizar el subárbol que corresponde a la jugada que se hizo, que en teoría ya fue explorado. Eso multiplicaría el valor de las iteraciones sin costar nada extra.

Otra limitación es que el filtro de jugadas seguras solo mira un nivel de profundidad: evita regalar victorias inmediatas al oponente, pero no detecta trampas de dos turnos ni amenazas dobles. Un análisis de dos niveles sería más robusto aunque más costoso.

La constante `c` del UCB1 está fija. En la práctica podria ser adaptativa según la fase del juego o segun lo que encontre al menos calibrada con más experimentos sistemáticos; el valor `√2` es el teórico pero no necesariamente el óptimo para Cuatro en Línea, por eso esta la opcion de hacer mas pruebas y experimentos.

El modo de bandera de tiempo es una solución de emergencia bastante cruda: el estimado de tiempo restante asume que todos los movimientos futuros van a costar lo mismo que el promedio hasta ahora (En eso se basa mi teoria vaga) lo cual tiende a no ser verdad porque los primeros movimientos del juego suelen ser más lentos. Se podría afinar con un modelo más preciso.

---

*Desarrollado por Mauricio Suárez — agente inspirado en cómo un jugador de ajedrez afronta una partida: primero lo urgente, luego lo profundo.*
