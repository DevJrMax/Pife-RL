# Pife

| Import | Values|
| --- | --- |
| Actions | Discrete |
| Parallel API | No |
| Manual Control | No |
| Agents | agents= ['player_0', 'player_1', ..., 'player_n'] |
| Agents | 2 <= n <= 8 |
| Action Shape | Discrete(107) |
| Action Values | Discrete(107) |
| Observation Shape | (8, 107) |
| Observation Values | [0,1] |


O Pife, também conhecido como Pif Paf, é um jogo de cartas popular no Brasil, especialmente na região Nordeste. Ele é geralmente jogado de dois até oito jogadores.

O objetivo do jogador é formar jogos com as cartas que receber ou comprar.

Os jogos podem ser combinações de três ou mais cartas, em trincas (três cartas do mesmo valor e de naipes diferentes) e sequências (três ou mais cartas seguidas, do mesmo naipe).

## Argumentos


## Observation Space

A observação é um dicionário que contém um elemento 'observation' que é a observação RL usual descrita abaixo, e uma 'action_mask' que contém os movimentos legais, descritos na seção Máscara de Ações Legais.

O espaço de observação principal é 8x104 com as linhas representando diferentes planos e as colunas representando as 104 cartas de dois baralhos de 52 cartas. As cartas são ordenadas por baralho (azul e vermelho), naipe (espadas, copas, ouros e depois paus) e dentro de cada naipe são ordenadas por valor (do Ás ao Rei).

| Row Index | Description                                    |
|:---------:|------------------------------------------------|
|     0     | Mão do jogador atual                       |
|     1     | Carta atual no topo do descarte                  |
|     2     | Cartas na pilha de descarte (excluindo a do topo) |
|     3     | Cartas conhecidas do oponente (cartas retiradas da pilha de descarte, mas não descartadas)                     |
|     4     | Cartas desconhecidas (cartas não retiradas do deck ou na mão do oponente)                                |
|     5     | Cartas que podem formar um jogo                              |
|     6     | 1o jogo formado                                |
|     7     | 2o jogo formado                                |
|     8     | 3o jogo formado                                |

<br>

| Column Index | Description                                       |
|:------------:|---------------------------------------------------|
|    0 - 12    | Baralho Azul - Espadas <br>_`0`: Ace, `1`: 2, ..., `12`: King_     |
|    13 - 25   | Baralho Azul - Copas <br>_`13`: Ace, `14`: 2, ..., `25`: King_   |
|    26 - 38   | Baralho Azul - Ouros <br>_`26`: Ace, `27`: 2, ..., `38`: King_ |
|    39 - 51   | Baralho Azul - Paus <br>_`39`: Ace, `40`: 2, ..., `51`: King_    |
|    52 - 64    | Baralho Vermelho - Espadas <br>_`52`: Ace, `53`: 2, ..., `64`: King_     |
|    65 - 77   | Baralho Vermelho - Copas <br>_`65`: Ace, `66`: 2, ..., `77`: King_   |
|    78 - 90   | Baralho Vermelho - Ouros <br>_`78`: Ace, `79`: 2, ..., `90`: King_ |
|    91 - 103   | Baralho Vermelho - Paus <br>_`91`: Ace, `92`: 2, ..., `103`: King_    |

<br>

## Legal Action mask

Os movimentos legais disponíveis para o agente atual são encontrados no elemento action_mask da observação do dicionário. O action_mask é um vetor binário onde cada índice do vetor representa se a ação é legal ou não. A action_mask será totalmente zero para qualquer agente, exceto aquele de quem é a vez. Fazer um movimento ilegal encerra o jogo com uma recompensa de -1 para o agente que se move ilegalmente e uma recompensa de 0 para todos os outros agentes.

## Action Space


Existem 107 ações no Pife.

| Action ID | Action                                                                                                                                                                                 |
|:---------:|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|     0     | Draw a card  |
|     1     | Pick top card from Discard pile |
|     2     | Win |
|     3     | Freeze formed game 1 |
|     4     | Freeze formed game 2 |
|     5     | Freeze formed game 3 |


|   3 - 54  | Discarta carta baralho azul <br>_`3`: A-Espadas, `4`: 2-Espadas, ..., `15`: K-Espadas <br> `16`: A-Copas ... `28`: K-Copas<br>`29`: A-Ouros ... `41`: K-Ouros<br>`42`: A-Paus ... `54`: K-Paus_ |
|  55 - 106 | Discarta carta baralho vermelho <br>_`55`: A-Espadas, `56`: 2-Espadas, ..., `67`: K-Espadas<br>`68`: A-Copas ... `80`: K-Copas<br>`81`: A-Ouros ... `93`: K-Ouros<br>`94`: A-Paus ... `106`: K-Paus_       |

# Recompensas

TBD