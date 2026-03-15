# sociogen
> *Des agents identiques, placés dans un environnement social, peuvent-ils développer des intelligences structurellement différentes ?*

SOCIOGEN is a multi-agent simulation in which agents starting from an empty neural network develop structurally heterogeneous cognitive architectures through social interaction alone. Every neuron is traceable to the social event that created it : the network is a biography.

SOCIOGEN est une simulation multi-agents dans laquelle chaque agent démarre avec un réseau neuronal **vide** et **zéro connaissance**. Par le seul jeu d'interactions sociales, des architectures cognitives hétérogènes émergent et chaque différence est entièrement explicable.

---

## Points forts

- **Naissance d'une intelligence à partir de rien** : les agents construisent leur réseau neurone par neurone, strictement à la demande.
- **Divergence cognitive émergente** : deux agents issus de la même famille ne suivent jamais le même chemin développemental.
- **Concepts sociaux spontanés** : des associations comme *pankakes+cinema* émergent sans avoir été programmées, validées collectivement par les interactions répétées.
- **IA entièrement explicable** : chaque neurone porte l'étiquette de l'interaction sociale qui l'a créé. Le réseau est une biographie.

---

## Principe du modèle

Les agents sont regroupés en **familles** : chaque famille encode les mêmes concepts via des signaux distincts, comme des dialectes. Le modèle est le symétrique formel des Talking Heads de Steels : là où Steels fixe les signaux et laisse émerger les mots, SOCIOGEN fixe les mots et fait varier les signaux.

```
Phase parentale   →  chaque agent apprend les signaux de sa famille
                     (1 neurone Tower ajouté par concept)

Phase sociale     →  les agents interagissent par paires au hasard
   mismatch       →  l'auditeur apprend et son Tower grandit
   match          →  un compteur d'association est incrémenté
                     (→ concept social si count ≥ 3 avec 3 partenaires)

Arrêt             →  convergence lexicale ≥ 85 %
                     ou stagnation neuronale (60 ticks sans croissance)
                     ou 500 ticks maximum
```

Le cœur algorithmique est le **Tower Algorithm de Gallant (1990)** : un algorithme constructif booléen à poids entiers qui ajoute des neurones à la demande et ne les oublie jamais (gel irréversible). C'est cette propriété d'inoubli qui rend le réseau lisible comme une biographie sociale.

---

## Structure du dépôt

```
engine.py               — moteur SOCIOGEN (architecture mono-Tower)
engine_multitower.py    — variante multi-Tower (1 Tower par concept)
run.py                  — interface interactive pas-à-pas
trace.py                — biographie cognitive d'un agent (A0)

testTower/
    XOR2.py             — XOR classique (test de sanité, doit donner 2 neurones)
    XOR_N.py            — XOR généralisé à N bits  →  python3 XOR_N.py 5
    Parity_N.py         — parité à N bits           →  python3 Parity_N.py 8
    Multiplexeur_S.py   — multiplexeur S bits de selecteur       →  python3 Multiplexeur_S.py 3
    Animaux.py          — apprentissage par vagues (3 neurones)
```

---

## Démarrage rapide

```bash
# Lancer la simulation interactive
python3 run.py

# Commandes disponibles :
#   Entrée    avancer d'1 step
#   n         avancer de 20 steps
#   s         état de tous les agents
#   a A2      inspecter un agent en détail
#   g         généalogie de tous les agents
#   q         quitter et résumé final
```

```bash
# Lire la biographie cognitive de A0
python3 trace.py         # mono-Tower
python3 trace.py multi   # multi-Tower
```

---

## Benchmarks du Tower Algorithm

```bash
python3 XOR2.py    # doit donner 2 neurones
python3 Parity_N.py 3 # parité n bits
```

```
n= 2 | exemples=    4 | neurones=2
n= 3 | exemples=    8 | neurones=2
n= 4 | exemples=   16 | neurones=3
n= 5 | exemples=   32 | neurones=3
n= 6 | exemples=   64 | neurones=4
n= 8 | exemples=  256 | neurones=5
n=10 | exemples= 1024 | neurones=6
```

Avec `Parity_N` on constate que nombre de neurones croît d'une unité tous les deux bits, tandis que l'espace des exemples double à chaque incrément. La cascade augmentée de Gallant est exponentiellement plus compacte que l'espace qu'elle représente.

---

## Paramètres configurables

```python
sim = Simulation(
    n_agents     = 8,    # nombre d'agents
    n_families   = 4,    # nombre de familles (diversité initiale)
    n_activities = 8,    # taille du vocabulaire partagé
    n_signals    = 2,    # complexité des représentations familiales
    seed         = 42,
)
```

La constante `SIMILAR` dans `engine.py` contrôle la variante de phase parentale :
- `SIMILAR = False` — chaque agent apprend indépendamment (variations intra-familiales authentiques)
- `SIMILAR = True`  — clonage exact : tous les agents d'une même famille démarrent avec le même réseau

---

## Citation

```bibtex
@inproceedings{beaufils2025sociogen,
  author      = {Beaufils, Victor and Mathieu, Philippe},
  title       = {De l'identique au différent : divergence cognitive par
                 interaction sociale dans un système multi-agents},
  booktitle   = {à paraître},
  year        = {2025},
  institution = {Univ. Lille, CNRS, Centrale Lille, UMR 9189, CRIStAL}
}
```

---

## Références

- Gallant, S.I. (1990). *Perceptron-based learning algorithms*. IEEE Transactions on Neural Networks, 1(2), 179–191.
- Steels, L. (1995). *A self-organizing spatial vocabulary*. Artificial Life, 2(3), 319–332.
- Mézard, M. & Nadal, J.-P. (1989). *Learning in feedforward layered networks: the tiling algorithm*. Journal of Physics A, 22(12).
