"""
lexique_trace.py — Évolution du lexique de chaque agent
========================================================

Affiche le lexique de tous les agents à trois moments clés :
  1. Avant la phase parentale  (réseaux et lexiques vides)
  2. Après la phase parentale  (lexique familial acquis)
  3. En fin de simulation      (lexique enrichi par les interactions)

Le lexique est la clé de lecture de SOCIOGEN : c'est lui qui dit
"l'agent A2 sait que les signaux [S004, S011] correspondent au mot
crêpes". Le Tower ne sert qu'à retrouver ce mot quand on lui présente
ces signaux — le lexique est la mémoire déclarative, le Tower est
le mécanisme de reconnaissance.

Usage :
  python3 lexique_trace.py           (simulation complète)
  python3 lexique_trace.py multi     (variante multi-Tower)
"""

import sys
from engine import Simulation

# ── Couleurs ANSI ─────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
WHITE  = "\033[97m"

FAM_COLORS = [
    "\033[94m",   # bleu
    "\033[95m",   # magenta
    "\033[92m",   # vert
    "\033[93m",   # jaune
    "\033[96m",   # cyan
    "\033[91m",   # rouge
    "\033[97m",   # blanc
    "\033[33m",   # orange
]

def clr(agent):
    return FAM_COLORS[agent.family_id % len(FAM_COLORS)]


# ── Affichage du lexique de tous les agents ───────────

def print_lexiques(sim, titre: str):
    """
    Affiche le lexique de chaque agent sous forme de tableau :
    colonnes = mots du vocabulaire
    lignes   = agents
    cellule  = signaux connus pour ce mot (ou vide si inconnu)
    """
    words = sorted(sim.activities.keys())
    col   = 16   # largeur d'une colonne

    print(f"\n{'═'*72}")
    print(f"  {BOLD}{CYAN}{titre}{RESET}  {DIM}(t={sim.time}){RESET}")
    print(f"{'═'*72}\n")

    # En-tête : noms des mots
    header = f"  {'Agent':<8} {'Fam':<5}"
    for w in words:
        header += f"  {w:<{col}}"
    print(BOLD + header + RESET)
    print("  " + "─" * (13 + (col + 2) * len(words)))

    for agent in sim.agents:
        c    = clr(agent)
        line = f"  {c}{BOLD}{agent.id:<8}{RESET} {DIM}F{agent.family_id:<4}{RESET}"
        # Collecter toutes les entrées de signaux par mot
        sigs_par_mot = {}
        for w in words:
            sigs_par_mot[w] = [" ".join(k)
                               for k, v in agent.lexicon.items() if v == w]

        # Nombre de lignes nécessaires pour cet agent
        max_sigs = max((len(s) for s in sigs_par_mot.values()), default=1)

        for row in range(max_sigs):
            if row == 0:
                prefix = f"  {c}{BOLD}{agent.id:<8}{RESET} {DIM}F{agent.family_id:<4}{RESET}"
            else:
                prefix = f"  {' ':<13}"
            row_line = prefix
            for w in words:
                sigs = sigs_par_mot[w]
                if row < len(sigs):
                    tag = "[F] " if row == 0 else "[+] "
                    cell = tag + sigs[row]
                    row_line += f"  {c}{cell:<{col}}{RESET}"
                elif row == 0:
                    row_line += f"  {DIM}{'—':<{col}}{RESET}"
                else:
                    row_line += f"  {' ':<{col}}"
            print(row_line)

    print()

    # Résumé : taille du lexique et du réseau par agent
    print(f"  {'Agent':<8} {'Fam':<5} {'Mots connus':<14} {'Neurones'}")
    print("  " + "─" * 40)
    for agent in sim.agents:
        c = clr(agent)
        print(f"  {c}{BOLD}{agent.id:<8}{RESET}"
              f" {DIM}F{agent.family_id:<4}{RESET}"
              f" {len(agent.known_words):<14}"
              f" {agent.network.size}")
    print()


# ── Programme principal ───────────────────────────────

multitower = len(sys.argv) > 1 and sys.argv[1] == "multi"

if multitower:
    from engine_multitower import Simulation as SimMulti
    SimClass = SimMulti
    print(f"\n{BOLD}Mode : multi-Tower{RESET}")
else:
    SimClass = Simulation
    print(f"\n{BOLD}Mode : mono-Tower{RESET}")

# ── Moment 1 : avant la phase parentale ───────────────
# On instancie sans déclencher la phase parentale pour
# montrer l'état initial vide. On le fait en réinitialisant
# manuellement les agents après construction.

sim = SimClass(n_agents=8, n_families=4, n_activities=8,
               n_signals=2, seed=42)

# Reconstruire un état vide pour l'affichage initial
# (la phase parentale est appelée dans __init__, donc on
# reconstruit les agents à la main pour l'état t=0)
import copy
agents_apres_parentale = sim.agents

# Créer une version "vide" pour l'affichage avant phase parentale
from engine import Agent, TowerNetwork
all_signals    = sim.all_signals
signal_to_idx  = sim.signal_to_idx
activity_names = list(sim.activities.keys())

agents_vides = [
    Agent(f"A{i}", i % sim.n_families, all_signals, signal_to_idx)
    for i in range(sim.n_agents)
]
sim_vide = copy.copy(sim)
sim_vide.agents = agents_vides
sim_vide.time   = 0
print_lexiques(sim_vide, "AVANT LA PHASE PARENTALE — lexiques vides")

# ── Moment 2 : après la phase parentale ───────────────
sim_apres = copy.copy(sim)
sim_apres.agents = agents_apres_parentale
sim_apres.time   = 0
print_lexiques(sim_apres, "APRÈS LA PHASE PARENTALE — lexique familial acquis")

# ── Moment 3 : fin de simulation ──────────────────────
from run import StopConditions

stop = StopConditions()
stop_reason = ""
while True:
    log = sim.step()
    if log is None:
        stop_reason = "lexique vide"
        break
    stop.update(log, sim)
    done, reason = stop.check(sim)
    if done:
        stop_reason = reason
        break

print_lexiques(sim, f"FIN DE SIMULATION — {stop_reason}")
