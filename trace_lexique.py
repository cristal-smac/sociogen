"""
lexique_trace.py -- Evolution du lexique de chaque agent
=========================================================

Affiche le lexique de tous les agents a trois moments cles :
  1. Avant la phase parentale  (reseaux et lexiques vides)
  2. Apres la phase parentale  (lexique familial acquis)
  3. En fin de simulation      (lexique enrichi par les interactions)

Le lexique est la cle de lecture de SOCIOGEN : c'est lui qui dit
"l'agent A2 sait que les signaux [S004, S011] correspondent au mot
crepes". Le Tower ne sert qu'a retrouver ce mot quand on lui presente
ces signaux -- le lexique est la memoire declarative, le Tower est
le mecanisme de reconnaissance.

Usage :
  python3 lexique_trace.py           (simulation complete)
  python3 lexique_trace.py multi     (variante multi-Tower)
"""

import sys
import copy
from engine import Simulation, Agent


# ── Affichage du lexique de tous les agents ───────────

def print_lexiques(sim, titre):
    """
    Affiche le lexique de chaque agent sous forme de tableau :
    colonnes = mots du vocabulaire
    lignes   = agents
    cellule  = signaux connus pour ce mot (ou - si inconnu)

    [F] = signaux appris lors de la phase parentale (familiaux)
    [S] = signaux appris lors de la phase sociale (mismatch)
    [C] = concepts sociaux (affiches en bas, dans le Tower uniquement)
    """
    words = sorted(sim.activities.keys())
    col   = 16   # largeur d'une colonne

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  {titre}  (t={sim.time})")
    print(f"{sep}\n")

    # En-tete : noms des mots
    header = f"  {'Agent':<8} {'Fam':<5}"
    for w in words:
        header += f"  {w:<{col}}"
    print(header)
    print("  " + "-" * (13 + (col + 2) * len(words)))

    for agent in sim.agents:
        # Collecter les signaux par mot
        sigs_par_mot = {}
        for w in words:
            sigs_par_mot[w] = [" ".join(k)
                               for k, v in agent.lexicon.items() if v == w]

        # Nombre de lignes necessaires pour cet agent (au moins 1)
        max_sigs = max((len(s) for s in sigs_par_mot.values()), default=0)
        max_sigs = max(max_sigs, 1)

        for row in range(max_sigs):
            if row == 0:
                prefix = f"  {agent.id:<8} F{agent.family_id:<4}"
            else:
                prefix = f"  {' ':<13}"
            row_line = prefix
            for w in words:
                sigs = sigs_par_mot[w]
                if row < len(sigs):
                    tag  = "[F] " if row == 0 else "[S] "
                    cell = tag + sigs[row]
                    row_line += f"  {cell:<{col}}"
                elif row == 0:
                    row_line += f"  {'-':<{col}}"
                else:
                    row_line += f"  {' ':<{col}}"
            print(row_line)

    print()

    # Resume : taille du lexique et du reseau par agent
    print(f"  {'Agent':<8} {'Fam':<5} {'Mots connus':<14} {'Neurones'}")
    print("  " + "-" * 40)
    for agent in sim.agents:
        print(f"  {agent.id:<8}"
              f" F{agent.family_id:<4}"
              f" {len(agent.known_words):<14}"
              f" {agent.network.size}")
    print()

    # Concepts sociaux [C] : dans le Tower, pas dans le lexique
    concepts = []
    for agent in sim.agents:
        sc = [e for e in agent.associations.values() if e.is_social_concept]
        if sc:
            concepts.append((agent, sc))

    if concepts:
        print(f"  Concepts sociaux [C] :")
        print("  " + "-" * 40)
        for agent, sc_list in concepts:
            for e in sc_list:
                print(f"  {agent.id:<8} F{agent.family_id}"
                      f"  [C] {e.concept1} + {e.concept2}"
                      f"  (x{e.count}, {len(e.partners)} partenaires)")
        print()
    else:
        print(f"  Concepts sociaux [C] : aucun\n")


# ── Programme principal ───────────────────────────────

multitower = len(sys.argv) > 1 and sys.argv[1] == "multi"

if multitower:
    from engine_multitower import Simulation as SimMulti
    SimClass = SimMulti
    print("\nMode : multi-Tower")
else:
    SimClass = Simulation
    print("\nMode : mono-Tower")

# ── Moment 1 : avant la phase parentale ───────────────
# La phase parentale est appelee dans __init__, donc on
# reconstruit des agents vides pour montrer l'etat initial.

sim = SimClass(n_agents=8, n_families=4, n_activities=8,
               n_signals=2, seed=42)

agents_vides = [
    Agent(f"A{i}", i % sim.n_families,
          sim.all_signals, sim.signal_to_idx)
    for i in range(sim.n_agents)
]
sim_vide        = copy.copy(sim)
sim_vide.agents = agents_vides
sim_vide.time   = 0
print_lexiques(sim_vide, "AVANT LA PHASE PARENTALE -- lexiques vides")

# ── Moment 2 : apres la phase parentale ───────────────
sim_apres        = copy.copy(sim)
sim_apres.agents = sim.agents
sim_apres.time   = 0
print_lexiques(sim_apres, "APRES LA PHASE PARENTALE -- lexique familial acquis")

# ── Moment 3 : fin de simulation ──────────────────────
from run import StopConditions

stop        = StopConditions()
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

print_lexiques(sim, f"FIN DE SIMULATION -- {stop_reason}")
