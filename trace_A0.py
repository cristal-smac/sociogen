"""
trace.py - Biographie cognitive de A0
======================================
Trace l'evolution du reseau de A0 :
  1. Apres phase parentale
  2. Apres chaque interaction ou A0 est recepteur
  3. Jusqu'a la creation du premier concept social

Usage :
  python3 trace.py            (mono-Tower)
  python3 trace.py multi      (multi-Tower)
"""

import sys
import numpy as np

if len(sys.argv) > 1 and sys.argv[1] == "multi":
    sys.path.insert(0, '..')
    from engine_multitower import Simulation
    MODE = "multi-Tower"
else:
    sys.path.insert(0, '..')
    from engine import Simulation
    MODE = "mono-Tower"

SEP  = "-" * 70
SEP2 = "=" * 70

def neuron_icon(birth_reason):
    if "parental"       in birth_reason: return "[P]"
    if "concept social" in birth_reason: return "[*]"
    if "appris de"      in birth_reason: return "[S]"
    return "[ ]"

def print_network_state(agent, step_label):
    print("")
    print(SEP)
    print("  %s  --  %s  (%d neurone(s))" % (step_label, agent.id, agent.network.size))
    print(SEP)
    if agent.network.size == 0:
        print("  (reseau vide)")
        return
    for i, n in enumerate(agent.network.frozen):
        icon = neuron_icon(n.birth_reason)
        print("  Neurone #%02d  %s  [%s]" % (i, icon, n.label))
        print("    Raison : %s" % n.birth_reason)

def find_index_by_id(frozen_list, neuron):
    """Trouve l'index d'un neurone par son identite objet."""
    for i, n in enumerate(frozen_list):
        if id(n) == id(neuron):
            return i
    return -1

def main():
    print("")
    print(SEP2)
    print("  BIOGRAPHIE COGNITIVE DE A0  (%s)" % MODE)
    print(SEP2)

    sim = Simulation(
        n_agents=8, n_families=4, n_activities=8, n_signals=2, seed=42
    )
    target = sim.agents[0]

    print_network_state(target, "APRES PHASE PARENTALE")
    print("")
    print("  Lexique : %s" % sorted(target.known_words))

    print("")
    print(SEP2)
    print("  PHASE SOCIALE -- jusqu'au premier concept social de A0")
    print(SEP2)

    found = False
    step  = 0

    while not found and step < 2000:
        # Snapshot des identites AVANT le step
        ids_avant = set(id(n) for n in target.network.frozen)

        step += 1
        log = sim.step()
        if log is None:
            continue

        listener_id = log["listener"]
        speaker_id  = log["speaker"]
        word        = log["speaker_word"]
        signal      = log["signal"]
        matched     = log["match"]

        # Neurones reellement nouveaux (par identite objet)
        nouveaux = [n for n in target.network.frozen if id(n) not in ids_avant]

        if listener_id == target.id and not matched:
            print("")
            print(SEP)
            print("  Step %4d  --  A0 recoit de %s" % (step, speaker_id))
            print(SEP)
            print("  Signal  : %s" % signal)
            print("  Mot     : %s" % word)
            if nouveaux:
                for n in nouveaux:
                    idx = find_index_by_id(target.network.frozen, n)
                    print("  -> Neurone #%02d cree :" % idx)
                    print("    Label  : %s" % n.label)
                    print("    Raison : %s" % n.birth_reason)
            else:
                print("  -> Deja connu (aucun neurone cree)")

        elif listener_id == target.id and matched:
            assoc = log.get("association_formed")
            if assoc:
                key  = tuple(sorted(assoc))
                edge = target.associations.get(key)
                if edge:
                    print("  Step %4d  --  A0 reconnait '%s' (de %s)"
                          "  ->  %s+%s (x%d, %d partenaire(s))" % (
                          step, word, speaker_id,
                          key[0], key[1], edge.count, len(edge.partners)))

        # Concept social emerge chez A0 ?
        if target.social_concepts and not found:
            found = True
            sc    = target.social_concepts[0]
            print("")
            print(SEP2)
            print("  Step %4d  --  [*] CONCEPT SOCIAL EMERGE chez A0 !" % step)
            print(SEP2)
            print("  Concept     : %s + %s" % (sc.concept1, sc.concept2))
            print("  Occurrences : %d" % sc.count)
            print("  Partenaires : %s" % sc.partners)

    if not found:
        print("")
        print("  Aucun concept social apres 2000 steps.")

    print_network_state(target, "TOUR FINALE")

    print("")
    print(SEP)
    print("  RESUME")
    print(SEP)
    counts = {"parental": 0, "apprentissage social": 0, "concept social": 0}
    for n in target.network.frozen:
        r = n.birth_reason
        if   "parental"       in r: counts["parental"]             += 1
        elif "concept social" in r: counts["concept social"]       += 1
        elif "appris de"      in r: counts["apprentissage social"] += 1
    for k, v in counts.items():
        print("  %-25s : %d neurone(s)" % (k, v))
    print("  %-25s : %d neurone(s)" % ("total", target.network.size))
    print("")

if __name__ == "__main__":
    main()
