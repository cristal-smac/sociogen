"""
run.py — Simulation multi-agents en mode texte
Usage: python3 run.py

Commandes pendant la simulation :
  Entrée    → avancer d'1 step
  n         → avancer de 20 steps
  a <id>    → inspecter un agent en détail  (ex: a A2)
  s         → état de tous les agents
  g         → généalogie de tous les agents
  c         → état des critères d'arrêt
  h         → aide
  q         → quitter et afficher le résumé final
"""

import os
from engine import Simulation, ACTIVITY_NAMES

# ── Couleurs ANSI ──────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
WHITE  = "\033[97m"
ORANGE = "\033[33m"

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

def bar(value, max_val, width=24, color=""):
    filled = int(value / max_val * width) if max_val > 0 else 0
    return color + "█" * filled + DIM + "░" * (width - filled) + RESET


# ══════════════════════════════════════════════════════
#  CRITÈRES D'ARRÊT
# ══════════════════════════════════════════════════════

class StopConditions:
    """
    Trois critères d'arrêt scientifiquement pertinents :
    1. Convergence lexicale inter-familles
    2. Stagnation neuronale
    3. Limite maximale de steps
    """

    def __init__(self, max_steps=500, convergence_threshold=0.85,
                 convergence_window=30, stagnation_steps=60, min_steps=40):
        self.max_steps             = max_steps
        self.convergence_threshold = convergence_threshold
        self.convergence_window    = convergence_window
        self.stagnation_steps      = stagnation_steps
        self.min_steps             = min_steps
        self._match_window         = []
        self._last_neuron_at       = 0
        self._total_neurons_prev   = 0

    def update(self, log, sim):
        if log is None:
            return
        a1 = next((a for a in sim.agents if a.id == log['speaker']), None)
        a2 = next((a for a in sim.agents if a.id == log['listener']), None)
        if a1 and a2 and a1.family_id != a2.family_id:
            self._match_window.append(1 if log['match'] else 0)
            if len(self._match_window) > self.convergence_window:
                self._match_window.pop(0)
        total = sum(a.network.size for a in sim.agents)
        if total > self._total_neurons_prev:
            self._last_neuron_at     = sim.time
            self._total_neurons_prev = total

    def check(self, sim):
        t = sim.time
        if t < self.min_steps:
            return False, ""
        if t >= self.max_steps:
            return True, f"limite maximale atteinte ({self.max_steps} steps)"
        if len(self._match_window) >= self.convergence_window:
            rate = sum(self._match_window) / len(self._match_window)
            if rate >= self.convergence_threshold:
                return True, (f"convergence lexicale inter-familles "
                              f"({rate*100:.0f}% >= {self.convergence_threshold*100:.0f}%)")
        stag = t - self._last_neuron_at
        if stag >= self.stagnation_steps:
            return True, f"stagnation neuronale ({stag} steps sans nouveau neurone)"
        return False, ""

    def status(self, sim):
        t    = sim.time
        rate = (f"{sum(self._match_window)/len(self._match_window)*100:.0f}%"
                f"/{self.convergence_threshold*100:.0f}%"
                if self._match_window else "—")
        stag = t - self._last_neuron_at
        return (f"match inter-fam={rate}  "
                f"stagnation={stag}/{self.stagnation_steps}  "
                f"steps={t}/{self.max_steps}")


# ══════════════════════════════════════════════════════
#  AFFICHAGE
# ══════════════════════════════════════════════════════

def print_agents(sim):
    max_n = max(a.network.size for a in sim.agents) if sim.agents else 1
    print(f"\n{BOLD}{WHITE}{'─'*72}{RESET}")
    print(f"{BOLD}{CYAN}  AGENTS{RESET}  {DIM}t={sim.time}{RESET}"
          f"   interactions={sum(a.interaction_count for a in sim.agents)}"
          f"   concepts sociaux={sum(len(a.social_concepts) for a in sim.agents)}")
    print(f"{BOLD}{WHITE}{'─'*72}{RESET}\n")

    for a in sim.agents:
        c     = clr(a)
        stars = "★" * len(a.social_concepts)
        print(f"  {c}{BOLD}{a.id}{RESET}  {DIM}Famille {a.family_id}{RESET}"
              + (f"  {YELLOW}{stars}{RESET}" if stars else ""))

        print(f"    Neurones  {bar(a.network.size, max_n, color=c)}"
              f"  {c}{BOLD}{a.network.size:2d}{RESET}"
              f"  {DIM}(+{a.network.size - 1} depuis init){RESET}")

        print(f"    Lexique   {bar(len(a.known_words), sim.n_activities, color=DIM)}"
              f"  {DIM}{len(a.known_words):2d}/{sim.n_activities} words{RESET}")

        if a.interaction_count > 0:
            rate  = a.successful_matches / a.interaction_count * 100
            color_rate = GREEN if rate >= 50 else (YELLOW if rate >= 25 else RED)
            print(f"    Succès    {color_rate}{rate:.0f}%{RESET}"
                  f"  {DIM}({a.successful_matches}/{a.interaction_count}){RESET}")

        if a.social_concepts:
            sc = "  ".join(
                f"{YELLOW}{e.concept1}+{e.concept2}{RESET}"
                f"{DIM}(×{e.count}, {len(e.partners)}👥){RESET}"
                for e in a.social_concepts
            )
            print(f"    {YELLOW}★ Concepts:{RESET} {sc}")
        print()


def print_agent_detail(agent, sim):
    c = clr(agent)
    print(f"\n{BOLD}{WHITE}{'─'*72}{RESET}")
    print(f"{BOLD}  Inspection — {c}{agent.id}{RESET}"
          f"  {DIM}Famille {agent.family_id}{RESET}")
    print(f"{BOLD}{WHITE}{'─'*72}{RESET}\n")

    # Réseau Tower
    print(f"  {BOLD}Réseau Tower ({agent.network.size} neurones){RESET}")
    for i, n in enumerate(agent.network.frozen):
        marker = f"{ORANGE}►{RESET}" if i == agent.network.size - 1 else f"{DIM}·{RESET}"
        print(f"    {marker} {DIM}#{i:02d}{RESET}"
              f"  {c}{n.label:<20}{RESET}"
              f"  {DIM}{n.birth_reason}{RESET}")

    # Lexique
    print(f"\n  {BOLD}Lexique ({len(agent.lexicon)} entrées){RESET}")
    for key, word in sorted(agent.lexicon.items(), key=lambda x: x[1]):
        sigs = " ".join(key)
        print(f"    {DIM}{sigs:<30}{RESET}  {c}{word}{RESET}")

    # Associations
    print(f"\n  {BOLD}Associations ({len(agent.associations)}){RESET}")
    sorted_assoc = sorted(agent.associations.values(),
                          key=lambda e: e.count, reverse=True)
    if sorted_assoc:
        for e in sorted_assoc:
            tag = f"  {GREEN}{BOLD}★ CONCEPT SOCIAL{RESET}" \
                  if e.is_social_concept else ""
            print(f"    {YELLOW}{e.concept1:<15}{RESET}"
                  f" + {YELLOW}{e.concept2:<15}{RESET}"
                  f"  ×{e.count}  {len(e.partners)} partenaires{tag}")
    else:
        print(f"    {DIM}Aucune association encore{RESET}")

    # Généalogie
    print()
    agent.print_genealogy()
    print()


def print_interaction(log, sim):
    a1 = next((a for a in sim.agents if a.id == log['speaker']), None)
    a2 = next((a for a in sim.agents if a.id == log['listener']), None)
    if not a1 or not a2:
        return
    c1, c2 = clr(a1), clr(a2)
    cross   = f"{DIM}⇄{RESET}" if a1.family_id != a2.family_id else f"{DIM}↔{RESET}"
    sigs    = " ".join(log.get('signal', []))

    if log['match']:
        result = f"{GREEN}{BOLD}✓ MATCH{RESET}"
        detail = ""
        if log.get('association_formed'):
            w1, w2 = log['association_formed']
            detail = f"  {DIM}lien: {w1}+{w2}{RESET}"
    elif log.get('network_growth'):
        result = f"{YELLOW}{BOLD}+ APPRIS{RESET} {DIM}+1 neurone{RESET}"
        detail = f"  {CYAN}→ {log.get('new_concept_learned','?')}{RESET}"
    else:
        result = f"{RED}✗{RESET}"
        detail = f"  {DIM}→ {log.get('new_concept_learned','?')}{RESET}"

    print(f"  {c1}{BOLD}{log['speaker']}{RESET}"
          f"{cross}"
          f"{c2}{BOLD}{log['listener']}{RESET}"
          f"  {DIM}{sigs}{RESET}"
          f"  {DIM}\"{log['speaker_word']}\"{RESET}"
          f"  {result}{detail}")


def print_stop_reason(reason):
    print(f"\n{BOLD}{YELLOW}{'━'*72}{RESET}")
    print(f"{BOLD}{YELLOW}  ⏹  ARRÊT : {reason}{RESET}")
    print(f"{BOLD}{YELLOW}{'━'*72}{RESET}\n")


def print_summary(sim, stop_reason=""):
    print(f"\n{BOLD}{WHITE}{'═'*72}{RESET}")
    print(f"{BOLD}{CYAN}  RÉSUMÉ FINAL — t={sim.time}{RESET}")
    if stop_reason:
        print(f"  {DIM}Arrêt : {stop_reason}{RESET}")
    print(f"{BOLD}{WHITE}{'═'*72}{RESET}\n")

    sizes  = [a.network.size for a in sim.agents]
    mean   = sum(sizes) / len(sizes)
    std    = (sum((s - mean)**2 for s in sizes) / len(sizes)) ** 0.5
    hetero = std / mean * 100 if mean > 0 else 0

    print(f"  {BOLD}Paramètres{RESET}")
    print(f"    agents={sim.n_agents}  familles={sim.n_families}"
          f"  activités={sim.n_activities}  signal/act={sim.n_signals}\n")

    print(f"  {BOLD}Hétérogénéité des réseaux{RESET}")
    print(f"    Min={min(sizes)}  Max={max(sizes)}"
          f"  Moy={mean:.1f}  σ={std:.1f}"
          f"  → {YELLOW}{hetero:.0f}% d'hétérogénéité{RESET}\n")

    print(f"  {BOLD}Classement par taille de réseau{RESET}")
    for rank, a in enumerate(
            sorted(sim.agents, key=lambda a: a.network.size, reverse=True), 1):
        c = clr(a)
        b = bar(a.network.size, max(sizes), width=22, color=c)
        print(f"    {rank}. {c}{BOLD}{a.id}{RESET}  {b}"
              f"  {c}{a.network.size:2d}{RESET} neurones"
              f"  {DIM}Famille {a.family_id}{RESET}")
    print()

    print(f"  {BOLD}Concepts sociaux émergents{RESET}")
    found = False
    for a in sim.agents:
        for e in a.social_concepts:
            found = True
            print(f"    {clr(a)}{a.id}{RESET}"
                  f"  {YELLOW}{e.concept1} + {e.concept2}{RESET}"
                  f"  {DIM}×{e.count}, {len(e.partners)} partenaires{RESET}")
    if not found:
        print(f"    {DIM}Aucun concept social émergé{RESET}")
    print()

    print(f"  {BOLD}Tower log — agent A0{RESET}"
          f"  {DIM}(biographie cognitive complète){RESET}")
    for i, n in enumerate(sim.agents[0].network.frozen):
        print(f"    {DIM}#{i:02d}{RESET}"
              f"  {CYAN}{n.label:<20}{RESET}"
              f"  {DIM}{n.birth_reason}{RESET}")

    print(f"\n{BOLD}{WHITE}{'═'*72}{RESET}\n")

    # Statistiques globales
    sim.stats()


def print_help(sim):
    print(f"\n  {BOLD}Commandes disponibles{RESET}"
          f"  {DIM}(simulation : {sim.n_agents} agents,"
          f" {sim.n_families} familles,"
          f" {sim.n_activities} activités,"
          f" {sim.n_signals} signal){RESET}")
    print(f"  {CYAN}Entrée{RESET}    avancer d'1 step")
    print(f"  {CYAN}n{RESET}         avancer de 20 steps")
    print(f"  {CYAN}a <id>{RESET}    inspecter un agent  (ex: {DIM}a A2{RESET})")
    print(f"  {CYAN}s{RESET}         état de tous les agents")
    print(f"  {CYAN}g{RESET}         généalogie de tous les agents")
    print(f"  {CYAN}c{RESET}         état des critères d'arrêt")
    print(f"  {CYAN}h{RESET}         cette aide")
    print(f"  {CYAN}q{RESET}         quitter et résumé final\n")


# ══════════════════════════════════════════════════════
#  MAIN — paramètres modifiables ici
# ══════════════════════════════════════════════════════

def main():
    # ── Paramètres de la simulation ────────────────────
    N_AGENTS     = 8
    N_FAMILIES   = 4
    N_ACTIVITIES = 8
    N_SIGNALS    = 2
    SEED         = 42

    # ── Critères d'arrêt ───────────────────────────────
    stop = StopConditions(
        max_steps             = 500,
        convergence_threshold = 0.85,
        convergence_window    = 30,
        stagnation_steps      = 60,
        min_steps             = 40,
    )

    # ── Initialisation ─────────────────────────────────
    sim = Simulation(
        n_agents     = N_AGENTS,
        n_families   = N_FAMILIES,
        n_activities = N_ACTIVITIES,
        n_signals    = N_SIGNALS,
        seed         = SEED,
    )

    print(f"\n{BOLD}{CYAN}  SOCIOGEN — Émergence Cognitive Multi-Agents{RESET}")
    sim.describe()

    print(f"  {BOLD}Critères d'arrêt automatique{RESET}")
    print(f"    {YELLOW}•{RESET} Convergence  : match inter-familles"
          f" >= {stop.convergence_threshold*100:.0f}%"
          f" sur {stop.convergence_window} interactions")
    print(f"    {YELLOW}•{RESET} Stagnation   :"
          f" aucun neurone ajouté depuis {stop.stagnation_steps} steps")
    print(f"    {YELLOW}•{RESET} Limite max   : {stop.max_steps} steps\n")

    print(f"  {BOLD}Phase parentale terminée.{RESET}")
    print(f"  {DIM}Chaque agent a appris le lexique de sa famille.{RESET}")
    print_agents(sim)
    print_help(sim)

    agent_ids   = [a.id for a in sim.agents]
    stop_reason = ""

    while True:
        should_stop, reason = stop.check(sim)
        if should_stop:
            print_stop_reason(reason)
            stop_reason = reason
            break

        try:
            raw = input(f"  {DIM}[t={sim.time}] > {RESET}").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            stop_reason = "interruption utilisateur"
            break

        # ── Avancer d'1 step ──────────────────────────
        if raw == "":
            log = sim.step()
            stop.update(log, sim)
            if log:
                print_interaction(log, sim)

        # ── Avancer de 20 steps ───────────────────────
        elif raw == "n":
            print(f"  {DIM}→ 20 steps...{RESET}")
            for _ in range(20):
                should_stop, reason = stop.check(sim)
                if should_stop:
                    print_stop_reason(reason)
                    stop_reason = reason
                    break
                log = sim.step()
                stop.update(log, sim)
                if log:
                    print_interaction(log, sim)
            if stop_reason:
                break
            print_agents(sim)

        # ── Inspecter un agent ────────────────────────
        elif raw.startswith("a"):
            parts = raw.split()
            if len(parts) == 2:
                aid   = parts[1].upper()
                agent = next((a for a in sim.agents if a.id == aid), None)
                if agent:
                    print_agent_detail(agent, sim)
                else:
                    print(f"  {RED}Agent '{aid}' introuvable."
                          f" Disponibles : {', '.join(agent_ids)}{RESET}")
            else:
                print(f"  {DIM}Usage : a <id>   ex: a A2{RESET}")

        # ── État de tous les agents ───────────────────
        elif raw == "s":
            print_agents(sim)

        # ── Généalogie ────────────────────────────────
        elif raw == "g":
            for a in sim.agents:
                a.print_genealogy()
            print()

        # ── Critères d'arrêt ──────────────────────────
        elif raw == "c":
            print(f"\n  {BOLD}Critères d'arrêt{RESET}")
            print(f"  {DIM}{stop.status(sim)}{RESET}\n")

        # ── Aide ──────────────────────────────────────
        elif raw == "h":
            print_help(sim)

        # ── Quitter ───────────────────────────────────
        elif raw == "q":
            stop_reason = "arrêt manuel"
            break

        else:
            print(f"  {DIM}Commande inconnue. Tapez 'h' pour l'aide.{RESET}")

    print_summary(sim, stop_reason)


if __name__ == "__main__":
    main()
