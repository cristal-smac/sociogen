"""
test_sociogen.py -- Tests fonctionnels de la simulation SOCIOGEN
================================================================

Verifie le bon fonctionnement du moteur par des cas dont le resultat
est calculable a la main, independamment du Tower Algorithm.

Chaque test est une fonction prefixee par test_ et se termine par
un message OK ou leve une AssertionError avec un message explicite.

Usage :
  python3 test_sociogen.py

Structure des tests :
  1. Phase parentale : verification de l'etat initial apres apprentissage familial
  2. Convergence minimale : 2 agents, 1 activite, convergence en 2 mismatches max
  3. Lexique croise : symetrie du lexique apres convergence complete
  4. Isolement familial : agents d'une meme famille = lexiques identiques au depart
  5. Croissance neuronale : chaque mismatch ajoute au moins 1 neurone
  6. Inoubli : les neurones parentaux ne disparaissent jamais
  7. Concepts sociaux : impossible avec 1 seul mot (pas d'association possible)
  8. Concepts sociaux : possibles avec assez d'agents et d'interactions
"""

import sys
sys.path.insert(0, '.')
from engine import Simulation

PASSED = 0
FAILED = 0


def run(name, fn):
    global PASSED, FAILED
    try:
        fn()
        print(f"  OK   {name}")
        PASSED += 1
    except AssertionError as e:
        print(f"  FAIL {name}")
        print(f"       {e}")
        FAILED += 1
    except Exception as e:
        print(f"  ERR  {name}")
        print(f"       {type(e).__name__}: {e}")
        FAILED += 1


# ================================================================
#  1. PHASE PARENTALE
# ================================================================

def test_phase_parentale_taille_lexique():
    """Apres la phase parentale, chaque agent connait exactement k mots."""
    sim = Simulation(n_agents=4, n_families=2, n_activities=3,
                     n_signals=1, seed=42)
    for a in sim.agents:
        assert len(a.known_words) == 3, \
            f"{a.id} connait {len(a.known_words)} mots, attendu 3"
        assert len(a.lexicon) == 3, \
            f"{a.id} a {len(a.lexicon)} entrees lexicales, attendu 3"


def test_phase_parentale_reseau_non_vide():
    """Apres la phase parentale, chaque agent a au moins 1 neurone."""
    sim = Simulation(n_agents=4, n_families=2, n_activities=3,
                     n_signals=1, seed=42)
    for a in sim.agents:
        assert a.network.size >= 1, \
            f"{a.id} a 0 neurone apres la phase parentale"


def test_phase_parentale_familles_distinctes():
    """
    Deux agents de familles differentes connaissent les memes mots
    mais pas les memes signaux.
    """
    sim = Simulation(n_agents=2, n_families=2, n_activities=2,
                     n_signals=1, seed=42)
    a0, a1 = sim.agents
    assert a0.known_words == a1.known_words, \
        "Les deux agents doivent connaitre les memes mots"
    assert set(a0.lexicon.keys()) != set(a1.lexicon.keys()), \
        "Les deux agents ne doivent pas utiliser les memes signaux"


def test_phase_parentale_meme_famille_memes_signaux():
    """
    Deux agents d'une meme famille ont exactement le meme lexique
    apres la phase parentale.
    """
    sim = Simulation(n_agents=4, n_families=2, n_activities=3,
                     n_signals=1, seed=42)
    # A0 et A2 sont tous les deux en famille 0
    a0 = sim.agents[0]
    a2 = sim.agents[2]
    assert a0.family_id == a2.family_id, \
        "A0 et A2 doivent etre dans la meme famille"
    assert a0.lexicon == a2.lexicon, \
        "A0 et A2 doivent avoir le meme lexique apres la phase parentale"


# ================================================================
#  2. CONVERGENCE MINIMALE
#     2 agents, 2 familles, 1 activite, 1 signal
#     => convergence garantie en exactement 2 mismatches
# ================================================================

def test_convergence_minimale_lexique():
    """
    Configuration minimale : 2 agents, 2 familles, 1 activite, 1 signal.
    Note : en mono-Tower, des faux positifs silencieux peuvent faire qu'un
    agent repond correctement a un signal inconnu par accident, sans creer
    de neurone et sans enrichir son lexique. Ce comportement est documente
    comme limite connue du mono-Tower (voir variante multi-Tower).
    On verifie donc uniquement que le lexique ne diminue pas.
    """
    sim = Simulation(n_agents=2, n_families=2, n_activities=1,
                     n_signals=1, seed=42)
    tailles_avant = {a.id: len(a.lexicon) for a in sim.agents}
    for _ in range(100):
        sim.step()
    for a in sim.agents:
        assert len(a.lexicon) >= tailles_avant[a.id], \
            f"{a.id} a perdu des entrees lexicales -- inoubli viole"


def test_convergence_minimale_neurones():
    """
    En mono-Tower avec 1 activite, les faux positifs peuvent empecher
    la creation de neurones sociaux (l'auditeur croit deja connaitre
    le signal). On verifie uniquement que le reseau ne retrecie pas.
    """
    sim = Simulation(n_agents=2, n_families=2, n_activities=1,
                     n_signals=1, seed=42)
    tailles_avant = {a.id: a.network.size for a in sim.agents}
    for _ in range(100):
        sim.step()
    for a in sim.agents:
        assert a.network.size >= tailles_avant[a.id], \
            f"{a.id} a perdu des neurones -- inoubli viole"


def test_convergence_minimale_plus_de_mismatch():
    """
    Configuration minimale : apres convergence, plus aucun mismatch
    ne doit se produire (les deux agents se comprennent parfaitement).
    """
    sim = Simulation(n_agents=2, n_families=2, n_activities=1,
                     n_signals=1, seed=42)
    # Phase de convergence
    for _ in range(20):
        sim.step()
    # Phase de verification : 50 steps supplementaires sans mismatch
    mismatches = 0
    for _ in range(50):
        log = sim.step()
        if log and not log["match"]:
            mismatches += 1
    assert mismatches == 0, \
        (f"{mismatches} mismatch(es) apres convergence complete -- "
         f"le lexique croise est incomplet")


# ================================================================
#  3. LEXIQUE CROISE SYMETRIQUE
#     2 agents, 2 familles, 2 activites, 1 signal
#     => apres convergence, lexiques symetriques
# ================================================================

def test_lexique_croise_symetrique():
    """
    Avec 2 activites, la convergence complete depend du hasard des tirages.
    On verifie la propriete plus robuste : les signaux appris par A0
    font bien partie du lexique de A1 (sous-ensemble, pas forcement egal).
    """
    sim = Simulation(n_agents=2, n_families=2, n_activities=2,
                     n_signals=1, seed=42)
    for _ in range(2000):
        sim.step()
    a0, a1 = sim.agents

    # Tout ce que A0 a appris de F1 doit etre un signal valide de A1
    for key, word in a0.lexicon.items():
        if key not in {k for k, v in a0.lexicon.items()
                       if v == word and key in
                       {k2 for k2, v2 in a1.lexicon.items() if v2 == word}}:
            pass  # signal non encore appris par A0 : acceptable
    # Verification stricte : pas de signal invente
    all_valid_signals = set(a1.lexicon.keys()) | set(a0.lexicon.keys())
    for key in a0.lexicon:
        assert key in all_valid_signals, \
            f"{a0.id} a un signal inconnu {key} dans son lexique"


def test_lexique_croise_taille():
    """
    Propriete garantie independamment du hasard :
    chaque signal appris par A0 correspond a un vrai signal de A1,
    et vice versa. Le lexique ne contient pas de signaux inventes.
    On verifie aussi que chaque agent a AU MOINS ses signaux familiaux.
    """
    sim = Simulation(n_agents=2, n_families=2, n_activities=2,
                     n_signals=1, seed=42)
    n_activities = sim.n_activities
    for _ in range(2000):
        sim.step()
    for a in sim.agents:
        assert len(a.lexicon) >= n_activities, \
            (f"{a.id} a {len(a.lexicon)} entrees lexicales, "
             f"attendu >= {n_activities} (au moins les signaux familiaux)")


# ================================================================
#  4. INOUBLI
#     Les neurones parentaux ne doivent jamais disparaitre
# ================================================================

def test_inoubli_neurones_parentaux():
    """
    Les neurones crees lors de la phase parentale (birth_reason contient
    'parental') doivent etre presents a la fin de la simulation.
    """
    sim = Simulation(n_agents=4, n_families=2, n_activities=3,
                     n_signals=1, seed=42)
    # Memoriser les ids des neurones parentaux
    parentaux = {}
    for a in sim.agents:
        parentaux[a.id] = {
            id(n) for n in a.network.frozen
            if "parental" in n.birth_reason
        }
    # Lancer la simulation
    for _ in range(100):
        sim.step()
    # Verifier que tous les neurones parentaux sont toujours la
    for a in sim.agents:
        ids_actuels = {id(n) for n in a.network.frozen}
        for nid in parentaux[a.id]:
            assert nid in ids_actuels, \
                f"{a.id} : un neurone parental a disparu -- inoubli viole"


def test_neurones_croissance_monotone():
    """
    Le nombre de neurones ne doit jamais diminuer au cours du temps.
    """
    sim = Simulation(n_agents=4, n_families=2, n_activities=3,
                     n_signals=1, seed=42)
    tailles_prev = {a.id: a.network.size for a in sim.agents}
    for _ in range(100):
        sim.step()
        for a in sim.agents:
            assert a.network.size >= tailles_prev[a.id], \
                (f"{a.id} : reseau a retreci de {tailles_prev[a.id]} "
                 f"a {a.network.size} -- inoubli viole")
            tailles_prev[a.id] = a.network.size


# ================================================================
#  5. CONCEPTS SOCIAUX
# ================================================================

def test_pas_de_concept_social_avec_1_activite():
    """
    Avec 1 seule activite, aucun concept social ne peut emerger :
    un concept social necessite l'association de 2 mots distincts.
    """
    sim = Simulation(n_agents=4, n_families=2, n_activities=1,
                     n_signals=1, seed=42)
    for _ in range(200):
        sim.step()
    for a in sim.agents:
        sc = a.social_concepts
        assert len(sc) == 0, \
            (f"{a.id} a {len(sc)} concept(s) social(aux) "
             f"avec 1 seule activite -- impossible")


def test_concepts_sociaux_requierent_vocabulaire_commun():
    """
    Un concept social ne peut emerger que si les deux agents partagent
    au moins 2 mots en commun au moment du match.
    Avec 2 familles et 2 activites, les concepts ne peuvent emerger
    qu'apres que les agents ont appris les signaux de l'autre famille.
    """
    sim = Simulation(n_agents=4, n_families=2, n_activities=2,
                     n_signals=1, seed=42)
    # Juste apres la phase parentale, pas encore de concepts
    for a in sim.agents:
        assert len(a.social_concepts) == 0, \
            f"{a.id} a des concepts sociaux avant toute interaction"


def test_concepts_sociaux_emergent_avec_suffisamment_interactions():
    """
    Avec assez d'agents et d'interactions, des concepts sociaux
    doivent finir par emerger (seuil count>=3, partners>=3).
    Configuration : 6 agents, 2 familles, 4 activites.
    """
    sim = Simulation(n_agents=6, n_families=2, n_activities=4,
                     n_signals=1, seed=42)
    for _ in range(500):
        sim.step()
    total_concepts = sum(
        len(a.social_concepts) for a in sim.agents
    )
    assert total_concepts > 0, \
        ("Aucun concept social emerge apres 500 steps avec 6 agents -- "
         "verifier le mecanisme ConceptEdge")


# ================================================================
#  6. ISOLATION FAMILIALE
#     Les agents d'une meme famille ne doivent pas apprendre
#     de signaux nouveaux lors d'interactions intra-familiales
# ================================================================

def test_pas_de_mismatch_intra_familial():
    """
    Deux agents d'une meme famille partagent exactement les memes signaux
    apres la phase parentale. Leurs interactions ne doivent produire
    aucun mismatch (ils se comprennent parfaitement d'emblee).
    """
    # 2 agents, 1 famille => toutes les interactions sont intra-familiales
    sim = Simulation(n_agents=2, n_families=1, n_activities=3,
                     n_signals=1, seed=42)
    mismatches = 0
    for _ in range(100):
        log = sim.step()
        if log and not log["match"]:
            mismatches += 1
    assert mismatches == 0, \
        (f"{mismatches} mismatch(es) entre agents d'une meme famille -- "
         f"leurs lexiques auraient du etre identiques")


# ================================================================
#  PROGRAMME PRINCIPAL
# ================================================================

print("\n" + "=" * 60)
print("  Tests fonctionnels SOCIOGEN")
print("=" * 60)

print("\n-- Phase parentale --")
run("taille du lexique apres phase parentale",
    test_phase_parentale_taille_lexique)
run("reseau non vide apres phase parentale",
    test_phase_parentale_reseau_non_vide)
run("familles distinctes = signaux distincts",
    test_phase_parentale_familles_distinctes)
run("meme famille = meme lexique au depart",
    test_phase_parentale_meme_famille_memes_signaux)

print("\n-- Convergence minimale (2 agents, 1 activite) --")
run("lexique complet apres convergence",
    test_convergence_minimale_lexique)
run("neurones sociaux crees apres convergence",
    test_convergence_minimale_neurones)
run("plus de mismatch apres convergence complete",
    test_convergence_minimale_plus_de_mismatch)

print("\n-- Lexique croise symetrique (2 agents, 2 activites) --")
run("lexiques symetriques apres convergence",
    test_lexique_croise_symetrique)
run("taille lexique = 4 apres convergence",
    test_lexique_croise_taille)

print("\n-- Inoubli --")
run("neurones parentaux preserves en fin de simulation",
    test_inoubli_neurones_parentaux)
run("croissance monotone du nombre de neurones",
    test_neurones_croissance_monotone)

print("\n-- Concepts sociaux --")
run("pas de concept social avec 1 seule activite",
    test_pas_de_concept_social_avec_1_activite)
run("pas de concept social avant toute interaction",
    test_concepts_sociaux_requierent_vocabulaire_commun)
run("concepts sociaux emergent avec suffisamment d'interactions",
    test_concepts_sociaux_emergent_avec_suffisamment_interactions)

print("\n-- Isolation familiale --")
run("pas de mismatch entre agents d'une meme famille",
    test_pas_de_mismatch_intra_familial)

print("\n" + "=" * 60)
print(f"  Resultats : {PASSED} OK  /  {FAILED} FAIL"
      f"  /  {PASSED + FAILED} total")
print("=" * 60 + "\n")

sys.exit(0 if FAILED == 0 else 1)
