"""
engine.py — Moteur SOCIOGEN (architecture multi-Tower)
=======================================================
Tower Algorithm (Gallant 1990) + Talking Heads (Steels 1995)

Architecture :
  - 1 Tower par activité (niveau 1) : reconnaît les signaux -> mot
  - 1 Tower de second niveau (niveau 2) : reconnaît les concepts sociaux
    en prenant en entrée les sorties des Towers de niveau 1

Paramètres configurables :
  n_agents     : nombre d'agents
  n_families   : nombre de familles
  n_activities : nombre d'activités (max 16)
  n_signals    : nombre de signaux par activité par famille
  seed         : graine aléatoire

Usage minimal :
  from engine import Simulation
  sim = Simulation(n_agents=8, n_families=4, n_activities=8, n_signals=2)
  log = sim.step()

Faire exec(open('TestTowerFromEngine.py').read())
pour pouvoir travailler en interactif une fois l'exp terminée
"""

import random
import copy
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────
#  CONSTANTES GLOBALES
# ─────────────────────────────────────────

SIMILAR = False
# Si True, les agents d'une même famille sont des clones exacts
# (mêmes poids, même réseau) après la phase parentale.
# Si False, chaque agent apprend indépendamment (légères variations
# dues à l'ordre de présentation des exemples via Pocket).


# ─────────────────────────────────────────
#  GÉNÉRATION DYNAMIQUE DU VOCABULAIRE
# ─────────────────────────────────────────

#ACTIVITY_NAMES = [
#    "crêpes", "football", "lecture", "cuisine",
#    "peinture", "musique", "jardinage", "cinéma",
#    "natation", "dessin", "danse", "théâtre",
#    "cyclisme", "poterie", "photographie", "yoga",
#]


ACTIVITY_NAMES = [
    "pancakes", "football", "reading", "cooking",
    "painting", "music", "gardening", "cinema",
    "swimming", "drawing", "dancing", "theater",
    "cycling", "pottery", "photography", "yoga",
]


def build_activities(n_activities: int, n_signals: int,
                     n_families: int, seed: int = 0):
    """
    Construit dynamiquement le vocabulaire de la simulation.

    Chaque activité possède n_families variantes, chacune composée
    de n_signals signaux uniques dans l'ensemble du vocabulaire.
    Les signaux sont nommés S000, S001, ... pour garantir l'unicité
    et éviter toute ambiguïté dans les vecteurs d'entrée du réseau.

    Retourne : (activities, all_signals, signal_to_idx)
    """
    rng         = random.Random(seed)
    n_total_sig = n_activities * n_families * n_signals
    all_signals = [f"S{i:03d}" for i in range(n_total_sig)]
    rng.shuffle(all_signals)

    activities = {}
    idx = 0
    for i in range(n_activities):
        name     = ACTIVITY_NAMES[i % len(ACTIVITY_NAMES)]
        variants = []
        for _ in range(n_families):
            variants.append(all_signals[idx: idx + n_signals][:])
            idx += n_signals
        activities[name] = variants

    used       = all_signals[:idx]
    sig_to_idx = {s: i for i, s in enumerate(used)}
    return activities, used, sig_to_idx


# ─────────────────────────────────────────
#  NEURONE BOOLÉEN
# ─────────────────────────────────────────

@dataclass
class BooleanNeuron:
    """
    Neurone à seuil booléen (Gallant 1990).
    Sortie : 1 si Sigma(w_j · x_j) >= threshold, 0 sinon.
    Chaque neurone mémorise son label (concept appris)
    et sa birth_reason (événement social qui l'a créé).
    weights est un np.ndarray pour les calculs vectoriels.
    """
    weights     : np.ndarray
    threshold   : float
    input_size  : int
    label       : str = ""
    birth_reason: str = ""

    def activate(self, inputs) -> int:
        """inputs peut être list ou np.ndarray."""
        return 1 if np.dot(self.weights, inputs) >= self.threshold else 0

    def activate_batch(self, X: np.ndarray) -> np.ndarray:
        """Activation vectorisée sur un batch (n_examples, n_inputs)."""
        return (X @ self.weights >= self.threshold).astype(np.int8)

    def to_dict(self):
        return {
            "weights"     : self.weights.tolist(),
            "threshold"   : self.threshold,
            "label"       : self.label,
            "birth_reason": self.birth_reason,
        }


# ─────────────────────────────────────────
#  TOWER NETWORK (Gallant 1990)
# ─────────────────────────────────────────

class TowerNetwork:
    """
    Implémentation fidèle du Tower Algorithm (Gallant 1990, Fig. 11).

    Architecture : chaque neurone reçoit les p entrées originales
    PLUS la sortie du neurone immédiatement en dessous.
    Chaque neurone a donc exactement p+1 poids (+ biais).

    Utilisation :
      - tower_fit(examples, label, reason) : entraîne la tour sur
        tous les exemples d'un coup (usage autonome, ex: XOR)
      - tower_fit_incremental([(inputs, expected)], label, reason) :
        ajoute un ou plusieurs exemples à la mémoire et crée un neurone
        si nécessaire (usage incrémental, ex: apprentissage social SOCIOGEN)
    """

    PERCEPTRON_ITERATIONS = 2000
    LEARNING_RATE         = 1

    def __init__(self, input_size: int):
        self.input_size = input_size
        self.frozen     : list[BooleanNeuron] = []
        self.memory     : list[tuple]         = []
        self._X_cache   : np.ndarray | None   = None  # cache matrice augmentée
        self._y_cache   : np.ndarray | None   = None

    # ── Cache numpy ───────────────────────────────────

    def _build_cache(self):
        """
        Construit la matrice augmentée X (n_examples, p+k) et
        le vecteur y (n_examples,) à partir de self.memory.
        Appelé une seule fois par neurone dans _pocket_train.
        """
        n = len(self.memory)
        p = self.input_size
        k = len(self.frozen)
        X = np.zeros((n, p + k), dtype=np.float64)
        y = np.zeros(n, dtype=np.int8)
        for i, (inp, exp) in enumerate(self.memory):
            aug = self._augment(inp)
            X[i, :len(aug)] = aug
            y[i] = exp
        self._X_cache = X
        self._y_cache = y

    # ── Augmentation ──────────────────────────────────

    def _augment(self, inputs) -> np.ndarray:
        """
        Reconstruit le vecteur d'entrée en chaîne :
        entrées originales + sortie de chaque neurone gelé.
        Fidèle à Gallant : chaque neurone voit p entrées + 1 sortie.
        """
        aug = np.array(inputs, dtype=np.float64)
        for n in self.frozen:
            out = np.float64(n.activate(aug))
            aug = np.append(aug, out)
        return aug

    # ── Prédiction ────────────────────────────────────

    def predict(self, inputs) -> int:
        """Sortie du neurone au sommet de la tour."""
        if not self.frozen:
            return 0
        return int(self._augment(inputs)[-1])

    def predict_vector(self, inputs) -> list:
        """Activation de chaque étage (hors entrées de base)."""
        return self._augment(inputs)[self.input_size:].tolist()

    # ── Pocket Perceptron (numpy) ─────────────────────

    def _pocket_train(self, neuron: BooleanNeuron, examples: list) -> int:
        """
        Entraîne neuron sur tous les examples par l'algorithme Pocket.
        Entièrement vectorisé avec numpy : pas de boucle Python par exemple.
        Retourne le nombre d'erreurs résiduelles.
        """
        # Construction de la matrice augmentée (une seule fois)
        self._build_cache()
        X = self._X_cache[:, :len(neuron.weights)]  # (n, p+k)
        y = self._y_cache.astype(np.float64)        # (n,)

        w     = neuron.weights.copy()
        theta = neuron.threshold
        best_w     = w.copy()
        best_theta = theta
        preds      = (X @ w >= theta).astype(np.float64)
        best_errs  = int(np.sum(preds != y))

        lr  = self.LEARNING_RATE
        n   = len(y)
        idx = np.arange(n)

        # On traite chaque exemple un par un et on met à jour w immédiatement
        # après chaque erreur. C'est le comportement exact du perceptron en
        # ligne de Rosenblatt, que le Pocket est censé encapsuler.
        for _ in range(self.PERCEPTRON_ITERATIONS):
            np.random.shuffle(idx)
            # Vrai perceptron en ligne : mise à jour exemple par exemple
            for i in idx:
                xi = X[i]
                yi = y[i]
                pred = 1.0 if np.dot(xi, w) >= theta else 0.0
                ei   = yi - pred          # +1, 0, ou -1
                if ei != 0:
                    w     = np.round(w + lr * ei * xi)
                    theta = round(theta - lr * ei)
            preds = (X @ w >= theta).astype(np.float64)
            curr  = int(np.sum(preds != y))
            if curr < best_errs:
                best_errs  = curr
                best_w     = w.copy()
                best_theta = theta
            if best_errs == 0:
                break


        neuron.weights   = best_w
        neuron.threshold = best_theta
        return best_errs

    def _new_neuron(self, label="", reason="") -> BooleanNeuron:
        """Crée un nouveau neurone avec p+k entrées (numpy)."""
        ext_size = self.input_size + len(self.frozen)
        # w = np.random.uniform(-0.5, 0.5, ext_size)
        # poids entiers comme Gallant le suggère
        w = np.random.choice([-1, 0, 1], size=ext_size).astype(np.float64)
        return BooleanNeuron(
            weights=w, threshold=0.0, input_size=ext_size,
            label=label, birth_reason=reason
        )

    # ── Tower fit (tous les exemples d'un coup) ───────

    def tower_fit(self, examples: list, label="", reason="") -> int:
        """
        Entraîne la tour sur tous les exemples d'un coup (Gallant 1990).
        Ajoute des neurones jusqu'à ce que tous les exemples soient
        correctement classifiés.
        Retourne le nombre de neurones ajoutés.
        """
        self.memory = list(examples)
        added = 0
        while True:
            # Vérifie si le réseau classifie déjà tout correctement
            errors = [(inp, exp) for inp, exp in self.memory
                      if self.predict(inp) != exp]
            if not errors:
                break
            # Crée et entraîne un nouveau neurone sur tous les exemples
            new_n = self._new_neuron(label=label, reason=reason)
            self._pocket_train(new_n, self.memory)
            self.frozen.append(new_n)
            added += 1
        return added


    def tower_fit_incremental(self, new_examples, label="", reason=""):
        """
        Ajoute de nouveaux exemples à la mémoire existante et
        force la création d'un neurone si les nouveaux exemples
        ne sont pas tous correctement classifiés.

        Le precedent tower_fit, réinitialise tout à chaque appel.
        Du coup on ne maitrise pas les neurones créés.
        Avec celui ci, on ne réinitialise pas. On peut donc construire un reseau incrémentalement.
        Voir le test Animals.py
        """
        self.memory.extend(new_examples)
        errors = [e for e in self.memory if self.predict(e[0]) != e[1]]
        if errors:
            new_n = self._new_neuron(label=label, reason=reason)
            self._pocket_train(new_n, self.memory)
            self.frozen.append(new_n)
            return True
        return False


    @property
    def size(self) -> int:
        return len(self.frozen)

    def to_dict(self):
        return {
            "input_size": self.input_size,
            "neurons"   : [n.to_dict() for n in self.frozen],
            "size"      : self.size,
        }

    def display_strategy(self, feature_names: list = None):
        """
        Affiche la stratégie apprise par la tour :
        pour chaque neurone, ses poids, son seuil, son label
        et sa birth_reason. Permet de visualiser ce que chaque
        niveau de la tour a appris.

        feature_names : noms optionnels des entrées originales
        """
        print(f"\n{'─'*60}")
        print(f"  Tower Strategy ({self.size} neurone(s), {self.input_size} entrées)")
        print(f"{'─'*60}")

        if not self.frozen:
            print("  Tour vide.")
            return

        for i, n in enumerate(self.frozen):
            print(f"\n  Neurone #{i}  [{n.label}]")
            print(f"  Raison     : {n.birth_reason}")
            print(f"  Seuil      : {n.threshold:.4f}")
            print(f"  Poids      :")

            # Poids des entrées originales
            for j in range(self.input_size):
                fname = feature_names[j] if feature_names and j < len(feature_names) else f"x{j}"
                print(f"    {fname:<15} : {n.weights[j]:+.4f}")

            # Poids de la sortie du neurone précédent (si présent)
            if len(n.weights) > self.input_size:
                print(f"    {'neurone #'+str(i-1):<15} : {n.weights[self.input_size]:+.4f}")


# ─────────────────────────────────────────
#  CONCEPT EDGE
# ─────────────────────────────────────────

@dataclass
class ConceptEdge:
    """
    Association entre deux concepts.
    Devient un concept social si utilisée >= 3 fois
    avec >= 3 partenaires distincts.
    """
    concept1: str
    concept2: str
    count   : int = 1
    partners: set = field(default_factory=set)
    neuron_created: bool = False

    @property
    def is_social_concept(self) -> bool:
        return self.count >= 3 and len(self.partners) >= 3


# ─────────────────────────────────────────
#  FACADE RÉSEAU — vue unifiée pour run.py
# ─────────────────────────────────────────

class AgentNetworkFacade:
    """
    Façade qui présente l'ensemble des Towers (niveau 1 + niveau 2)
    comme un seul réseau unifié pour la compatibilité avec run.py.

    Expose :
      .size    → nombre total de neurones dans tous les Towers
      .frozen  → liste unifiée de tous les neurones (tous Towers confondus)
    """

    def __init__(self, towers_l1: dict, tower_l2: TowerNetwork):
        self._towers_l1 = towers_l1
        self._tower_l2  = tower_l2

    @property
    def size(self) -> int:
        total  = sum(t.size for t in self._towers_l1.values())
        total += self._tower_l2.size
        return total

    @property
    def frozen(self) -> list:
        """Liste unifiée de tous les neurones, tous Towers confondus."""
        all_neurons = []
        for t in self._towers_l1.values():
            all_neurons.extend(t.frozen)
        all_neurons.extend(self._tower_l2.frozen)
        return all_neurons


# ─────────────────────────────────────────
#  CONCEPT EDGE
# ─────────────────────────────────────────

@dataclass
class ConceptEdge:
    """
    Association entre deux concepts.
    Devient un concept social si utilisée >= 3 fois
    avec >= 3 partenaires distincts.
    """
    concept1: str
    concept2: str
    count   : int = 1
    partners: set = field(default_factory=set)
    neuron_created: bool = False

    @property
    def is_social_concept(self) -> bool:
        return self.count >= 3 and len(self.partners) >= 3


# ─────────────────────────────────────────
#  AGENT# ─────────────────────────────────────────
#  AGENT
# ─────────────────────────────────────────

class Agent:
    """
    Agent cognitif autonome.

    Architecture multi-Tower :
      - _towers_l1 : dict mot → TowerNetwork (niveau 1, 64 entrées)
        Un Tower dédié par activité, reconnaît les variantes de signaux
        pour ce mot (familiales et apprises socialement).
      - _tower_l2  : TowerNetwork (niveau 2, n_activities entrées)
        Reconnaît les concepts sociaux à partir des sorties des Towers L1.

    Expose agent.network (AgentNetworkFacade) pour compatibilité run.py.
    """

    def __init__(self, agent_id: str, family_id: int,
                 all_signals: list, signal_to_idx: dict,
                 activity_names: list):
        self.id              = agent_id
        self.family_id       = family_id
        self._all_signals    = all_signals
        self._sig_to_idx     = signal_to_idx
        self._activity_names = activity_names

        # Niveau 1 : un Tower par activité (64 entrées chacun)
        self._towers_l1 : dict[str, TowerNetwork] = {
            word: TowerNetwork(input_size=len(all_signals))
            for word in activity_names
        }

        # Niveau 2 : Tower des concepts sociaux (n_activities entrées)
        self._tower_l2 = TowerNetwork(input_size=len(activity_names))

        # Façade unifiée pour run.py
        self.network = AgentNetworkFacade(self._towers_l1, self._tower_l2)

        self.lexicon            : dict[tuple, str]         = {}
        self.associations       : dict[tuple, ConceptEdge] = {}
        self.known_words        : set[str]                 = set()
        self.interaction_count  = 0
        self.successful_matches = 0
        self.conversation_log   : list[dict]               = []

    # ── Helpers ───────────────────────────────────────

    def _to_vector(self, signal_seq: list) -> list:
        """Vecteur binaire de 64 entrées pour les Towers L1."""
        v = [0] * len(self._all_signals)
        for s in signal_seq:
            if s in self._sig_to_idx:
                v[self._sig_to_idx[s]] = 1
        return v

    def _word_activation_vector(self, words: list) -> list:
        """
        Vecteur d'entrée du Tower L2 :
        position i = 1 si activity_names[i] est dans words, 0 sinon.
        """
        return [1 if w in words else 0
                for w in self._activity_names]

    # ── Phase parentale ───────────────────────────────

    def learn_from_parent(self, signal_seq: list, word: str):
        """Apprend une association signal→word depuis la famille."""
        key = tuple(sorted(signal_seq))
        vec = self._to_vector(signal_seq)
        if word in self._towers_l1:
            self._towers_l1[word].tower_fit_incremental(
                [(vec, 1)], label=word,
                reason=f"parental: {signal_seq} → {word}"
            )
        self.lexicon[key] = word
        self.known_words.add(word)

    # ── Reconnaissance ────────────────────────────────

    def get_word_for_signal(self, signal_seq: list) -> Optional[str]:
        """
        Reconnaît le mot associé à une séquence de signaux.
        Interroge le Tower L1 dédié à chaque mot et retourne
        celui dont le dernier neurone répond 1.
        Simple, direct et non ambigu.
        """
        vec = self._to_vector(signal_seq)
        for word, tower in self._towers_l1.items():
            if tower.predict(vec) == 1:
                return word
        return None

    # ── Apprentissage social ──────────────────────────

    def receive_new_concept(self, signal_seq: list,
                            word: str, from_agent: str) -> bool:
        """
        Apprend un nouveau concept depuis un pair.
        Entraîne uniquement le Tower L1 dédié à ce mot.
        Retourne True si un neurone a été ajouté.
        """
        key          = tuple(sorted(signal_seq))
        vec          = self._to_vector(signal_seq)
        neuron_added = False
        if word in self._towers_l1:
            neuron_added = self._towers_l1[word].tower_fit_incremental(
                [(vec, 1)], label=word,
                reason=f"appris de {from_agent}"
            )
        self.lexicon[key] = word
        self.known_words.add(word)
        return neuron_added

    # ── Associations & concepts sociaux ───────────────

    def form_association(self, word1: str, word2: str, partner_id: str):
        """
        Renforce une association entre deux mots.
        Si elle atteint le seuil social (3×3), entraîne le Tower L2
        avec le vecteur d'activation des deux mots concernés.
        """
        key = tuple(sorted([word1, word2]))
        if key not in self.associations:
            self.associations[key] = ConceptEdge(key[0], key[1])
        edge = self.associations[key]
        edge.count += 1
        edge.partners.add(partner_id)

        if edge.is_social_concept and not edge.neuron_created:
            edge.neuron_created = True
            vec_l2 = self._word_activation_vector([word1, word2])
            self._tower_l2.tower_fit_incremental(
                [(vec_l2, 1)],
                label  = f"{word1}+{word2}",
                reason = f"concept social (×{edge.count}, "
                         f"{len(edge.partners)} partenaires)"
            )

    def prune_associations(self):
        """Supprime les associations trop faibles."""
        to_remove = [k for k, e in self.associations.items()
                     if not e.is_social_concept and e.count < 2]
        for k in to_remove:
            del self.associations[k]

    @property
    def social_concepts(self) -> list:
        return [e for e in self.associations.values()
                if e.is_social_concept]

    # ── Généalogie ────────────────────────────────────

    def print_genealogy(self):
        """
        Affiche la biographie cognitive de l'agent Tower par Tower.
        Niveau 1 : un Tower par activité.
        Niveau 2 : Tower des concepts sociaux.
        """
        print(f"\n--- Généalogie de {self.id} (Famille {self.family_id}) ---")
        stats = {"parental": 0, "apprentissage social": 0, "concept social": 0}

        for word, tower in self._towers_l1.items():
            if tower.size > 0:
                for n in tower.frozen:
                    if "parental" in n.birth_reason:
                        stats["parental"] += 1
                    elif "appris de" in n.birth_reason:
                        stats["apprentissage social"] += 1

        for n in self._tower_l2.frozen:
            stats["concept social"] += 1

        for rtype, count in sorted(stats.items()):
            if count > 0:
                pct = count / self.network.size * 100 if self.network.size > 0 else 0
                print(f"  {rtype:<25} : {count:2d} ({pct:>5.1f}%)")

    # ── Sérialisation ─────────────────────────────────

    def to_dict(self):
        return {
            "id"                : self.id,
            "family_id"         : self.family_id,
            "network_size"      : self.network.size,
            "lexicon_size"      : len(self.lexicon),
            "known_words"       : list(self.known_words),
            "social_concepts"   : [(e.concept1, e.concept2, e.count)
                                   for e in self.social_concepts],
            "interaction_count" : self.interaction_count,
            "successful_matches": self.successful_matches,
            "towers_l1"         : {w: t.to_dict()
                                   for w, t in self._towers_l1.items()},
            "tower_l2"          : self._tower_l2.to_dict(),
        }


# ─────────────────────────────────────────
#  SIMULATION
# ─────────────────────────────────────────

class Simulation:
    """
    Simulation SOCIOGEN complète et paramétrique.

    Paramètres
    ----------
    n_agents     : nombre d'agents (>= n_families)
    n_families   : nombre de familles
    n_activities : nombre d'activités (1–16)
    n_signals    : nombre de signaux par activité par famille
    seed         : graine aléatoire pour reproductibilité
    """

    def __init__(self, n_agents=8, n_families=4,
                 n_activities=8, n_signals=2, seed=42):

        assert n_agents >= n_families, \
            "Il faut au moins autant d'agents que de familles."
        assert 1 <= n_activities <= len(ACTIVITY_NAMES), \
            f"n_activities doit être entre 1 et {len(ACTIVITY_NAMES)}."
        assert n_signals >= 1, \
            "n_signals doit être >= 1."

        random.seed(seed)

        self.n_agents     = n_agents
        self.n_families   = n_families
        self.n_activities = n_activities
        self.n_signals    = n_signals
        self.time         = 0

        # Vocabulaire dynamique
        self.activities, self.all_signals, self.signal_to_idx = \
            build_activities(n_activities, n_signals, n_families, seed=seed)

        self._activity_names = list(self.activities.keys())

        # Création des agents
        self.agents : list[Agent] = []
        for i in range(n_agents):
            fid = i % n_families
            self.agents.append(
                Agent(f"A{i}", fid,
                      self.all_signals, self.signal_to_idx,
                      self._activity_names)
            )

        # Phase parentale
        self._parental_phase()

    # ── Phase parentale ───────────────────────────────

    def _parental_phase(self):
        """
        Chaque agent apprend le lexique de sa famille.

        Si SIMILAR est True : un seul agent par famille effectue
        l'apprentissage, puis ses towers (L1 + L2) et son lexique
        sont clonés vers tous les autres membres de la famille.
        Si SIMILAR est False : chaque agent apprend indépendamment
        (légères variations dues à l'algorithme Pocket).
        """
        if not SIMILAR:
            # ── Mode standard : apprentissage indépendant ──
            for agent in self.agents:
                fid = agent.family_id
                for word, variants in self.activities.items():
                    variant = variants[fid % len(variants)]
                    agent.learn_from_parent(variant, word)
        else:
            # ── Mode clone : 1 apprentissage par famille ───
            # Regrouper les agents par famille
            families: dict[int, list] = {}
            for agent in self.agents:
                families.setdefault(agent.family_id, []).append(agent)

            for fid, members in families.items():
                # Le premier membre de la famille apprend
                reference = members[0]
                for word, variants in self.activities.items():
                    variant = variants[fid % len(variants)]
                    reference.learn_from_parent(variant, word)

                # Les autres membres reçoivent une copie exacte
                for agent in members[1:]:
                    # Cloner les towers L1 (un par activité)
                    agent._towers_l1 = {
                        word: copy.deepcopy(tower)
                        for word, tower in reference._towers_l1.items()
                    }
                    # Cloner le tower L2
                    agent._tower_l2 = copy.deepcopy(reference._tower_l2)
                    # Reconstruire la façade avec les nouveaux towers
                    agent.network = AgentNetworkFacade(
                        agent._towers_l1, agent._tower_l2
                    )
                    # Cloner le lexique et les mots connus
                    agent.lexicon     = dict(reference.lexicon)
                    agent.known_words = set(reference.known_words)

    # ── Pas de simulation ─────────────────────────────

    def step(self) -> Optional[dict]:
        """
        Un tick de simulation :
        - Tire aléatoirement un locuteur et un auditeur
        - Le locuteur produit un signal de son lexique
        - L'auditeur interroge son Tower L1 dédié au mot reçu
        - En cas d'échec, l'auditeur apprend (Tower L1 step)
        - En cas de succès, une association est renforcée
          et éventuellement gravée dans le Tower L2
        """
        self.time += 1
        a1, a2 = random.sample(self.agents, 2)
        if not a1.lexicon:
            return None

        signal_key, word = random.choice(list(a1.lexicon.items()))
        signal_seq       = list(signal_key)
        a2_word          = a2.get_word_for_signal(signal_seq)

        a1.interaction_count += 1
        a2.interaction_count += 1

        log = {
            "time"               : self.time,
            "speaker"            : a1.id,
            "listener"           : a2.id,
            "signal"             : signal_seq,
            "speaker_word"       : word,
            "response"           : a2_word,
            "match"              : a2_word == word,
            "network_growth"     : False,
            "new_concept_learned": None,
        }

        if a2_word == word:
            a1.successful_matches += 1
            a2.successful_matches += 1
            common = a1.known_words & a2.known_words - {word}
            if common:
                other = random.choice(list(common))
                a1.form_association(word, other, a2.id)
                a2.form_association(word, other, a1.id)
                log["association_formed"] = (word, other)
        else:
            grew                       = a2.receive_new_concept(signal_seq, word, a1.id)
            log["network_growth"]      = grew
            log["new_concept_learned"] = word

        if self.time % 10 == 0:
            for a in self.agents:
                a.prune_associations()

        a1.conversation_log.append(log)
        a2.conversation_log.append(log)
        return log

    def run(self, n_steps: int) -> list:
        """Lance n_steps ticks et retourne la liste des logs."""
        return [self.step() for _ in range(n_steps)]

    # ── Statistiques globales ─────────────────────────

    def stats(self):
        """
        Analyse l'influence des familles et l'origine des neurones.
        Distingue les neurones L1 (par activité) et L2 (concepts sociaux).
        """
        print(f"\n{'─'*20} STATISTIQUES GLOBALES (t={self.time}) {'─'*20}")

        family_influence = {i: 0 for i in range(self.n_families)}
        global_reasons   = {
            "parental"            : 0,
            "concept social (L2)" : 0,
            "apprentissage social": 0,
        }

        for agent in self.agents:
            for tower in agent._towers_l1.values():
                for n in tower.frozen:
                    r = n.birth_reason
                    if "parental" in r:
                        global_reasons["parental"] += 1
                    elif "appris de" in r:
                        global_reasons["apprentissage social"] += 1
                        try:
                            teacher_id = r.split("appris de ")[1].strip()
                            teacher    = next(
                                (a for a in self.agents if a.id == teacher_id), None
                            )
                            if teacher:
                                family_influence[teacher.family_id] += 1
                        except IndexError:
                            pass
            for n in agent._tower_l2.frozen:
                global_reasons["concept social (L2)"] += 1

        print("\n  [Répartition du savoir]")
        for k, v in global_reasons.items():
            print(f"    {k:<25} : {v}")

        print("\n  [Influence culturelle par famille]")
        for fid, count in family_influence.items():
            barre = "█" * count
            print(f"    Famille {fid} : {barre} ({count})")
        print()

    # ── Paramètres affichés ───────────────────────────

    def describe(self):
        """Affiche les paramètres courants de la simulation."""
        print(f"\n  Simulation SOCIOGEN")
        print(f"    agents      : {self.n_agents}")
        print(f"    familles    : {self.n_families}")
        print(f"    activités   : {self.n_activities}  "
              f"({', '.join(list(self.activities.keys())[:4])}...)")
        print(f"    signal/act  : {self.n_signals}")
        print(f"    signaux tot : {len(self.all_signals)}")
        print(f"    arch.       : {self.n_activities} Towers L1 "
              f"+ 1 Tower L2 par agent")
        print(f"    temps       : {self.time}\n")
