import sys
import random
sys.path.insert(0, '..')
from engine import TowerNetwork

# --- SCRIPT DE TEST POUR L'ARTICLE : APPRENTISSAGE PAR VAGUES ---

# Graine fixée pour reproductibilité (donne exactement 3 neurones)
random.seed(3)


net = TowerNetwork(input_size=4)


# Vague 1 : La règle évidente (Vole)
vague_1 = [
    ([1, 0, 1, 0], 1), # Aigle
    ([0, 1, 0, 0], 0), # Poisson
]
print("\n--- Vague 1 : Apprentissage de la règle de base ---")
net.tower_fit_incremental(vague_1, label="vague1", reason="vague 1")

# Vague 2 : Introduction de la contradiction (La Chauve-souris)
vague_2 = [
    ([1, 0, 1, 1], 0), # Chauve-souris (vole mais mammifère)
]
print("\n--- Vague 2 : Correction des mammifères volants ---")
net.tower_fit_incremental(vague_2, label="vague2", reason="vague 2")

# Vague 3 : Le cas spécifique (Le Pingouin)
vague_3 = [
    ([0, 1, 1, 0], 1), # Pingouin (ne vole pas mais pattes/nage)
    ([1, 1, 1, 0], 1), # Canard
    ([0, 0, 1, 1], 0), # Chien
    ([0, 1, 0, 1], 0), # Dauphin
]
print("\n--- Vague 3 : Finalisation avec les cas complexes ---")
net.tower_fit_incremental(vague_3, label="vague3", reason="vague 3")

print(f"\nNombre final de neurones : {net.size}")
net.display_strategy()
