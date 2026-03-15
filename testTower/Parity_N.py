# Faire exec(open('parity4.py').read())
# pour pouvoir travailler en interactif une fois l'exp terminée

import sys
sys.path.insert(0, '..')
from engine import TowerNetwork

# Problème de la parité à N entrées : 
# Renvoie 1 si le nombre de bits à 1 est impair, 0 sinon.


# --- GESTION DU PARAMÈTRE ---
# sys.argv[0] est le nom du script, sys.argv[1] est le premier argument
if len(sys.argv) > 1:
    try:
        N = int(sys.argv[1])
    except ValueError:
        print("Erreur : Le paramètre doit être un entier.")
        sys.exit(1)
else:
    # Valeur par défaut si aucun paramètre n'est fourni
    N = 4

# ---------------------

print(f"--- Problème de Parité à N={N} entrées ---")

# 1. Génération de l'espace de vérité
examples = []
for i in range(2**N):
    inputs = [(i >> j) & 1 for j in reversed(range(N))]
    expected = sum(inputs) % 2
    examples.append((inputs, expected))

# 2. Initialisation et Entraînement
net = TowerNetwork(input_size=N)
net.tower_fit(examples, label=f"parity_{N}", reason=f"test_n{N}")

# 3. Vérification (Correction du NameError)
print(f"Neurones générés : {net.size}")
errors = 0
for inputs, expected in examples:
    pred = net.predict(inputs)
    # On définit bien status ici pour éviter l'erreur
    status = "OK" if pred == expected else "NO"
    
    if pred != expected:
        errors += 1
    
    # Affichage des résultats individuels
    print(f"  Parity{inputs} = {expected}  -> {pred}  {status}")



print("-" * 30)
print(f"Total erreurs : {errors}/16")

print("\n--- Analyse de la structure de la Tour ---")
# On affiche la hiérarchie des neurones gelés
for i, n in enumerate(net.frozen):
    print(f"Neurone #{i} | Label: {n.label} | Raison: {n.birth_reason}")    
    print(f"           | Poids: {n.weights.tolist()} | Seuil: {n.threshold:.2f}")
