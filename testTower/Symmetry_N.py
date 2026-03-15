# Faire : python3 Symmetry_N.py 6
# Ou depuis l'interpréteur : N=6; exec(open('Symmetry_N.py').read())

import sys
sys.path.insert(0, '..')
from engine import TowerNetwork

# Problème de symétrie à N entrées :
# Un vecteur est symétrique (palindrome) si bits[i] == bits[N-1-i]
# pour tout i. Renvoie 1 si symétrique, 0 sinon.
# Exemple N=4 : [1,0,0,1] -> 1  |  [1,0,1,0] -> 0
# N doit être pair pour que le problème soit non trivial.

# --- GESTION DU PARAMÈTRE N ---
if len(sys.argv) > 1:
    try:
        N = int(sys.argv[1])
    except ValueError:
        print("Erreur : L'argument doit être un entier.")
        sys.exit(1)
else:
    N = 6

if N < 2:
    print("Erreur : N doit être >= 2.")
    sys.exit(1)

print(f"--- Problème de symétrie à N={N} entrées ---")
if N >= 8:
    print(f"  (avertissement : N={N} peut être très lent —"
          f" seulement {2**(N//2)}/{2**N} exemples positifs)")

# 1. Génération de la table de vérité (2^N exemples)
examples = []
for i in range(2**N):
    inputs = [(i >> j) & 1 for j in reversed(range(N))]
    expected = 1 if inputs == inputs[::-1] else 0
    examples.append((inputs, expected))

n_sym     = sum(1 for _, y in examples if y == 1)
n_non_sym = sum(1 for _, y in examples if y == 0)
print(f"Exemples générés   : {len(examples)}")
print(f"  dont symétriques : {n_sym}")
print(f"  dont asymétriques: {n_non_sym}")

# 2. Initialisation et entraînement
net = TowerNetwork(input_size=N)
net.tower_fit(examples, label=f"sym_{N}", reason=f"test_sym_N{N}")

# 3. Vérification
print(f"\nNeurones générés : {net.size}")
errors = 0
for inputs, expected in examples:
    pred = net.predict(inputs)
    if pred != expected:
        errors += 1

    # Affichage individuel pour les petites dimensions uniquement
    if N <= 4:
        tag = "SYM" if expected == 1 else "   "
        ok  = "OK" if pred == expected else "NO"
        print(f"  {tag} {inputs} = {expected}  -> {pred}  {ok}")

# 4. Bilan
print(f"\n--- Bilan ---")
print(f"Total erreurs : {errors}/{len(examples)}")
if errors == 0:
    print("Succès : Le problème est résolu à 100%.")
else:
    print(f"Echec : {errors} erreurs résiduelles.")

# 5. Détails des neurones
print("\n--- Détails des neurones ---")
for i, n in enumerate(net.frozen):
    print(f"Neurone #{i} | Label: {n.label} | Raison: {n.birth_reason}")
    print(f"           | Poids: {n.weights.tolist()} | Seuil: {n.threshold}")
