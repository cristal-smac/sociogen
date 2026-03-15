# Faire : python3 XOR_N.py 5
# Ou depuis l'interpréteur : S=5; exec(open('XOR_N.py').read())

import sys
sys.path.insert(0, '..')
from engine import TowerNetwork

# Un XOR à N entrées renvoie 1 si le nombre de 1 est impair.

# --- GESTION DU PARAMÈTRE N ---
# On récupère N depuis la ligne de commande, sinon par défaut N=4
if len(sys.argv) > 1:
    try:
        N = int(sys.argv[1])
    except ValueError:
        print("Erreur : L'argument doit être un entier.")
        sys.exit(1)
else:
    N = 4 

print(f"--- Problème XOR à N={N} entrées ---")

# 1. Génération de la table de vérité (2^N exemples)
# Un XOR à N entrées renvoie 1 si le nombre de 1 est impair.
examples = []
for i in range(2**N):
    # Conversion de l'index en liste de N bits
    inputs = [(i >> j) & 1 for j in reversed(range(N))]
    # Sortie 1 si la somme des bits est impaire
    expected = sum(inputs) % 2
    examples.append((inputs, expected))

print(f"Nombre d'exemples générés : {len(examples)}")

# 2. Initialisation du réseau Tower avec la taille d'entrée N
net = TowerNetwork(input_size=N)

# 3. Entraînement de la tour
# tower_fit ajoute des neurones jusqu'à résolution complète
net.tower_fit(examples, label=f"xor_{N}", reason=f"test_N{N}")

# 4. Vérification des résultats
print(f"Nombre de neurones générés : {net.size}")
errors = 0
for inputs, expected in examples:
    pred = net.predict(inputs)
    if pred != expected:
        errors += 1
    
    # Affichage individuel pour les petites dimensions uniquement
    if N <= 4:
        ok = "OK" if pred == expected else "NO"
        print(f"  XOR{inputs} = {expected}  -> {pred}  {ok}")

# 5. Bilan
print(f"\n--- Bilan ---")
print(f"Total erreurs : {errors}/{len(examples)}")
if errors == 0:
    print("Succès : Le problème est résolu à 100%.")

    
# 6. Détails de la hiérarchie construite

print("\n--- Détails des neurones ---")
for i, n in enumerate(net.frozen):
    print(f"Neurone #{i} | Label: {n.label} | Raison: {n.birth_reason}")    
    print(f"           | Poids: {n.weights.tolist()} | Seuil: {n.threshold}")
