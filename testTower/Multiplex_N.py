import sys
sys.path.insert(0, '..')
from engine import TowerNetwork


# Sortie : valeur du bit de données sélectionné par le selecteur


# ---------------------

# S :  Nombre de bits du sélecteur (S=1 -> Mux3, S=2 -> Mux6, S=3 -> Mux11)

# --- GESTION DU PARAMÈTRE ---
# sys.argv[0] est le nom du script, sys.argv[1] est le premier argument
if len(sys.argv) > 1:
    try:
        S = int(sys.argv[1])
    except ValueError:
        print("Erreur : Le paramètre doit être un entier.")
        sys.exit(1)
else:
    # Valeur par défaut si aucun paramètre n'est fourni
    S = 2

# ---------------------

# Calcul automatique des dimensions
num_data_bits = 2**S
N = S + num_data_bits

print(f"--- Multiplexieur avec S={S} selecteurs et N={N} entrées ---")

# 1. Génération de la table de vérité (2^N exemples)
examples = []
for i in range(2**N):
    # Génère une liste de N bits
    bits = [(i >> b) & 1 for b in range(N)]
    
    # Les S premiers bits sont le sélecteur
    selectors = bits[:S]
    # Les bits restants sont les données
    data = bits[S:]
    
    # Calcul de l'adresse sélectionnée (conversion binaire -> entier)
    address = 0
    for j in range(S):
        if selectors[j] == 1:
            address += 2**j
            
    # La sortie est le bit de donnée situé à l'adresse indiquée
    output = data[address]
    examples.append((bits, output))

print(f"Nombre d'exemples à traiter : {len(examples)}")

# 2. Initialisation du réseau Tower
net = TowerNetwork(input_size=N)

# 3. Entraînement
# Note : Pour Mux11 (2048 exemples), cela peut prendre du temps.
net.tower_fit(examples, label=f"mux_{N}", reason=f"test_S{S}")

# 4. Vérification et Statistiques
print(f"Neurones générés : {net.size}")
errors = 0
for inputs, expected in examples:
    pred = net.predict(inputs)
    if pred != expected:
        errors += 1

success_rate = (1 - (errors / len(examples))) * 100
print(f"Total erreurs : {errors}/{len(examples)}")
print(f"Taux de réussite : {success_rate:.1f}%")

# 5. Détail des neurones
print("\n--- Détails des neurones ---")
for i, n in enumerate(net.frozen):
    print(f"Neurone #{i} | Label: {n.label} | Raison: {n.birth_reason}")

