# Faire exec(open('XOR2.py').read())
# pour pouvoir travailler en interactif une fois l'exp terminée


# le problème fondateur, celui qui a tué le perceptron simple en 1969
# (Minsky & Papert). Tout algorithme constructif doit le résoudre en 2
# neurones. C'est le test de base !!


import sys
sys.path.insert(0, '..')
from engine import TowerNetwork

examples = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]

net = TowerNetwork(input_size=2)

net.tower_fit(examples, label="xor", reason="test")

print(f"Neurones : {net.size}")
for inputs, expected in examples:
    pred = net.predict(inputs)
    ok = "OK" if pred == expected else "NO"
    print(f"  XOR{inputs} = {expected}  -> {pred}  {ok}")


print("\n--- Détails des neurones ---")
for i, n in enumerate(net.frozen):
    print(f"Neurone #{i} | Label: {n.label} | Raison: {n.birth_reason}")    
    print(f"           | Poids: {n.weights.tolist()} | Seuil: {n.threshold}")


#===========================================
# commandes en interactif
# net.size
# net.input_size
# net.frozen
# net.predict([0, 1, 0, 0])
# net.predict_vector([0, 1, 0, 0])

# n=net.frozen[0]   # le premier neurone
# n.label
# n.birth_reason
# n.threshold
# n.weights   
