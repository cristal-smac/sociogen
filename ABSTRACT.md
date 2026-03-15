# Résumé de l'algorithme

On considère 8 activités (crêpes, football, lecture, cuisine…) et 8 agents répartis en 4 familles (2 agents par famille). Chaque activité est représentée par une séquence de 2 signaux (ex. S004, S011), et chaque famille possède sa propre variante de signaux pour chaque activité — comme des "dialectes" distincts.

## Initialisation : 
Chaque agent est créé avec un réseau Tower vide et un lexique vide.

(** C'est la premiere variante possible : 1 tower par agent ou 1 tower par agent et par activité**)

Le réseau Tower est dimensionné dès le départ avec 64 entrées (8 activités × 4 familles × 2 signaux), représentant l'espace de tous les signaux possibles dans la simulation — y compris ceux des autres familles que l'agent ne connaît pas encore. Tous les agents connaissent l'ensemble des mots d'activités (crêpes, football, lecture…), et tous les agents d'une même famille partagent exactement les mêmes signaux pour chaque activité — des signaux entièrement distincts de ceux des autres familles. C'est seulement lors de la phase parentale que la correspondance entre mots et signaux est établie pour chaque agent. Le fait que le vecteur d'entrée couvre l'ensemble des signaux de toutes les familles est ce qui rendra possible l'apprentissage social : lorsqu'un agent reçoit des signaux d'une autre famille, il peut les placer correctement dans son vecteur et les soumettre à son Tower, même s'il ne sait pas encore les interpréter.

## L'algorithme s'exécute en 2 phases:

### Phase parentale: 
avant toute interaction sociale, chaque agent établit la correspondance entre ses signaux familiaux et les mots d'activités (le lexique). Pour chaque activité, l'agent associe la variante de signaux de sa famille au mot correspondant, et grave cette association dans son réseau Tower sous forme d'un ou plusieurs neurones booléens à seuil, chacun entraîné par l'algorithme Pocket sur l'ensemble des exemples mémorisés jusqu'alors. À l'issue de cette phase, tous les agents d'une même famille partagent exactement les mêmes correspondances mot<-->signaux, bien que leurs réseaux Tower puissent légèrement différer en raison de la part d'aléatoire de l'entraînement Pocket.

(** C'est la seconde variante possible : clonage ou différenciations dans une meme famille**)


## Phase sociale: 
À chaque tick, deux agents sont tirés aléatoirement : un locuteur et un auditeur. Le locuteur choisit un mot dans son lexique et émet la séquence de signaux correspondante. L'auditeur soumet ces signaux à son Tower : deux cas se présentent.
- Mismatch : le Tower de l'auditeur ne reconnaît pas les signaux (il répond 0, ou retourne un label incorrect). L'auditeur apprend alors la nouvelle correspondance signal<->mot et son Tower ajoute un neurone via l'algorithme Pocket, calibré pour reconnaître exactement ces signaux. Ce neurone porte en mémoire l'identité du locuteur source, traçant ainsi dans l'architecture même du réseau l'origine sociale de cet apprentissage. Un agent accumule ainsi, au fil du temps, plusieurs correspondances pour un même mot : la sienne, héritée de sa famille, et celles apprises au contact d'agents d'autres familles.
- Match : le Tower de l'auditeur reconnaît correctement le mot. Aucun neurone n'est créé. En revanche, les deux agents mettent à jour leur graphe d'associations personnel : un dictionnaire de paires de mots (mot1, mot2), dont chaque entrée est un objet ConceptEdge portant un compteur de co-occurrences (count) et un ensemble de partenaires distincts (partners). Concrètement, le mot reconnu est associé à un autre mot que les deux agents ont en commun, le count de la paire est incrémenté, et l'identité du partenaire est ajoutée à partners. Ce graphe est entièrement indépendant du Tower : il évolue exclusivement lors des matchs, là où le Tower évolue exclusivement lors des mismatches.

Lorsqu'une ConceptEdge franchit le seuil count >= 3 et partners >= 3, l'association cesse d'être anecdotique et acquiert le statut de concept social : une connaissance de plus haut niveau, comme "crêpes+cinéma", émergée spontanément de l'histoire collective des interactions sans avoir été programmée. C'est seulement à ce moment que le Tower est sollicité : un neurone dédié est ajouté pour graver ce concept dans l'architecture de l'agent, rejoignant ainsi la biographie cognitive aux côtés des neurones parentaux et sociaux. Le double seuil garantit qu'un concept social n'est pas le produit d'une relation dyadique particulière, mais d'une validation collective impliquant plusieurs partenaires distincts.

Toutes les 10 interactions, les ConceptEdge trop faibles (count < 2 et non encore sociales) sont élagées du graphe, évitant l'accumulation d'associations fortuites sans lendemain.


## Arrêt: 
La simulation s'arrête soit par convergence lexicale inter-familles (>= 85 % de matchs sur les 30 dernières interactions croisées), soit par stagnation neuronale (aucun nouveau neurone depuis 60 ticks), soit au bout de 500 ticks maximum.


## Bilan: 
à l'issue de la simulation, les réseaux Tower des agents ont divergé en fonction de leurs interactions sociales respectives. Chacun a maintenant donné naissance à sa propre "intelligence" (matérialisée par son Tower), enrichie non seulement de correspondances signal<-->mot apprises auprès d'autres agents, mais aussi de concepts de plus haut niveau (comme "crêpes+cinéma") émergés spontanément de l'histoire collective des interactions et partagés par les agents qui les ont co-construits. Grâce à la traçabilité du Tower Algorithm (chaque neurone portant l'étiquette du concept appris et l'origine de sa création) on peut reconstituer la "biographie cognitive" de chaque agent, distinguer la part du savoir hérité de la famille de celle acquise socialement, et mesurer l'influence culturelle de chaque famille sur l'ensemble de la population


# Résumé: Il y a donc 4 points forts dans ce modèle
- Naissance d'une intelligence à partir de rien : les agents démarrent avec un réseau Tower vide et construisent progressivement leur connaissance.
- Émergence d'intelligences différenciées : bien que partant d'une base commune, chaque agent développe un réseau unique, façonné par l'histoire particulière de ses interactions sociales.
- Émergence de concepts sociaux de haut niveau : des associations entre mots (comme "crêpes+cinéma") émergent spontanément de la répétition des interactions, sans avoir été programmées.
- IA explicable : grâce à la traçabilité du Tower Algorithm, chaque neurone porte l'étiquette de son origine, permettant de reconstituer intégralement la biographie cognitive de chaque agent.




# Remarques

Chaque agent possède un tower avec 64 entrées binaires et 1 sortie.
Quand on lui présente un vecteur avec des 1 pour les signaux, il dit si oui ou non il connait ce concept. Mais en aucun cas le concept (crèpes par exemple) n'est codé dans le réseau. Le seul moyen de le savoir c'est de regarder les labels. ET donc, il faut lancer l'apprentissage un exemple à la fois (et non pas en paquets) de manière à ce que le label ne contienne qu'un seul concept.

Deux versions sont possibles :
- Monotower : un seul Tower par agent
  - Avantage : ça prend moins de place
  - Inconvénient :
      - code compliqué : il faut parcourir tous les neurones activés et regarder les labels pour savoir quel concept correspond au signal présenté. e neurone existant répond 1 au nouveau vecteur par accident, donc le réseau croit connaître le concept alors qu'il ne l'a jamais appris explicitement. 
      - il se peut qu'avec un nouveau vecteur (donc nouveau concept), aucun neurone ne soit créé, et donc ce concept ne sera dans aucun label
  
- Multi-tower : un tower par activité dans chaque agent
  - inconvénient: pas de lien entre les concepts : l'apprentissage de l'un ne sert pas à l'apprentissage de l'autre.
  - avantage: Chaque Tower L1 est un expert isolé de son activité. Le lien entre concepts existe, mais il est géré séparément par le Tower L2 qui prend en entrée les sorties des Towers L1 — c'est là que les associations entre activités ("crêpes+cinéma") sont codées. 



# Questions

- Quelle est l'influence de la variante ? (on fait 1 seule fois l'apprentissage par famille et on clone le tower obtenu chez tous les membres)

- Quelle est l'influence du nombre d'agents par famille. A priori avec 1 seul, ça devrait aussi fonctionner ?
La seule contrainte c'est d'avoir suffisamment d'agents au total au regard de la condition d'arrêt (ici au moins 4, pour qu'un agent puisse interagir avec 3 partenaires distincts)


- Est-ce que l'on pourrait supprimer les familles et affecter aléatoirement les signaux à chaque agent ?
(Actuellement les familles servent à introduire une barrière lexicale initiale : deux agents de familles différentes ne partagent pas les mêmes signaux pour un même mot, ce qui force des échecs de reconnaissance et donc de l'apprentissage social inter-familles.)