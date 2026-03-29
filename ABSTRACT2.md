Initialement, dans chaque famille, chaque agent a un lexicon vide et un tower vide (Lignes 417-420). Le lexique est une mémoire exacte (l'acquis). Le réseau est son intuition (la plasticité cérébrale).

## Phase familiale :

- On distribue aux enfants n_activities x n_signals (par défaut 8x2) rangés dans leur lexicon (Lignes 561-563). Par exemple : A0 et son frère A1 reçoivent tous deux le signal 'S001' pour le mot 'pancakes'. C'est leur socle commun : ils n'ont pas besoin de réfléchir pour se comprendre sur ce mot.
- On lance ensuite l'apprentissage de ce lexique chez chaque agent (Ligne 563). Si SIMILAR = False : l'ordre est aléatoire et les poids initiaux sont aléatoires (Lignes 547-568). Bien qu'A0 et A1 apprennent le même dictionnaire, le cerveau d'A0 se configure d'une certaine manière, et celui d'A1 d'une autre. A0 pourrait devenir très doué pour reconnaître les signaux en début de liste, tandis qu'A1 sera plus sensible aux signaux en fin de liste. Leurs intuitions divergent déjà (Lignes 211-218).
Si SIMILAR = True : Le code force les agents d'une même famille à être des copies conformes via un clonage profond (Lignes 552-558).

La couche d'entrée du réseau possède une taille fixe égale à len(all_signals) (Ligne 420). En phase familiale, on utilise un One-Hot Encoding : un seul '1' dans le vecteur d'entrée (Lignes 424-429). La sortie est booléenne (0 ou 1) (Ligne 116). L'agent parcourt sa 'Tour' et le mot reconnu correspond au label du dernier neurone activé (Lignes 444-448). Par exemple : Si A0 entend un signal inconnu 'S099', sa tour pourrait s'allumer par erreur sur le neurone 'pancakes' car le signal ressemble à 'S001'. C'est un faux positif, mais c'est le début de l'interprétation.

## Phase sociale :

Dès que la phase sociale commence, les agents se parlent au hasard, qu'ils soient de la même famille ou pas (Ligne 578). L'agent utilise toujours son Lexique (mémoire exacte) avant son Réseau (intuition) (Lignes 442-443). Par exemple : A0 rencontre A1 (d'une autre famille). A1 utilise le mot 'music' avec le signal 'S050'. A0 ne connaît pas ce signal. Il fouille son lexique (échec), puis interroge sa Tower (intuition). Si il ne comprend rien, il demande la correction et écrit 'S050: music' en dur dans son lexique (Ligne 461).

Quand deux agents se parlent, le locuteur part du sens (le mot) pour arriver au son (le signal) (Ligne 582). Par exemple : A0 veut parler de 'pancakes'. Elle regarde ses known_words (Ligne ?). Elle voit qu'elle a deux signaux pour ce mot ('S001' et le nouveau 'S088' appris d'un étranger). Elle en choisit un au hasard pour le dire à son interlocuteur (Ligne 582).

- Le Mismatch (Stabilisation) : Si l'auditeur se trompe ou ne reconnaît rien (a2_word != word), un cycle de correction s'enclenche (Ligne 599). Par exemple : A0 dit 'football' avec le signal 'S012'. A5 comprend 'pancakes'. Mismatch ! A0 corrige A5. A5 effectue un apprentissage immédiat pour ajuster les poids de sa Tower (Lignes 451-463). On stabilise la langue.
- Le Match (Exploration) : Si l'auditeur reconnaît le bon label (a2_word == word), la communication est réussie (Ligne 590). C'est ici que l'on innove : les agents identifient leurs connaissances communes (common) pour tenter d'enrichir leur échange (Lignes 593-594).



L'innovation naît de la répétition de succès partagés (La règle du '3x3'):

Lors d'un match, les agents tirent au sort un mot commun pour former un couple de concepts via form_association (Lignes 596-597). Par exemple : A0 et A5 se sont compris sur 'pancakes'. Ils décident d'y associer 'music'. Ils créent un lien temporaire 'pancakes+music'.

L'association n'est validée que si elle est utilisée au moins 3 fois avec au moins 3 partenaires distincts (Lignes 406). Par exemple : Si A0 fait cette association avec A5, puis avec A7, puis avec A3, son ConceptEdge devient mature (Lignes 393-407).
L'agent crée alors un nouvel étage dans sa Tower avec le label fusionné 'pancakes+music' (Lignes 481-488). Ces nouveaux concepts résident exclusivement dans la structure neuronale.

 Pour identifier 'pancakes+music', l'agent utilise sa TowerNetwork (Ligne 438). Sa tour parcourt les activations en partant du sommet pour trouver le label social (Lignes 443-448).
Pour parler de cette association, l'agent combine les signaux originaux. Le vecteur d'entrée contient alors deux '1' simultanément (Lignes 424-429). Par exemple : Pour dire 'pancakes+music', A0 envoie les signaux 'S001' et 'S050'. Le cerveau de A5 reçoit ces deux impulsions en même temps. Sa Tower, entraînée, active le neurone spécifique au sommet de sa tour : le concept social est transmis. C'est l'émergence du langage !


Pour éviter que le cerveau des agents ne soit encombré par des milliers d'associations qui n'ont servi qu'une seule fois, un mécanisme de nettoyage automatique intervient régulièrement (Ligne 603). Si une association n'est pas encore devenue un concept social et qu'elle n'a été utilisée qu'une seule fois (edge.count < 2), elle est supprimée du graphe de l'agent (ligne 490). Par exemple : Si A0 a associé 'pancakes' et 'football' une seule fois avec A5 et que cela ne s'est plus jamais reproduit, cette idée est effacée pour laisser la place à des concepts plus robustes. C'est une forme d'oubli sélectif qui favorise l'émergence d'une culture stable.



Ce qui est remarquable !

    Convergence sans uniformité : A0 et A5 se comprennent sur 'pancakes+music' sans avoir les mêmes poids synaptiques. Le langage est le pont entre deux cerveaux différents.

    La stabilité crée la complexité : Ce n'est pas l'erreur qui est créatrice, mais le succès. C'est parce qu'A0 et Bob sont en phase (Match) qu'ils peuvent se permettre d'inventer de nouveaux sens (Lignes 590-597).

    L'intuition dépasse l'acquis : C'est parce que la Tower est capable de gérer des entrées complexes (plusieurs '1') que le langage peut dépasser le dictionnaire figé hérité des parents (Lignes 481-488).

En résumé : Succès -> Association -> Validation sociale (3x3) -> Innovation.