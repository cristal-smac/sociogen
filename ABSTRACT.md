# Résumé de l'algorithme

On considère 8 activités (crêpes, football, lecture, cuisine…) et 8 agents répartis en 4 familles (2 agents par famille). Chaque activité est représentée par une séquence de 2 signaux (ex. S004, S011), et chaque famille possède sa propre variante de signaux pour chaque activité, comme des "dialectes" distincts.

## Initialisation : 
Chaque agent est créé avec un réseau Tower vide et un lexique vide.

(** C'est la premiere variante possible : 1 tower par agent ou 1 tower par agent et par activité**)

Le réseau Tower est dimensionné dès le départ avec 64 entrées (8 activités × 4 familles × 2 signaux) et 1 seule sortie binaire : 1 si le signal présenté est reconnu, 0 sinon. Ces 64 entrées représentent l'espace de tous les signaux possibles dans la simulation, y compris ceux des autres familles que l'agent ne connaît pas encore. Tous les agents connaissent l'ensemble des mots d'activités (crêpes, football, lecture…), et tous les agents d'une même famille partagent exactement les mêmes signaux pour chaque activité, des signaux entièrement distincts de ceux des autres familles. C'est seulement lors de la phase parentale que la correspondance entre mots et signaux est établie pour chaque agent. Le fait que le vecteur d'entrée couvre l'ensemble des signaux de toutes les familles est ce qui rendra possible l'apprentissage social : lorsqu'un agent reçoit des signaux d'une autre famille, il peut les placer correctement dans son vecteur et les soumettre à son Tower, même s'il ne sait pas encore les interpréter. On notera que c'est le lexique de l'agent qui dit "l'agent A2 sait que les signaux [S004, S011] correspondent au mot crêpes". Le Tower n'est que le mécanisme qui permet de retrouver ce mot quand on lui présente ces signaux. Une propriété fondamentale (et une limite) du Tower tel qu'utilisé dans SOCIOGEN est qu'il n'apprend que des exemples positifs : on lui dit "ce signal correspond à ce concept" (réponse attendue = 1), jamais "ce signal ne correspond pas à ce concept" (réponse attendue = 0). Autrement dit, on lui pose toujours la question "est-ce que tu connais ce signal ?" et on lui apprend uniquement à répondre "oui". Il ne sait jamais explicitement dire "non" à un signal inconnu, il répond 0 par défaut, faute d'activation, pas par apprentissage délibéré. Cette asymétrie est la source des faux positifs silencieux : un neurone entraîné pour reconnaître S002->crêpes peut accidentellement répondre 1 à S009->football, non pas parce qu'il a appris à le faire, mais parce qu'il n'a jamais appris à ne pas le faire.


## L'algorithme s'exécute en 2 phases:

### Phase parentale: 
avant toute interaction sociale, chaque agent établit la correspondance entre ses signaux familiaux et les mots d'activités (le lexique). Pour chaque activité, l'agent associe la variante de signaux de sa famille au mot correspondant, et grave cette association dans son réseau Tower sous forme d'un ou plusieurs neurones booléens à seuil, chacun entraîné par l'algorithme Pocket sur l'ensemble des exemples mémorisés jusqu'alors. À l'issue de cette phase, A0 et A4 (tous deux en F0) ont exactement le même lexique : S002->crêpes, S008->cinéma, etc. Mais A1 (F1) dit "crêpes" avec S001, pas S002. Cette phase est la condition nécessaire de toute la suite : sans elle, tous les agents auraient les mêmes signaux pour les mêmes mots, personne ne se tromperait jamais, personne n'apprendrait jamais rien, et les Tower resteraient vides. C'est l'inégalité de départ, contrôlée et structurée par l'appartenance familiale, qui rend la divergence cognitive possible.

(** C'est la seconde variante possible : clonage ou différenciations dans une meme famille**)


## Phase sociale: 
À chaque tick, deux agents sont tirés aléatoirement : un locuteur et un auditeur. Le locuteur choisit un mot dans son lexique et émet la séquence de signaux correspondante. L'auditeur soumet ces signaux à son Tower : deux cas se présentent.
- Mismatch : le Tower de l'auditeur ne reconnaît pas les signaux (il répond 0, ou retourne un label incorrect). L'auditeur apprend alors la nouvelle correspondance signal<->mot et son Tower ajoute un neurone via l'algorithme Pocket, calibré pour reconnaître exactement ces signaux. Ce neurone porte en mémoire l'identité du locuteur source, traçant ainsi dans l'architecture même du réseau l'origine sociale de cet apprentissage. Un agent accumule ainsi, au fil du temps, plusieurs correspondances pour un même mot : la sienne, héritée de sa famille, et celles apprises au contact d'agents d'autres familles. si A0 a émis le signal S002 pour "crêpes" en direction de A1, et qu'il y a eu mismatch, alors A1 a appris S002 -> crêpes. Cette entrée est dans son lexique et son Tower est entraîné dessus.
- Match : le Tower de l'auditeur reconnaît correctement le mot. Aucun neurone n'est créé. En revanche, les deux agents mettent à jour leur graphe d'associations personnel : un dictionnaire de paires de mots (mot1, mot2), dont chaque entrée est un objet ConceptEdge portant un compteur de co-occurrences (count) et un ensemble de partenaires distincts (partners). Concrètement, le mot reconnu est associé à un autre mot que les deux agents ont en commun, le count de la paire est incrémenté, et l'identité du partenaire est ajoutée à partners. Ce graphe est entièrement indépendant du Tower : il évolue exclusivement lors des matchs, là où le Tower évolue exclusivement lors des mismatches. A0 émet S002 ("crêpes"), A1 reconnaît "crêpes" -> match. Les deux agents incrémentent le ConceptEdge (crêpes, cinéma) si cinéma est dans leur vocabulaire commun.

Lorsqu'une ConceptEdge franchit le seuil count >= 3 et partners >= 3, l'association cesse d'être anecdotique et acquiert le statut de concept social : une connaissance de plus haut niveau, comme "crêpes+cinéma", émergée spontanément de l'histoire collective des interactions sans avoir été programmée. C'est seulement à ce moment que le Tower est sollicité : un neurone dédié est ajouté pour graver ce concept dans l'architecture de l'agent, rejoignant ainsi la biographie cognitive aux côtés des neurones parentaux et sociaux. Le double seuil garantit qu'un concept social n'est pas le produit d'une relation dyadique particulière, mais d'une validation collective impliquant plusieurs partenaires distincts.

Toutes les 10 interactions, les ConceptEdge trop faibles (count < 2 et non encore sociales) sont élagués du graphe, évitant l'accumulation d'associations fortuites sans lendemain.


## Arrêt: 
La simulation s'arrête soit par convergence lexicale inter-familles (>= 85 % de matchs sur les 30 dernières interactions croisées), soit par stagnation neuronale (aucun nouveau neurone depuis 60 ticks), soit au bout de 500 ticks maximum.


## Bilan: 
À l'issue de la simulation, les réseaux Tower des agents ont divergé en fonction de leurs interactions sociales respectives. Chaque agent conserve les signaux de sa famille pour produire des messages, A0 dira toujours "crêpes" avec S002 et S008, jamais autrement, mais a explicitement appris, lors des mismatches, à reconnaître les signaux des autres familles : si A1 a émis S001 en direction de A0 et qu'il y a eu mismatch, alors A0 a appris S001 -> crêpes, cette entrée est gravée dans son lexique et son Tower est entraîné dessus de façon permanente. Le modèle ne produit pas une langue commune, les agents ne parlent pas tous de la même façon, mais une compréhension mutuelle effective et traçable : quand A1 émet S001, A0 sait qu'il parle de crêpes parce qu'il l'a appris, pas par accident.

Cette compréhension est cependant partielle et asymétrique en cours de simulation : si A3 et A7 ne se sont jamais rencontrés, A7 ignore encore les signaux de F3. Plus un agent a interagi avec des partenaires de familles diverses, plus son lexique est riche et son Tower étendu. C'est pourquoi deux agents d'une même famille peuvent, après la phase sociale, avoir des Tower de tailles différentes : A0 a peut-être rencontré des agents de F1, F2 et F3, tandis que A4, issu de la même famille, avec le même lexique initial, n'a croisé que des agents de F1.

Si l'on garantit en revanche que tous les agents ont interagi avec tous les autres, la compréhension mutuelle devient complète et universelle : chaque agent reconnaît alors l'intégralité des signaux de toutes les familles, S023 S025 pour "crêpes" selon F0, S007 S022 selon F1, S033 S045 selon F2, S019 S059 selon F3, et sait sans ambiguïté de quel concept il s'agit, quelle que soit la façon de le dire. Pourtant leurs réseaux restent structurellement différents : l'ordre dans lequel A0 a appris ces signaux, et les neurones que cet apprentissage a créés, ne sont pas les mêmes que ceux de A4. La compréhension converge, la biographie diverge, et c'est précisément le résultat central du modèle.

Chacun a ainsi donné naissance à sa propre "intelligence" (matérialisée par son Tower), enrichie non seulement de correspondances signal↔mot apprises auprès d'autres agents, mais aussi de concepts de plus haut niveau (comme "crêpes+cinéma") émergés spontanément de l'histoire collective des interactions et partagés par les agents qui les ont co-construits. Grâce à la traçabilité du Tower Algorithm, chaque neurone portant l'étiquette du concept appris et l'origine de sa création, on peut reconstituer la "biographie cognitive" de chaque agent, distinguer la part du savoir hérité de la famille de celle acquise socialement, et mesurer l'influence culturelle de chaque famille sur l'ensemble de la population.




# Résumé: Il y a donc 4 points forts dans ce modèle
- Naissance d'une intelligence à partir de rien : les agents démarrent avec un réseau Tower vide et construisent progressivement leur connaissance.
- Émergence d'intelligences différenciées : bien que partant d'une base commune, chaque agent développe un réseau unique, façonné par l'histoire particulière de ses interactions sociales.
- Émergence de concepts sociaux de haut niveau : des associations entre mots (comme "crêpes+cinéma") émergent spontanément de la répétition des interactions, sans avoir été programmées.
- IA explicable : grâce à la traçabilité du Tower Algorithm, chaque neurone porte l'étiquette de son origine, permettant de reconstituer intégralement la biographie cognitive de chaque agent.



# Ce qui est remarquable
C'est remarquable pour trois raisons qui se renforcent mutuellement.
- La divergence émerge sans être programmée
Personne n'a dit aux agents "soyez différents". Ils partent tous avec le même réseau vide, les mêmes mots, les mêmes règles d'interaction. La seule source de diversité est l'ordre aléatoire des rencontres, qui parle à qui, dans quel ordre, sur quel mot. Et pourtant les réseaux finaux sont tous structurellement distincts. La divergence cognitive est un attracteur du système, pas un paramètre.
A0 et A4 partent avec le même Tower après la phase familiale. Après 100 interactions, A0 a 12 neurones, A4 en a 9, simplement parce qu'ils n'ont pas rencontré les mêmes agents dans le même ordre.
- La divergence et la convergence coexistent, et c'est paradoxal !
Dans tous les modèles classiques de type Talking Heads, convergence du langage et convergence cognitive vont de pair : quand les agents se comprennent, ils pensent pareil. SOCIOGEN montre que ce couplage n'est pas une nécessité, on peut avoir une compréhension mutuelle complète (tous les signaux reconnus par tous) tout en ayant des architectures cognitives irréconciliables. Se comprendre ne veut pas dire penser pareil. C'est un résultat nouveau.
- Chaque différence est explicable.
Dans un réseau profond entraîné par rétropropagation, on peut observer que A0 et A4 se comportent différemment, mais on ne peut pas expliquer pourquoi. Dans SOCIOGEN, chaque différence entre les Tower de A0 et A4 est traçable à un événement social précis : A0 a rencontré A3 au step 47 sur le mot "football", d'où ce neurone-là, avec ce poids-là. La divergence n'est pas seulement mesurable, elle est lisible. Le réseau est une biographie. A0 a un neurone #9 portant "appris de A3, step 47, football" — A4 n'a pas ce neurone. Cette différence est entièrement traçable.

Ces trois points ensemble répondent à une question que personne n'avait posée dans ces termes : comment des esprits différents peuvent-ils émerger d'agents identiques, par la seule force des interactions sociales, de façon entièrement explicable ? C'est ce que SOCIOGEN démontre.



# Frequently Asked Questions (F.A.Q) :

- Quelle est la taille des Tower ? est-ce que c'est la même pour tous ? <p/>
Tous les Tower ont exactement la même taille d'entrée : k×m×s bits, soit dans la configuration de référence 8×4×2=64 entrées binaires. C'est fixé une fois pour toutes à la création et ça ne change jamais. Chaque bit correspond à un signal possible dans toute la population, y compris les signaux des autres familles que l'agent ne connaît pas encore.
Le Tower a toujours 1 seule sortie binaire : 0 ou 1. Il répond simplement "je connais / je ne connais pas" pour le signal présenté.
- Comment sont les réseaux après la phase familiale ? <p/>
tous les agents ont le même nombre de neurones : exactement k neurones, un par activité. Par exemple avec 8 activités, chaque agent a 8 neurones, qu'il soit A0 ou A7. C'est la seule fois où tout le monde est identique.
Après la phase sociale , les tailles divergent. Chaque mismatch ajoute au moins un neurone chez l'auditeur, et chaque concept social en ajoute un de plus. Deux agents de la même famille qui ont eu des trajectoires différentes auront des Tower de tailles différentes. 
-  À quoi sert la phase familiale ?<p/>
Elle crée la diversité initiale sans laquelle il ne se passe rien. Sans elle, tous les agents auraient les mêmes signaux pour les mêmes mots, personne ne se tromperait jamais, personne n'apprendrait jamais rien, les Tower resteraient vides. La phase familiale est l'inégalité de départ qui rend la divergence possible.
- À quoi sert le Tower ? Si les agents enregistraient directement les associations signal↔concept dans leur lexique, ça changerait quoi ? <p/>
Rien pour la communication, les agents se comprendraient exactement pareil. Le Tower sert uniquement à retrouver un mot à partir de signaux inconnus, là où le lexique est un dictionnaire à clé exacte. Mais surtout : sans Tower, il n'y aurait pas de biographie. C'est le Tower qui rend chaque apprentissage traçable, daté et attribuable à un pair. Le lexique dit quoi, le Tower dit comment et d'où.
- Si on laisse tourner jusqu'à ce que chaque agent ait parlé avec tous les autres, tout le monde se comprendrait ?<p/>
Oui, complètement. Chacun garderait ses propres signaux pour parler, mais reconnaîtrait tous les signaux de toutes les familles. On peut dire n'importe quoi de n'importe quelle façon, tout le monde suit.
- Pourquoi arrêter avant ? Quel est l'intérêt ?<p/>
Parce que c'est ce qui se passe dans la réalité, personne ne parle avec tout le monde. Et c'est précisément dans cet état inachevé que la divergence est la plus intéressante : deux agents de la même famille ont eu des parcours différents, donc des Tower différents, donc des biographies différentes. Si on attend la convergence complète, tout le monde a le même lexique et les biographies se ressemblent de plus en plus. L'inachèvement est la condition de la diversité.
- Si A0 apprend "crêpes" via S001 de A1, puis via S017 de A2, un neurone pour chacun, ou un seul ? <p/>
Un neurone pour chacun. Chaque mismatch crée au moins un neurone, calibré sur le signal reçu ce jour-là. Le Tower de A0 contient donc deux neurones distincts, chacun portant le nom de son créateur : "appris de A1" et "appris de A2".
- Est-ce qu'un agent peut désapprendre ? Que se passe-t-il si deux agents lui enseignent des choses contradictoires sur le même signal ? <p/>
Non, l'inoubli est absolu. Et les contradictions sont gérées par le Pocket algorithm : si S005 a été associé à "crêpes" par A1 puis à "football" par A3, le Tower crée un neurone supplémentaire. le Pocket algorithm fait de son mieux pour satisfaire les deux exemples contradictoires, quitte à laisser une erreur résiduelle. Néanmoins, il n'oublie rien, il s'adapte.
- Si on retire A0 à mi-parcours, est-ce que son influence survit ? <p/>
Oui, pour toujours. Chaque neurone créé grâce à A0 porte la mention "appris de A0" et reste gravé dans le Tower de ses interlocuteurs. A0 peut disparaître, sa trace cognitive est permanente.
- Un agent très sociable comprend-il mieux, ou se fait-il mieux comprendre ? <p/>
Il comprend mieux, son Tower est plus riche, son lexique plus large. Mais il ne se fait pas nécessairement mieux comprendre : ses propres signaux restent familiaux, identiques à ceux d'un agent peu sociable de la même famille.
- Deux agents qui ne se sont jamais rencontrés peuvent-ils se comprendre via un intermédiaire commun ? <p/>
Oui, si A0 et A7 ont tous les deux appris de A3, ils partagent les signaux de F3 sans s'être jamais parlé. La compréhension se propage par transitivité, exactement comme dans un réseau social humain.
- À quoi servent les ConceptEdge ? <p/>
À rien pour la communication, les agents se comprendraient exactement pareil sans eux. Ils servent à modéliser quelque chose de plus riche : le fait qu'après une interaction réussie, les deux agents ne font pas que valider un mot, ils renforcent un lien entre ce mot et un autre qu'ils ont en commun. "Aujourd'hui on a parlé de crêpes ensemble, et on connaît tous les deux le cinéma, ces deux concepts se retrouvent souvent associés dans nos échanges." Quand ce lien a été validé collectivement par suffisamment de partenaires, il devient un concept social et laisse une trace dans le Tower. C'est la couche sémantique au-dessus de la couche lexicale.
- Est-ce que les associations signal↔ConceptEdge sont stockées dans le lexique ? <p/>
Non, et c'est une distinction importante. Le lexique ne contient que des associations signal -> mot simples, comme S002 -> crêpes. Les ConceptEdge vivent dans un graphe séparé (self.associations), dont les nœuds sont des mots et les arêtes des compteurs. Le lexique dit comment reconnaître un mot, le graphe dit quels mots vont ensemble. Ce sont deux mémoires de nature différente, et elles évoluent dans des situations opposées : le lexique lors des mismatches, le graphe lors des matchs.



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
  - avantage: Chaque Tower L1 est un expert isolé de son activité. Le lien entre concepts existe, mais il est géré séparément par le Tower L2 qui prend en entrée les sorties des Towers L1, c'est là que les associations entre activités ("crêpes+cinéma") sont codées. 



# Questions

- Quelle est l'influence de la variante ? (on fait 1 seule fois l'apprentissage par famille et on clone le tower obtenu chez tous les membres)

- Quelle est l'influence du nombre d'agents par famille. A priori avec 1 seul, ça devrait aussi fonctionner ?
La seule contrainte c'est d'avoir suffisamment d'agents au total au regard de la condition d'arrêt (ici au moins 4, pour qu'un agent puisse interagir avec 3 partenaires distincts)


- Est-ce que l'on pourrait supprimer les familles et affecter aléatoirement les signaux à chaque agent ?
(Actuellement les familles servent à introduire une barrière lexicale initiale : deux agents de familles différentes ne partagent pas les mêmes signaux pour un même mot, ce qui force des échecs de reconnaissance et donc de l'apprentissage social inter-familles.)
