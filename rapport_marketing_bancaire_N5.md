![Université Hassan 1er — ENCG Settat](figures/banniere_uh1_encg.jpg)

# Prédire la souscription à un dépôt à terme : apprentissage automatique appliqué au marketing bancaire

**Projet N°5 — Science des données**

**ENCG Settat — Filière Finance, S7 G2**

Réalisé par : **EL OMARI AYA** et **EL ADRAOUI AYA**

Encadré par : **M. LAGHLIMI**

Année universitaire 2026-2027

---

## Sommaire

1. Contexte et problématique
2. Présentation et analyse des données
3. Démarche de modélisation
4. Résultats du modèle retenu
5. Lecture du modèle
6. Limites
7. Conclusion

---

## 1. Contexte et problématique

Les campagnes de démarchage téléphonique coûtent cher à une banque. Chaque appel mobilise un conseiller, et la grande majorité des clients contactés finissent par refuser l'offre.

Cette étude porte sur les campagnes d'une banque portugaise qui proposait à ses clients un dépôt à terme. Le jeu de données Bank Marketing (Moro, Cortez et Rita, 2014, UCI) recense 41 188 appels passés entre mai 2008 et novembre 2010. Chaque observation est décrite par 20 variables, qui se regroupent en quatre familles : le profil du client (âge, profession, situation familiale, crédits en cours), le dernier appel (type de téléphone, mois, jour, durée), l'historique des campagnes précédentes et le contexte économique (Euribor 3 mois, taux d'emploi, indices des prix et de confiance). La variable cible `y` indique si le client a souscrit ou non.

La réponse étant connue pour chaque client, il s'agit d'apprentissage supervisé, plus précisément d'une classification binaire (oui / non).

La problématique retenue est la suivante : **à partir des informations disponibles avant l'appel, peut-on identifier les clients les plus susceptibles de souscrire, afin de concentrer les appels sur eux sans perdre l'essentiel des ventes ?**

## 2. Présentation et analyse des données

### 2.1 Qualité des données

Le fichier ne contient aucune valeur vide. En revanche, certaines informations sont codées `unknown`, surtout pour `default` (8 597 clients dont on ignore s'ils ont déjà fait défaut). Ces valeurs ont été conservées comme une modalité à part entière, puisque l'absence d'information peut elle-même être révélatrice. On relève aussi 12 lignes en double, supprimées par la suite. Enfin, `pdays` vaut 999 pour 96 % des clients : la plupart n'avaient jamais été contactés auparavant.

### 2.2 Un échantillon très déséquilibré

![Graphique 1 – Répartition des clients selon la souscription](figures/cell17.png)

*Graphique 1 – Répartition des clients selon la souscription*

Seuls 11,3 % des clients ont souscrit. Ce déséquilibre conditionne tout le reste de l'analyse. Un modèle qui répondrait « non » à chaque fois aurait raison dans 88,7 % des cas, alors qu'il serait inutile pour la banque. L'exactitude n'est donc pas un bon critère ici. On s'appuiera plutôt sur l'AUC et sur le rappel, c'est-à-dire la part des souscripteurs effectivement repérés.

### 2.3 Variables numériques

![Graphique 2 – Distribution des variables numériques selon la souscription](figures/cell21.png)

*Graphique 2 – Distribution des variables numériques selon la souscription*

| Médiane | âge | campaign | euribor3m | emp.var.rate | nr.employed |
|---|---|---|---|---|---|
| Non | 38 | 2 | 4,86 | 1,1 | 5 195,8 |
| Oui | 37 | 2 | 1,27 | −1,8 | 5 099,1 |

Les écarts les plus nets concernent les variables économiques. La médiane de l'Euribor 3 mois est de 1,27 % chez les souscripteurs, contre 4,86 % chez les autres. Les souscriptions sont aussi plus fréquentes lorsque l'emploi recule (`emp.var.rate` négatif). L'âge joue peu en médiane, même si les plus jeunes et les plus âgés semblent un peu plus réceptifs. Quant au nombre d'appels pendant la campagne (`campaign`), on remarque que les clients relancés de nombreuses fois souscrivent rarement. On note aussi que `euribor3m`, `emp.var.rate` et `nr.employed` sont corrélées à plus de 0,9 : elles traduisent toutes la conjoncture.

### 2.4 Variables qualitatives

![Graphique 3 – Taux de souscription par modalité](figures/cell25.png)

*Graphique 3 – Taux de souscription par modalité (pointillés : moyenne de 11,3 %)*

Plusieurs variables qualitatives font nettement varier le taux de souscription :

- quand la campagne précédente avait abouti (`poutcome = success`), 65 % des clients souscrivent à nouveau, contre 9 % pour ceux qui n'avaient jamais été contactés ;
- les clients joints sur portable souscrivent environ trois fois plus que ceux appelés sur une ligne fixe (15 % contre 5 %) ;
- mars, septembre, octobre et décembre dépassent 40 %, mais très peu d'appels y ont été passés. Mai, le mois le plus chargé, affiche le taux le plus faible (6 %) ;
- les étudiants (31 %) et les retraités (25 %) sont les catégories professionnelles les plus réceptives.

### 2.5 Le cas de la durée d'appel

![Graphique 4 – Durée de l'appel selon la souscription](figures/cell29.png)

*Graphique 4 – Durée de l'appel selon la souscription*

Les appels qui se terminent par une souscription durent nettement plus longtemps (médiane de 449 secondes contre 164). Cette variable a pourtant été écartée. La durée n'est connue qu'une fois l'appel terminé, c'est-à-dire au moment où l'on connaît déjà la réponse du client. L'utiliser reviendrait à introduire une fuite de données, puisque l'objectif est justement de choisir qui appeler avant de décrocher le téléphone.

## 3. Démarche de modélisation

Après suppression des doublons et de `duration`, il reste 41 176 clients. Ils ont été répartis en 75 % pour l'apprentissage (30 882) et 25 % pour le test (10 294), avec un découpage stratifié qui conserve 11,3 % de souscripteurs dans chaque partie. Dans un pipeline, les variables numériques sont standardisées et les variables qualitatives transformées en indicatrices (one-hot).

Six modèles ont été comparés par validation croisée à 5 plis. Pour compenser le faible nombre de souscripteurs, l'option `class_weight="balanced"` a été activée sur la régression logistique, l'arbre et la forêt : elle donne plus de poids aux souscripteurs pendant l'apprentissage.

| Modèle | Exactitude | Rappel | AUC moyenne |
|---|---|---|---|
| Gradient boosting | 0,900 | 0,250 | 0,798 |
| Forêt aléatoire | 0,866 | 0,585 | 0,790 |
| Régression logistique | 0,830 | 0,622 | 0,788 |
| Arbre de décision | 0,841 | 0,594 | 0,774 |
| K plus proches voisins (k = 15) | 0,897 | 0,263 | 0,754 |
| Référence (toujours « non ») | 0,887 | 0,000 | 0,500 |

![Graphique 5 – Comparaison des modèles](figures/cell46.png)

*Graphique 5 – Exactitude et AUC des modèles en validation croisée*

Ce tableau illustre bien le piège de l'exactitude. La référence, qui ne repère aucun souscripteur, obtient 88,7 %, soit plus que la régression logistique. Avec l'AUC, elle retombe à 0,5, le niveau du hasard, et les vrais modèles se situent entre 0,75 et 0,80. Le gradient boosting arrive en tête (0,798), de peu devant la forêt aléatoire et la régression logistique. Son rappel de 25 % paraît faible, mais il s'explique par le seuil de décision par défaut (0,5), que l'on ajuste plus loin.

## 4. Résultats du modèle retenu

### 4.1 Performances sur l'échantillon test

Le gradient boosting, réentraîné sur l'ensemble de l'échantillon d'apprentissage, obtient une **AUC de 0,815** sur les 10 294 clients du test.

![Graphique 6 – Matrice de confusion](figures/cell53.png)

*Graphique 6 – Matrice de confusion sur l'échantillon test (seuil de 0,5)*

Avec le seuil par défaut, le modèle repère 287 souscripteurs sur 1 160 (25 %), en ne proposant que 141 appels inutiles. Il passe en revanche à côté de 873 clients qui auraient souscrit. Le modèle est donc trop prudent : il ne répond « oui » que lorsqu'il est quasiment certain. Pour une campagne commerciale, une vente manquée coûte généralement plus cher qu'un appel sans suite, ce qui justifie de revoir le seuil.

### 4.2 Choix du seuil

| Seuil | Clients à appeler | Souscripteurs repérés | Rappel | Appels inutiles |
|---|---|---|---|---|
| 0,1 | 2 163 | 781 | 67 % | 1 382 |
| 0,2 | 1 488 | 694 | 60 % | 794 |
| 0,3 | 1 129 | 569 | 49 % | 560 |
| 0,4 | 669 | 393 | 34 % | 276 |
| 0,5 | 428 | 287 | 25 % | 141 |

En abaissant le seuil de 0,5 à 0,2, le rappel passe de 25 % à 60 % : on repère 694 souscripteurs au lieu de 287, en appelant 1 488 clients au lieu de 428. Le nombre d'appels inutiles augmente (794 au lieu de 141), mais reste très inférieur à celui d'une campagne menée sans ciblage. Le bon seuil dépend de la capacité d'appel de la banque et de la marge dégagée par chaque souscription.

### 4.3 Courbe de gain

![Graphique 7 – Courbe de gain](figures/cell61.png)

*Graphique 7 – Part des souscripteurs touchés selon la part de clients appelés*

C'est sans doute le résultat le plus directement utilisable. En classant les clients du plus probable au moins probable, puis en appelant seulement les **20 % du haut de la liste, on touche 66 % des souscripteurs**, contre 20 % avec des appels au hasard. En appelant la moitié des clients, on en touche 84 %. Concrètement, la banque pourrait réduire fortement son volume d'appels tout en conservant la majorité de ses ventes.

La courbe d'apprentissage du modèle (non reproduite ici) montre par ailleurs que le score de validation se stabilise autour de 0,80. Ajouter des clients n'améliorerait plus beaucoup le modèle.

## 5. Lecture du modèle

### 5.1 Variables déterminantes

![Graphique 8 – Importance des variables par permutation](figures/cell70.png)

*Graphique 8 – Importance des variables (baisse de l'AUC lorsque la variable est mélangée)*

L'importance a été mesurée par permutation : on mélange les valeurs d'une variable et on observe la baisse de l'AUC. Le nombre de salariés dans l'économie (`nr.employed`) ressort nettement en premier, suivi des autres indicateurs économiques (`cons.price.idx`, `euribor3m`, `cons.conf.idx`). Le type de contact et le mois de l'appel comptent aussi. Les caractéristiques personnelles (profession, éducation, crédits en cours) pèsent peu.

Le moment et le contexte de l'appel comptent donc davantage que le profil du client. Un exemple fictif le confirme. Un étudiant appelé sur son portable en octobre, avec un Euribor bas, a une probabilité de souscription estimée à 0,44, soit environ quatre fois la moyenne. Appelé en mai, dans une conjoncture où l'Euribor est élevé, ce même client tombe à 0,05.

### 5.2 Ce que change la durée d'appel

Pour mesurer l'effet de la fuite de données, le même modèle a été réentraîné en conservant `duration`. Son AUC passe de 0,815 à 0,950. Le gain n'est qu'apparent : en pratique, ce modèle ne pourrait pas servir à choisir qui appeler, puisque la durée n'existe qu'après l'appel. Un très bon score ne garantit donc pas un modèle utile. Il faut toujours vérifier que chaque variable sera réellement disponible au moment de la prédiction.

## 6. Limites

Plusieurs réserves s'imposent avant toute utilisation opérationnelle :

- **une période particulière** : les données couvrent 2008-2010, en pleine crise financière. Le poids des variables économiques reflète en partie cette conjoncture et pourrait être différent aujourd'hui ;
- **des variables économiques redondantes** : très corrélées entre elles, elles se partagent l'importance, ce qui rend leur lecture individuelle délicate. De même, `poutcome` paraît peu importante parce que `pdays` porte presque la même information ;
- **une capacité prédictive plafonnée** : une AUC d'environ 0,82 reste correcte sans être excellente, et la courbe d'apprentissage montre que plus de données n'y changeraient pas grand-chose ;
- **des coûts non chiffrés** : le seuil a été discuté sans connaître le coût d'un appel ni la marge d'un dépôt, ce qui empêche de déterminer le seuil optimal ;
- **des hyperparamètres non optimisés** : les modèles ont été utilisés avec des réglages simples, et un gradient boosting mieux paramétré pourrait gagner quelques points ;
- **un seul établissement** : les résultats concernent une banque portugaise et ne se transposent pas directement à un autre marché, par exemple marocain.

## 7. Conclusion

Cette étude montre qu'un modèle d'apprentissage automatique peut aider une banque à mieux cibler ses campagnes d'appels. Le gradient boosting, retenu pour sa meilleure AUC (0,815 sur l'échantillon test), permet de toucher les deux tiers des souscripteurs en n'appelant qu'un client sur cinq.

Le travail a aussi mis en évidence deux points de méthode importants. D'abord, sur des classes aussi déséquilibrées, l'exactitude est trompeuse et doit être remplacée par l'AUC et le rappel. Ensuite, la variable `duration` devait être exclue : elle améliore artificiellement le score mais n'est pas connue avant l'appel.

Sur le fond, la décision de souscrire dépend surtout du contexte économique et du moment de l'appel, plus que du profil du client. Un tel outil gagnerait à être réentraîné sur des données récentes et enrichi d'informations financières sur les clients.
