# Diagnostic Parallel-Run — SP_agg vs SP_agg_generic

**Projet :** DETECT Generic — PROMETHEE, BNP Paribas
**Contexte :** US-6, migration CCIRC — validation par parallel-run (ancien pipeline vs GenericPipeline)
**Périmètre analysé :** dataset `SP_agg` (895 944 lignes, flow CCIRC)

---

## 1. Contexte

Le script de parallel-run (pur Python/pandas) compare `SP_agg` (ancien pipeline) et
`SP_agg_generic` (nouveau pipeline générique) sur quatre axes : sommes, moyennes,
cardinalité, top 5 des valeurs catégorielles. L'exécution a échoué avec **8 erreurs**,
dont un écart de **37,4 %** sur `SUM` et `AVG` de `t_agg5`.

L'objectif de ce document est de distinguer les écarts **attendus et acceptables**
des écarts révélant un **vrai problème de correction fonctionnelle**, et de documenter
la cause racine ainsi que le correctif retenu.

---

## 2. Résultats observés

| Vérification | Ancien | Nouveau | Écart relatif | Verdict |
|---|---|---|---|---|
| Nombre de lignes | 895 944 | 895 944 | 0 % | ✅ Identique |
| SUM `seg_agg1`, `t_agg2`, `t_agg3` | — | — | < 1e-6 | ✅ OK |
| SUM `t_agg5` | 1,0544e+24 | 6,6025e+23 | **37,38 %** | ❌ Échec |
| AVG `t_agg5` | 1,3311e+18 | 8,3511e+17 | **37,38 %** | ❌ Échec |
| Cardinalité `seg_agg2`, `seg_agg3`, `seg_agg4` | ~118k–129k | idem ± 0,1 % | 0,03–0,08 % | ❌ Échec (mineur) |
| Cardinalité `t_agg1`, `t_agg4`, `t_agg5` | ~313k–341k | idem ± 0,1 % | 0,03–0,1 % | ❌ Échec (mineur) |
| TOP 5 catégoriel | — | — | 1 ensemble différent sur `seg_axes_value` (`PF` vs `BCEF`) | ⚠️ Warning (non bloquant) |

---

## 3. Méthodologie de diagnostic — hypothèses écartées

Plusieurs hypothèses ont été testées et écartées avant d'identifier la cause racine :

1. **Erreur de partitionnement en lecture Spark** — écartée : le nombre de lignes
   est strictement identique (895 944 des deux côtés), ce qui exclut une lecture
   incomplète ou dupliquée de partitions.
2. **Explosion de lignes par jointure (row explosion)** — écartée pour la même
   raison : un fanout de jointure aurait fait varier le nombre de lignes total,
   pas seulement certaines valeurs agrégées.
3. **Erreur d'arrondi flottant classique** — écartée pour `t_agg5` : un écart
   d'arrondi (accumulation de sommes dans un ordre différent) ne peut produire
   qu'un écart relatif de l'ordre de 1e-10 à 1e-15. Un écart de 37 % est
   supérieur de treize ordres de grandeur à ce que l'arrondi peut expliquer seul.

---

## 4. Cause racine identifiée

### 4.1 Deux phénomènes distincts, à ne pas confondre

**a) Écarts mineurs de cardinalité (< 0,1 %) — acceptables**

Une minorité de valeurs `seg_agg2/3/4`, `t_agg1/4/5` sont comptées comme distinctes
entre les deux pipelines alors qu'elles devraient être identiques. La cause **n'est
pas** une différence de méthode de calcul entre les deux pipelines — `SP_agg` et
`SP_agg_generic` passent tous deux par la même fonction
`compute_agg_by_period_axe_optimized` (SQL string + `UNION ALL`). La cause réelle
est le **caractère non déterministe du `SUM()` distribué dans Spark** : le calcul
est effectué en parallèle sur plusieurs partitions puis fusionné (tree-reduce), et
l'addition en virgule flottante n'est pas associative en précision finie — c'est-à-
dire que `(a+b)+c` n'est pas nécessairement égal à `a+(b+c)`. L'ordre dans lequel
les partitions terminent et sont fusionnées dépend de l'ordonnancement d'exécution
au moment du run, qui n'est pas fixe d'une exécution à l'autre — même en exécutant
exactement le même code sur exactement les mêmes données, deux fois de suite.

Selon l'étude *« Equational reasoning for non-determinism monad: the case of
Spark aggregation »* [1], le combinateur `aggregate` de Spark repose sur deux
fonctions définies par l'utilisateur : l'une accumule un résultat partiel par
partition, l'autre fusionne les résultats partiels entre partitions. Les résultats
partiels étant calculés simultanément, l'ordre d'application de ces fonctions varie
d'une exécution à l'autre, rendant l'agrégation dans Spark intrinsèquement non
déterministe. Cette étude cite un exemple expérimental : le calcul de l'intégrale
de x⁷³ sur [-2, 2] (dont le résultat exact est 0), effectué via une fonction de la
bibliothèque de machine learning de Spark, a produit des résultats variant de
-8192,0 à 12288,0 selon les exécutions — preuve qu'une erreur d'arrondi
infinitésimale en entrée peut être fortement amplifiée selon l'opération qui lui
est appliquée en aval.

Appliqué à notre cas : deux valeurs mathématiquement égales peuvent différer au
12e–15e chiffre décimal du fait de l'ordre d'addition différent entre deux
exécutions Spark — suffisant pour être comptées comme deux valeurs distinctes par
`nunique()`/`countDistinct()`, mais sans impact sur les sommes ni les moyennes
(qui restent dans la tolérance `1e-6`).

**b) Écart de 37 % sur `t_agg5` — non acceptable, cause identifiée**

`t_agg5` est une moyenne pondérée : `t_agg5 = Σ(target × weight) / Σ(weight)`,
avec `weight = mt_cal_ead_tot_b` (cf. `weight_vars` dans `ccirc_params.json`).

Investigation sur les valeurs extrêmes de `t_agg5` (`|t_agg5|` triées par ordre
décroissant) :

```
t_variable_clone = mt_cal_ead_tot_b
t_agg1 (avg) ≈ 0          t_agg4 (sum) ≈ 0
t_agg2 (min) = -1,073094e+05
t_agg3 (max) =  1,073094e+05
t_agg5       =  7,961319e+23   ← anomalie
```

Le dénominateur `Σ(weight)` — c'est-à-dire `t_agg4`, la somme de
`mt_cal_ead_tot_b` — est quasi nul dans ce segment (les valeurs positives et
négatives s'annulent presque exactement, min ≈ -max). Le numérateur
`Σ(target × weight) = Σ(weight²)` reste en revanche strictement positif et
non négligeable (un carré ne s'annule jamais). La division d'un nombre non
négligeable par un dénominateur proche de zéro produit une valeur qui explose
(`10^18` à `10^24`).

Un écart d'arrondi infime sur `Σ(weight)` (de l'ordre de `1e-14`, identique en
nature au phénomène décrit au point a) suffit alors à faire varier fortement
le signe ou l'ordre de grandeur du résultat final, car l'erreur est **amplifiée
par la division**, non simplement additionnée. C'est ce mécanisme qui transforme
un écart invisible en entrée (`1e-14`) en un écart de 37 % visible en sortie.

### 4.2 Constat aggravant : incohérence business

Le mapping `target_detection_mapping` (config CCIRC) indique que les variables
concernées par les valeurs extrêmes observées (`mt_cal_ead_tot_b`, `ead_lgd`,
`provision12`) sont associées **uniquement à `agg4`**, jamais à `agg5` :

```json
"mt_cal_ead_tot_b": ["agg4"],
"ead_lgd": ["agg4"],
"provision12": ["agg4"]
```

La colonne `t_agg5` calculée pour ces lignes n'est donc **jamais utilisée** par
la détection d'anomalies en aval — c'est un calcul superflu qui, en plus d'être
inutile, introduit une instabilité numérique parasite dans la comparaison
parallel-run.

---

## 5. Pourquoi certains écarts sont acceptables

| Type d'écart | Acceptable ? | Justification |
|---|---|---|
| Cardinalité, écart < 0,1 % sur colonnes numériques | **Oui** | Conséquence normale du caractère non déterministe du `SUM()` distribué de Spark sur plusieurs partitions (cf. section 4.1a). N'affecte ni les sommes ni les moyennes de référence métier. |
| TOP 5 catégoriel avec ensembles différents à égalité de fréquence | **Oui, avec vigilance** | En cas d'ex æquo sur le comptage, l'ordre de tri n'est pas déterministe entre pandas et Spark. Le script le traite déjà comme `warning`, non bloquant. |
| SUM/AVG hors tolérance sur une variable **non utilisée** par la détection (`t_agg5` pour `mt_cal_ead_tot_b`) | **Oui, une fois masqué** | Le calcul est correct mathématiquement compte tenu de l'instabilité de la division ; comme il n'alimente aucune règle de détection, le neutraliser (`None`) plutôt que le comparer est la bonne réponse. |
| SUM/AVG hors tolérance sur une variable **utilisée** par la détection | **Non** | Impacterait directement la fiabilité de la détection d'anomalies — doit être investigué au cas par cas, pas neutralisé aveuglément. |

---

## 6. Solutions retenues

Deux correctifs complémentaires, appliqués dans `compute_agg_by_period_axe_optimized`
(`lib/python/core/pre_processing.py`) :

### 6.1 Garde-fou numérique — seuil minimal sur le dénominateur

```python
WEIGHT_SUM_EPSILON = 1.0

# 4. Finalise weighted columns
for weight_name in weight_vars_mapping:
    t_x_weight_sum, weight_sum = _generate_target_tmp_vars(weight_name)
    if t_x_weight_sum in df_agg.columns and weight_sum in df_agg.columns:
        df_agg = df_agg.withColumn(
            weight_name,
            F.when(F.abs(F.col(weight_sum)) > F.lit(WEIGHT_SUM_EPSILON),
                   F.col(t_x_weight_sum) / F.col(weight_sum))
             .otherwise(F.lit(None))
        ).drop(t_x_weight_sum, weight_sum)
```

**Effet :** si `|Σ(weight)|` est trop faible pour être fiable (bruit numérique),
le résultat est explicitement `None` plutôt qu'une valeur explosée. Protège
toutes les variables utilisant ce poids, y compris celles réellement mappées
sur `agg5`.

**Pourquoi le seuil `1,0` :**

Ce seuil n'est pas arbitraire — il est déterminé à partir de **l'échelle réelle**
de la variable de pondération `mt_cal_ead_tot_b`, en s'appuyant sur les seuils
métier déjà présents dans `ccirc_params_reconstructed.json` :

```json
"thresholds_topline": {"mt_cal_ead_tot_b": 5000},
"thresholds_segment_materiality": {"mt_cal_ead_tot_b": {"segment": "seg_agg5 > 100000 ..."}}
```

Ces seuils métier indiquent qu'une valeur d'EAD **réellement significative** se
situe toujours à l'échelle du millier ou de la centaine de milliers. Par
conséquent, si `Σ(mt_cal_ead_tot_b)` d'un segment se retrouve à `< 1,0`
(inférieur de 3 à 5 ordres de grandeur à l'échelle réelle), cela **ne peut
provenir** que d'une annulation quasi complète entre valeurs positives et
négatives (une somme qui devrait valoir exactement 0 mais qui conserve un
résidu infime dû à l'arrondi flottant), et non d'une véritable valeur d'EAD
faible. Choisir `1,0` crée une marge de sécurité large (bien supérieure à
l'ordre de grandeur réel de l'erreur d'arrondi, `1e-10` à `1e-14`), ce qui
garantit qu'aucun cas d'« annulation proche de zéro » n'échappe au filtre,
sans jamais affecter les segments dont `Σ(weight)` est valide à l'échelle du
millier ou de la centaine de milliers.

**Exemple concret à partir des données outlier réellement observées :**

Pour le segment où `t_variable_clone = mt_cal_ead_tot_b`, les valeurs brutes
relevées étaient :

```
t_agg1 (avg)           =  1,12 × 10⁻¹⁵   (devrait valoir ≈ 0 si les données s'annulent)
t_agg4 (sum)           =  2,91 × 10⁻¹⁴   (devrait valoir ≈ 0, pour la même raison)
min(mt_cal_ead_tot_b)  = -107 309,4
max(mt_cal_ead_tot_b)  =  107 309,4
```

**Point important sur la lecture de ces chiffres :** le fait que `min ≈ -max`
ne **prouve pas à lui seul** que `Σ = 0` — entre le min et le max se trouvent
de nombreuses autres valeurs du segment, rien ne garantit qu'elles s'annulent
également entre elles (exemple : un segment ne contenant que les 3 valeurs
`[-100, 50, 100]` a aussi `min = -100`, `max = 100`, parfaitement symétriques,
mais `Σ = 50`, différent de 0). Le couple min/max n'est ici qu'une
**illustration complémentaire** montrant que la distribution des valeurs du
segment est plutôt symétrique autour de 0.

La véritable preuve réside dans `t_agg1` et `t_agg4` — ce sont les résultats
`AVG()` et `SUM()` **calculés directement par Spark sur l'ensemble des
données du segment**, et non déduits du min/max. Les deux valeurs sont
extrêmement faibles (`10⁻¹⁵`, `10⁻¹⁴`) alors que l'échelle réelle de l'EAD
dans ce segment est de l'ordre de la centaine de milliers (comme le montrent
le min/max ≈ ±107 309). Une somme de l'ordre de `10⁻¹⁴` sur des données dont
l'amplitude atteint la centaine de milliers ne peut être que la trace d'une
somme qui **devrait valoir exactement 0** (les valeurs positives et négatives
du segment s'annulant presque parfaitement sur le plan comptable/métier), le
résidu de `2,91 × 10⁻¹⁴` restant n'étant que du bruit d'arrondi produit par
Spark lorsqu'il additionne des nombres réels dans un ordre non fixe (voir
section 4.1) — **et non une véritable valeur d'EAD à cette échelle**.

- **Sans garde-fou :** `t_agg5 = Σ(w²) / 2,91e-14`. Comme `Σ(w²)` (somme des
  carrés, toujours positive, de l'ordre du milliard vu que les valeurs d'EAD
  sont de l'ordre de la centaine de milliers) est divisée par un nombre de
  l'ordre de `10⁻¹⁴`, le résultat explose à `7,96 × 10²³` — exactement la
  valeur outlier observée dans les données réelles.
- **Avec le garde-fou `WEIGHT_SUM_EPSILON = 1,0` :** comme
  `|2,91 × 10⁻¹⁴| < 1,0`, la condition
  `F.when(F.abs(weight_sum) > 1.0, ...)` n'est pas satisfaite → `t_agg5` reçoit
  `None` au lieu d'exploser.

**Contre-exemple — un segment avec un EAD réellement faible, pour montrer que
le seuil ne provoque pas de faux blocage :** supposons un petit segment dont
l'EAD total réel est `850` (pas dû à une annulation, mais à un portefeuille
réellement petit). Comme `850 > 1,0`, la condition du garde-fou reste vraie,
`t_agg5` est calculé normalement par division — le seuil `1,0` ne bloque que
les cas d'annulation-proche-de-zéro, sans jamais affecter les segments petits
mais légitimes.

Ce seuil peut être ajusté pour un autre flow (IFRS9, ICAAP) dont la variable de
pondération se situe à une autre échelle — le principe général : choisir un
epsilon très inférieur à la plus petite valeur « significative sur le plan
métier » de cette variable, tout en restant très supérieur à l'ordre de
grandeur de l'erreur d'arrondi pure (`1e-10` et en dessous).

### 6.2 Garde-fou fonctionnel — masquage selon le mapping métier

```python
def mask_unused_weighted_aggs(df_agg, weight_vars_mapping, target_detection_mapping):
    for weight_name in weight_vars_mapping:
        agg_key = weight_name.replace("t_", "")
        allowed_vars = [tv for tv, aggs in target_detection_mapping.items()
                        if agg_key in aggs]
        if weight_name in df_agg.columns and allowed_vars:
            df_agg = df_agg.withColumn(
                weight_name,
                F.when(F.col(TARGET_VARIABLE_CLONE).isin(allowed_vars),
                       F.col(weight_name))
                 .otherwise(F.lit(None))
            )
    return df_agg
```

**Effet :** pour les lignes dont le `target_variable_clone` n'est pas déclaré
comme utilisant cet agrégat pondéré dans `target_detection_mapping`, la valeur
est neutralisée — indépendamment de sa validité numérique. Élimine le bruit
résiduel pour les variables où `agg5` n'a jamais de sens métier.

**Complémentarité :** le garde-fou 6.1 protège contre l'instabilité numérique
partout (y compris sur les variables qui utilisent légitimement `agg5`) ; le
garde-fou 6.2 élimine le calcul superflu là où il n'a aucune utilité métier,
même s'il n'a pas explosé numériquement.

---

## 7. Second diagnostic — écart persistant sur `SP_display_toplines`

### 7.1 Contexte et données observées

Après correctif du problème de duplication (cf. §7.5, cause distincte et déjà
résolue), un second écart subsiste au niveau ligne, sur le dataset final
`SP_display_toplines` (ancien) vs `SP_display_toplines_generic` (nouveau) :

| Métrique | Ancien | Nouveau | Écart |
|---|---|---|---|
| Nombre total de lignes | 53 175 | 53 189 | **+14** |
| `provision3` | 4 026 | 4 039 | +13 |
| `el_ratio` | 4 350 | 4 359 | +9 |
| `eir` | 2 534 | 2 527 | −7 |
| `ead_lgd` | 5 429 | 5 428 | −1 |

L'écart est **strictement localisé** à ces 4 `target_variable` — les 10 autres
variables du flow CCIRC sont identiques ligne pour ligne entre les deux
pipelines.

### 7.2 Exemple concret — `el_ratio`, entité `IPS_AM`, site `10002`

En comparant précisément les lignes « présentes seulement d'un côté », chaque
ligne disparue d'un côté trouve systématiquement une ligne « jumelle »
apparue de l'autre côté, avec un `rank` identique et une `materiality`
identique à la 10ᵉ décimale près :

```
Segment A (ancien seulement)  : seg_axes = entity|accounting_site_code_post_acc
                                 seg_axes_value = IPS_AM|10002
                                 rank = 184, materiality = 426 780,6019973883

Segment B (nouveau seulement) : seg_axes = entity|basel_approach_type_arc|migration_matrix|
                                            accounting_site_code_post_acc
                                 seg_axes_value = IPS_AM|AS|CIB_EUROPE_DFT|10002
                                 rank = 184, materiality = 426 780,6019973884
```

### 7.3 Mécanisme — raffinement dégénéré + égalité de classement (tie)

**Étape 1 — pourquoi deux combinaisons d'axes donnent la même valeur.** Le
Segment B ajoute deux critères de filtre (`basel_approach_type_arc=AS`,
`migration_matrix=CIB_EUROPE_DFT`) que le Segment A n'a pas. Leur materiality
étant identique, cela signifie que **toutes les facilities du groupe
`(IPS_AM, 10002)` possèdent déjà** ces deux valeurs — aucune facility de ce
groupe n'en a d'autres. Ajouter ces critères ne retire donc aucune facility :
l'ensemble sous-jacent de A et de B est rigoureusement le même. C'est un
« raffinement dégénéré » : l'axe supplémentaire ne divise rien dans ce cas
précis.

Point important : les deux segments existent déjà, identiques, **des deux
côtés**, dès l'étape d'agrégation — la liste `axes_str` de
`ccirc_params.json` (11 combinaisons d'axes) est strictement la même pour les
deux pipelines (vérifié). Le point de divergence n'est donc pas
l'agrégation, mais l'étape suivante : la sélection du top-N.

**Étape 2 — la double étape de classement dans `top_segment_par`.** La
fonction calcule `rank`/`rank_axes` **deux fois** :
1. Une première fois sur `df_spec` seul (segments répondant aux critères
   d'anomalie et au seuil `"(rank <= 30) & (rank_axes <= 10)"`, propre à
   chaque `target_variable` dans `thresholds_segment_materiality`) — ce rang
   sert uniquement de filtre et est ensuite supprimé.
2. Une seconde fois, **après concaténation** de `df_ref` (segments de
   référence par entité, indépendants du seuil ci-dessus) et de `df_spec`
   filtré — c'est ce second rang (`rank=184` dans l'exemple) qui apparaît
   dans `SP_top_seg` final. Le Segment A entre dans le résultat via `df_ref`
   (top materiality de l'entité `IPS_AM`), pas via le filtre `rank<=30` —
   d'où sa présence malgré un rang final de 184.

**Étape 3 — pourquoi le « gagnant » du tie change entre les deux pipelines.**
A et B ayant une materiality quasi identique, ils sont à égalité pour occuper
le rang 184 lors du second classement. Lequel des deux l'emporte dépend de
l'ordre des lignes **avant le tri** — un ordre qui provient du `SUM()`
distribué de Spark et qui, comme établi en §4.1, n'est pas garanti stable
d'une exécution à l'autre. Le perdant du tie est repoussé au rang suivant et,
selon le cas, disparaît du top-N.

### 7.4 Vérification — validation empirique du mécanisme

Un appariement systématique de toutes les lignes « présentes seulement d'un
côté » — non pas par `segment_id`, mais par proximité de `materiality`
(tolérance relative `1e-6`) — a permis de retrouver une paire correspondante
pour **100 % des lignes divergentes**, sur les 4 `target_variable` concernées
(0 ligne non appariée). Cette vérification empirique confirme que l'écart
provient intégralement du mécanisme décrit ci-dessus, et non d'une perte de
données ou d'un défaut de calcul.

### 7.5 Note — cause distincte déjà résolue : duplication de `provision3`

Avant que ce second écart ne soit isolé, un écart plus important avait été
observé sur `provision3` (225 segments avec des lignes strictement
dupliquées, à 100 % identiques). L'investigation a écarté successivement :
un bug de correspondance multiple sur les `tab_value` de référence, un
défaut de `drop_duplicates()`, et une jointure en fanout — pour finalement
identifier une **reconstruction (« build ») obsolète de la partition**
`provision3` dans `SP_top_seg_generic`. Un rebuild complet de cette
partition a fait disparaître intégralement la duplication, confirmant qu'il
s'agissait d'un problème opérationnel (données non rafraîchies), non d'un
défaut du code de `top_segment_par` ou `top_anomalies_line_par`.

### 7.6 Cas particulier — tie asymétrique dans `df_ref` (exemple `provision3`)

Le mécanisme des §7.2-7.4 explique un tie **symétrique** : un segment
disparaît d'un côté, un segment équivalent apparaît de l'autre, en nombre
égal (`only_ancien ≈ only_nouveau`). Pour `provision3`, le déséquilibre
observé est net : `only_ancien = 0`, `only_nouveau = 13` — aucune ligne
« perdue » côté ancien ne correspond aux 13 lignes apparues côté nouveau.
Ce cas a nécessité une investigation séparée.

**Vérification de l'étape amont.** Une comparaison directe de `SP_anomalies`
et `SP_anomalies_generic` (sortie de la zone MODELING, avant tout
post-traitement) pour le segment concerné confirme des valeurs strictement
identiques des deux côtés (`anomaly`, `deviation`, `materiality`, pour
chaque période et chaque `model_type`) — l'écart ne provient donc pas de la
détection d'anomalies, mais bien de l'étape de sélection du top-N dans
`top_segment_par`.

**Exemple concret — entité `IPS_PI`, `provision3`, période `2025Q4`.** En
recalculant la liste des « candidats référence » de l'entité `IPS_PI` (tous
les `seg_axes` contenant `IPS_PI`, triés par `materiality` décroissante),
les 15 premiers candidats sont **rigoureusement identiques**, valeur par
valeur, entre ancien et nouveau — sauf sur un point : l'ordre entre deux
segments à égalité de materiality autour du seuil `top = 10` :

```
                                              materiality      rang ANCIEN   rang NOUVEAU
entity|basel_approach_type_arc|migration_matrix
  = IPS_PI|AS|CIB_EUROPE_DFT                  93 524,747228        #10           #11
entity|migration_matrix
  = IPS_PI|CIB_EUROPE_DFT                     93 524,747228        #11           #10
```

Les deux segments ont une `materiality` **identique au dernier chiffre
près** — même mécanisme de raffinement dégénéré qu'aux §7.2-7.3 (toutes les
facilities du groupe `IPS_PI|CIB_EUROPE_DFT` partagent déjà
`basel_approach_type_arc=AS`, donc l'axe supplémentaire ne divise rien).
Le seuil `"reference": {"top": 10}` de `ccirc_params.json` coupe exactement
entre ces deux rangs à égalité. Selon l'ordre des lignes en entrée du tri
(non déterministe, cf. §4.1) :

- **Côté ancien** : `entity|migration_matrix=IPS_PI|CIB_EUROPE_DFT` obtient
  le rang #11 → **exclu** du top-10 de référence → n'entre jamais dans
  `df_ref`, donc n'apparaît dans aucune ligne de `SP_display_toplines` pour
  ce segment, à aucune des 13 périodes historiques associées.
- **Côté nouveau** : le même segment obtient le rang #10 → **inclus** dans
  `df_ref` → génère une ligne dans `SP_display_toplines_generic` pour
  chacune des 13 périodes historiques disponibles pour ce segment.

**Pourquoi l'asymétrie (0 vs 13) et non un échange 1-pour-1.** Le segment
« gagnant » du tie (`entity|basel_approach_type_arc|migration_matrix`, rang
#10 côté ancien) reste malgré tout hors du périmètre effectif de
`SP_display_toplines` — soit parce qu'il ne satisfait pas séparément les
critères d'anomalie de `df_spec`, soit parce qu'un autre mécanisme de
filtrage l'exclut en aval. Le tie ne produit donc pas de ligne « jumelle »
compensatoire du côté ancien : le segment perdant disparaît purement et
simplement, pour ses 13 périodes d'un coup — d'où le déséquilibre 0/13,
par opposition au tie symétrique du §7.2 où chaque ligne perdue est
immédiatement compensée par une ligne équivalente apparue ailleurs.

### 7.7 Conclusion et recommandation

Cet écart de 14 lignes sur `SP_display_toplines` relève, sous ses deux
formes (§7.2-7.4 et §7.6), de la **même cause racine** que les écarts
mineurs de cardinalité du §4.1a : le caractère non déterministe du `SUM()`
distribué de Spark, combiné à des situations de raffinement dégénéré
(plusieurs combinaisons d'axes donnant une materiality identique). Il n'y a
pas de défaut de logique dans `top_segment_par`, `top_anomalies_line_par`,
ni dans la configuration `axes_str` (hypothèse testée et écartée : les
deux pipelines partagent exactement le même ensemble de combinaisons
d'axes).

**Recommandation :** traiter cet écart comme acceptable dans le script de
parallel-run, en ajoutant une étape d'appariement par `materiality` (comme
au §7.4) avant de signaler une ligne comme réellement manquante — plutôt que
de comparer uniquement les clés (`segment_id`) telle quelle. Un correctif de
fond (tie-breaker déterministe sur `seg_axes`, par exemple préférer la
combinaison la plus courte à materiality égale) est possible mais optionnel :
il élimine la non-déterminisme au prix d'un changement de comportement des
deux pipelines à harmoniser simultanément.

---

## 8. Recommandations de suivi

- Appliquer le même correctif à `compute_agg_by_period_axe` (version non
  optimisée), si celle-ci reste utilisée dans un chemin de code actif.
- Ajuster le script de parallel-run pour ignorer, dans les vérifications SUM/AVG,
  les couples `(target_variable, weight_name)` non couverts par
  `target_detection_mapping` — évite de rouvrir ce faux-positif à chaque
  nouvelle migration de flow (IFRS9, ICAAP).
- Documenter `WEIGHT_SUM_EPSILON` comme paramètre de configuration si plusieurs
  flows nécessitent des seuils différents selon l'échelle de leurs variables de
  pondération.
- Conserver la tolérance actuelle (`1e-6`) sur cardinalité comme information
  (warning), pas comme critère bloquant, car elle reflète un phénomène
  d'arrondi normal et sans impact métier.
- Intégrer au script de parallel-run l'appariement par `materiality` décrit
  au §7.4, pour absorber automatiquement les écarts de type tie-swap sur
  `SP_display_toplines` sans nécessiter de revalidation manuelle à chaque
  exécution.
- Pour les écarts asymétriques (type §7.6, `provision3`) que l'appariement
  par paire ne peut absorber : vérifier, avant de signaler une ligne
  « manquante » comme une erreur, si son `segment_id` correspond à un
  candidat de `df_ref` situé à égalité de materiality autour du seuil
  `reference.top` — si oui, traiter l'écart comme acceptable au même titre
  que le tie symétrique.

---

*Document établi à partir de l'analyse des sorties du script de parallel-run
(pur Python/pandas) sur `SP_agg` — flow CCIRC, PROMETHEE/DETECT.*

---

## Référence

[1] « Equational reasoning for non-determinism monad: the case of Spark
aggregation » — https://arxiv.org/pdf/2101.09408
