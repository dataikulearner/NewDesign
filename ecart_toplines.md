écart persistant sur SP_display_toplines
7.1 Contexte et données observées

Après correctif du problème de duplication (cf. §7.5, cause distincte et déjà résolue), un second écart subsiste au niveau ligne, sur le dataset final SP_display_toplines (ancien) vs SP_display_toplines_generic (nouveau) :

Métrique	Ancien	Nouveau	Écart
Nombre total de lignes	53 175	53 189	+14
provision3	4 026	4 039	+13
el_ratio	4 350	4 359	+9
eir	2 534	2 527	−7
ead_lgd	5 429	5 428	−1

L'écart est strictement localisé à ces 4 target_variable — les 10 autres variables du flow CCIRC sont identiques ligne pour ligne entre les deux pipelines.

7.2 Exemple concret — el_ratio, entité IPS_AM, site 10002

En comparant précisément les lignes « présentes seulement d'un côté », chaque ligne disparue d'un côté trouve systématiquement une ligne « jumelle » apparue de l'autre côté, avec un rank identique et une materiality identique à la 10ᵉ décimale près :

Segment A (ancien seulement)  : seg_axes = entity|accounting_site_code_post_acc
                                 seg_axes_value = IPS_AM|10002
                                 rank = 184, materiality = 426 780,6019973883

Segment B (nouveau seulement) : seg_axes = entity|basel_approach_type_arc|migration_matrix|
                                            accounting_site_code_post_acc
                                 seg_axes_value = IPS_AM|AS|CIB_EUROPE_DFT|10002
                                 rank = 184, materiality = 426 780,6019973884
7.3 Mécanisme — raffinement dégénéré + égalité de classement (tie)

Étape 1 — pourquoi deux combinaisons d'axes donnent la même valeur. Le Segment B ajoute deux critères de filtre (basel_approach_type_arc=AS, migration_matrix=CIB_EUROPE_DFT) que le Segment A n'a pas. Leur materiality étant identique, cela signifie que toutes les facilities du groupe (IPS_AM, 10002) possèdent déjà ces deux valeurs — aucune facility de ce groupe n'en a d'autres. Ajouter ces critères ne retire donc aucune facility : l'ensemble sous-jacent de A et de B est rigoureusement le même. C'est un « raffinement dégénéré » : l'axe supplémentaire ne divise rien dans ce cas précis.

Point important : les deux segments existent déjà, identiques, des deux côtés, dès l'étape d'agrégation — la liste axes_str de ccirc_params.json (11 combinaisons d'axes) est strictement la même pour les deux pipelines (vérifié). Le point de divergence n'est donc pas l'agrégation, mais l'étape suivante : la sélection du top-N.

Étape 2 — la double étape de classement dans top_segment_par. La fonction calcule rank/rank_axes deux fois :

Une première fois sur df_spec seul (segments répondant aux critères d'anomalie et au seuil "(rank <= 30) & (rank_axes <= 10)", propre à chaque target_variable dans thresholds_segment_materiality) — ce rang sert uniquement de filtre et est ensuite supprimé.
Une seconde fois, après concaténation de df_ref (segments de référence par entité, indépendants du seuil ci-dessus) et de df_spec filtré — c'est ce second rang (rank=184 dans l'exemple) qui apparaît dans SP_top_seg final. Le Segment A entre dans le résultat via df_ref (top materiality de l'entité IPS_AM), pas via le filtre rank<=30 — d'où sa présence malgré un rang final de 184.

Étape 3 — pourquoi le « gagnant » du tie change entre les deux pipelines. A et B ayant une materiality quasi identique, ils sont à égalité pour occuper le rang 184 lors du second classement. Lequel des deux l'emporte dépend de l'ordre des lignes avant le tri — un ordre qui provient du SUM() distribué de Spark et qui, comme établi en §4.1, n'est pas garanti stable d'une exécution à l'autre. Le perdant du tie est repoussé au rang suivant et, selon le cas, disparaît du top-N.
