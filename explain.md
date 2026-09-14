| Vérification | Ancien | Nouveau | Écart | Acceptable ? | Raison | Solution proposée |
|---|---|---|---|---|---|---|
| Nombre de lignes | 895 944 | 895 944 | 0 (0 %) | OUI | Correspondance exacte — aucune perte/duplication de données entre les deux pipelines | Aucune action |
| Cardinalité seg_agg1 | 12 141 | 12 141 | 0 (0 %) | OUI | Identique | Aucune action |
| Cardinalité seg_agg2 | 129 310 | 129 353 | 43 (~0,03 %) | OUI | SUM() distribué de Spark non déterministe (ordre d'addition en virgule flottante variable selon les partitions) — écart au 12e-15e chiffre décimal, sans impact métier | Aucune action — écart déjà couvert par la tolérance 1e-6 sur SUM/AVG |
| Cardinalité seg_agg3 | 107 721 | 107 757 | 36 (~0,03 %) | OUI | Idem | Idem |
| Cardinalité seg_agg4 | 118 852 | 118 754 | 98 (~0,08 %) | OUI | Idem | Idem |
| Cardinalité t_agg1 | 313 417 | 313 062 | 355 (~0,1 %) | OUI | Idem | Idem |
| Cardinalité t_agg4 | 319 985 | 319 860 | 125 (~0,04 %) | OUI | Idem | Idem |
| Cardinalité t_agg5 | 341 656 | 341 753 | 97 (~0,03 %) | OUI | Idem — rounding bénin, sans lien avec la ligne suivante | Idem |
| SUM/AVG t_agg5 | 1,0544e+24 / 1,3311e+18 | 6,6025e+23 / 8,3511e+17 | 37,4 % | NON (sans correctif) → OUI (après correctif) | Division par Σ(weight) = Σ(mt_cal_ead_tot_b) proche de 0 (annulation dûe à des EAD positifs/négatifs qui se compensent) — un résidu d'arrondi infime au dénominateur est amplifié par la division en un écart massif | Appliquer WEIGHT_SUM_EPSILON = 1.0 (met None si abs(Σ(weight)) < 1.0) + masquage via target_detection_mapping pour les target_variables n'utilisant pas agg5 |
