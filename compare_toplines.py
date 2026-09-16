"""
Script de comparaison parallel-run — SP_display_toplines vs SP_display_toplines_generic

Implémente la recommandation 5 du diagnostic : au lieu de comparer les lignes
uniquement par clé stricte (segment_id / ctpr_id_generic / period_clone), les
lignes "présentes seulement d'un côté" sont d'abord appariées par proximité de
materiality (même mécanisme que la vérification empirique du §7.4). Les paires
ainsi appariées sont considérées comme un tie-swap acceptable (cf. §7.3) et ne
sont plus comptées comme des erreurs — seules les lignes réellement non
appariées restent signalées.
"""

import dataiku
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

LINE_KEY_COLS = ["t_variable_clone", "segment_id", "ctpr_id_generic", "period_clone"]
MATERIALITY_COL = "seg_materiality"          # ajuster selon le nom réel de la colonne
                                               # de materiality au niveau segment dans
                                               # SP_display_toplines (issue du renommage
                                               # "seg_" appliqué dans top_anomalies_line_par)
RANK_COL = "seg_rank"                         # idem, ajuster si le nom diffère
MATERIALITY_REL_TOLERANCE = 1e-6

df_old = dataiku.Dataset("SP_display_toplines").get_dataframe()
df_new = dataiku.Dataset("SP_display_toplines_generic").get_dataframe()

errors = []
warnings = []
info_matched = []


def find_only_in_side(df_old: pd.DataFrame, df_new: pd.DataFrame, target_var: str):
    """Retourne les lignes présentes uniquement côté ancien et uniquement côté
    nouveau, pour un target_variable donné, selon la clé stricte LINE_KEY_COLS."""
    d_old = df_old[df_old["t_variable_clone"] == target_var]
    d_new = df_new[df_new["t_variable_clone"] == target_var]

    key_old = d_old[LINE_KEY_COLS].drop_duplicates()
    key_new = d_new[LINE_KEY_COLS].drop_duplicates()

    only_old = pd.merge(key_old, key_new, on=LINE_KEY_COLS, how="left", indicator=True)
    only_old = only_old[only_old["_merge"] == "left_only"].drop(columns="_merge")

    only_new = pd.merge(key_new, key_old, on=LINE_KEY_COLS, how="left", indicator=True)
    only_new = only_new[only_new["_merge"] == "left_only"].drop(columns="_merge")

    # Récupère les colonnes complètes (dont materiality) pour ces lignes
    only_old_full = pd.merge(only_old, d_old, on=LINE_KEY_COLS, how="left")
    only_new_full = pd.merge(only_new, d_new, on=LINE_KEY_COLS, how="left")

    return only_old_full, only_new_full


def reconcile_by_materiality(only_old: pd.DataFrame, only_new: pd.DataFrame,
                              target_var: str, tolerance=MATERIALITY_REL_TOLERANCE):
    """Tente d'apparier chaque ligne 'only_old' avec une ligne 'only_new' de
    materiality proche (au niveau segment). Une paire appariée est considérée
    comme un tie-swap acceptable (cf. §7.3-7.4 du diagnostic), pas une erreur.

    Returns:
        (n_matched, unmatched_old, unmatched_new)
    """
    if MATERIALITY_COL not in only_old.columns or MATERIALITY_COL not in only_new.columns:
        print(f"  ⚠️ Colonne '{MATERIALITY_COL}' absente — appariement par materiality impossible, "
              f"comparaison stricte par clé conservée pour '{target_var}'")
        return 0, only_old, only_new

    used_new_idx = set()
    matched_pairs = []

    # Regroupe par segment_id pour éviter de réappliquer l'appariement ligne
    # par ligne (une seule materiality par segment, potentiellement plusieurs
    # lignes/facilities associées)
    seg_old = only_old[["segment_id", MATERIALITY_COL]].drop_duplicates()
    seg_new = only_new[["segment_id", MATERIALITY_COL]].drop_duplicates()

    for _, row_old in seg_old.iterrows():
        m_old = row_old[MATERIALITY_COL]
        candidates = seg_new[~seg_new.index.isin(used_new_idx)].copy()
        if candidates.empty:
            continue
        candidates["rel_diff"] = (candidates[MATERIALITY_COL] - m_old).abs() / max(abs(m_old), 1e-12)
        best = candidates.nsmallest(1, "rel_diff")
        if not best.empty and best.iloc[0]["rel_diff"] < tolerance:
            matched_pairs.append((row_old["segment_id"], best.iloc[0]["segment_id"]))
            used_new_idx.add(best.index[0])

    matched_old_segids = {p[0] for p in matched_pairs}
    matched_new_segids = {p[1] for p in matched_pairs}

    unmatched_old = only_old[~only_old["segment_id"].isin(matched_old_segids)]
    unmatched_new = only_new[~only_new["segment_id"].isin(matched_new_segids)]

    return len(matched_pairs), unmatched_old, unmatched_new


# =============================================================================
# EXÉCUTION
# =============================================================================

print("=== 0. Vérifications de base ===")
print(f"Nombre total de lignes : ancien={len(df_old):,}, nouveau={len(df_new):,}")

target_vars = sorted(set(df_old["t_variable_clone"].unique()) |
                     set(df_new["t_variable_clone"].unique()))

for tv in target_vars:
    n_old = (df_old["t_variable_clone"] == tv).sum()
    n_new = (df_new["t_variable_clone"] == tv).sum()

    only_old, only_new = find_only_in_side(df_old, df_new, tv)

    if only_old.empty and only_new.empty:
        continue  # rien à signaler pour ce target_variable

    print(f"\n--- {tv} : ancien={n_old}, nouveau={n_new} "
          f"(only_ancien={len(only_old)}, only_nouveau={len(only_new)}) ---")

    n_matched, unmatched_old, unmatched_new = reconcile_by_materiality(only_old, only_new, tv)

    if n_matched > 0:
        msg = (f"  ℹ️ {n_matched} paire(s) de segments appariée(s) par materiality "
               f"(tie-swap acceptable, cf. §7.3-7.4 du diagnostic)")
        print(msg)
        info_matched.append(f"{tv}: {n_matched} paire(s) tie-swap absorbée(s)")

    if not unmatched_old.empty:
        msg = f"  ❌ {len(unmatched_old)} ligne(s) 'ancien' SANS correspondance (rank/materiality) côté nouveau"
        print(msg)
        errors.append(f"{tv}: {len(unmatched_old)} ligne(s) manquante(s) non expliquée(s) par tie-swap")
        print(unmatched_old[["segment_id", RANK_COL, MATERIALITY_COL]].head(5).to_string(index=False)
              if RANK_COL in unmatched_old.columns and MATERIALITY_COL in unmatched_old.columns
              else unmatched_old.head(5).to_string(index=False))

    if not unmatched_new.empty:
        msg = f"  ❌ {len(unmatched_new)} ligne(s) 'nouveau' SANS correspondance (rank/materiality) côté ancien"
        print(msg)
        errors.append(f"{tv}: {len(unmatched_new)} ligne(s) en trop non expliquée(s) par tie-swap")
        print(unmatched_new[["segment_id", RANK_COL, MATERIALITY_COL]].head(5).to_string(index=False)
              if RANK_COL in unmatched_new.columns and MATERIALITY_COL in unmatched_new.columns
              else unmatched_new.head(5).to_string(index=False))


# =============================================================================
# RÉSULTAT FINAL
# =============================================================================

print("\n" + "=" * 60)
if info_matched:
    print(f"ℹ️ {len(info_matched)} target_variable(s) avec tie-swap absorbé(s) (non bloquant) :")
    for i in info_matched:
        print(f"  - {i}")

if errors:
    error_msg = f"❌ PARALLEL-RUN ÉCHOUÉ ({len(errors)} écart(s) réel(s), non expliqué(s) par tie-swap) :\n" + \
                "\n".join(f"  - {e}" for e in errors)
    print(error_msg)
    raise Exception(error_msg)
else:
    print("✅ PARALLEL-RUN RÉUSSI — tous les écarts observés s'expliquent par un "
          "tie-swap acceptable (cf. §7 du diagnostic)")
