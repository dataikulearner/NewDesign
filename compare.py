import dataiku
import pandas as pd
import numpy as np

df_old = dataiku.Dataset("SP_agg").get_dataframe()
df_new = dataiku.Dataset("SP_agg_generic").get_dataframe()

errors = []
warnings = []


# =============================================================================
# FONCTION 1 : Comparaison des SOMMES (colonnes numériques)
# =============================================================================

def compare_sums(df_old, df_new, common_cols, tolerance=1e-6):
    """Compare la somme totale de chaque colonne numérique."""
    print("\n=== 1. Comparaison des SOMMES ===")
    numeric_cols = [c for c in common_cols if pd.api.types.is_numeric_dtype(df_old[c])]
    local_errors = []

    for col in numeric_cols:
        sum_old = df_old[col].sum()
        sum_new = df_new[col].sum()
        diff = abs(sum_old - sum_new)
        rel_diff = diff / abs(sum_old) if sum_old != 0 else diff

        if rel_diff >= tolerance:
            msg = f"  ❌ {col}: ancien={sum_old:,.2f}, nouveau={sum_new:,.2f}, écart relatif={rel_diff:.2%}"
            print(msg)
            local_errors.append(f"SUM '{col}': écart relatif {rel_diff:.2%}")
        else:
            print(f"  ✅ {col}: {sum_old:,.2f} (OK)")

    return local_errors


# =============================================================================
# FONCTION 2 : Comparaison des MOYENNES (colonnes numériques)
# =============================================================================

def compare_averages(df_old, df_new, common_cols, tolerance=1e-6):
    """Compare la moyenne de chaque colonne numérique."""
    print("\n=== 2. Comparaison des MOYENNES ===")
    numeric_cols = [c for c in common_cols if pd.api.types.is_numeric_dtype(df_old[c])]
    local_errors = []

    for col in numeric_cols:
        avg_old = df_old[col].mean()
        avg_new = df_new[col].mean()
        diff = abs(avg_old - avg_new)
        rel_diff = diff / abs(avg_old) if avg_old != 0 else diff

        if rel_diff >= tolerance:
            msg = f"  ❌ {col}: ancien={avg_old:,.4f}, nouveau={avg_new:,.4f}, écart relatif={rel_diff:.2%}"
            print(msg)
            local_errors.append(f"AVG '{col}': écart relatif {rel_diff:.2%}")
        else:
            print(f"  ✅ {col}: {avg_old:,.4f} (OK)")

    return local_errors


# =============================================================================
# FONCTION 3 : Comparaison de la CARDINALITÉ (nombre de valeurs distinctes)
# =============================================================================

def compare_cardinality(df_old, df_new, common_cols):
    """Compare le nombre de valeurs distinctes (nunique) par colonne — toutes colonnes."""
    print("\n=== 3. Comparaison de la CARDINALITÉ ===")
    local_errors = []

    for col in common_cols:
        card_old = df_old[col].nunique()
        card_new = df_new[col].nunique()

        if card_old != card_new:
            msg = f"  ❌ {col}: ancien={card_old} valeurs distinctes, nouveau={card_new} valeurs distinctes"
            print(msg)
            local_errors.append(f"CARDINALITY '{col}': {card_old} vs {card_new}")
        else:
            print(f"  ✅ {col}: {card_old} valeurs distinctes (OK)")

    return local_errors


# =============================================================================
# FONCTION 4 : Comparaison du TOP 5 (colonnes catégorielles / texte)
# =============================================================================

def compare_top5(df_old, df_new, common_cols, top_n=5):
    """
    Compare le classement des N valeurs les plus fréquentes pour les
    colonnes catégorielles (string/object). Signale un WARNING (pas une
    erreur bloquante) si l'ENSEMBLE des top N diffère — l'ordre exact
    peut légitimement varier en cas d'égalité (ties).
    """
    print(f"\n=== 4. Comparaison du TOP {top_n} (colonnes catégorielles) ===")
    local_warnings = []
    categorical_cols = [c for c in common_cols if df_old[c].dtype == object]

    for col in categorical_cols:
        top_old = set(df_old[col].value_counts().head(top_n).index)
        top_new = set(df_new[col].value_counts().head(top_n).index)

        if top_old != top_new:
            only_old = top_old - top_new
            only_new = top_new - top_old
            msg = f"  ⚠️ {col}: ensembles différents — seulement ancien={only_old}, seulement nouveau={only_new}"
            print(msg)
            local_warnings.append(f"TOP{top_n} '{col}': ensembles différents")
        else:
            print(f"  ✅ {col}: top {top_n} identique")

    return local_warnings


# =============================================================================
# EXÉCUTION — appel des 4 fonctions + vérifications existantes (shape, colonnes)
# =============================================================================

print("=== 0. Vérifications de base ===")
if df_old.shape[0] != df_new.shape[0]:
    errors.append(f"Nombre de lignes différent : ancien={df_old.shape[0]}, nouveau={df_new.shape[0]}")
else:
    print(f"  ✅ Nombre de lignes identique : {df_old.shape[0]:,}")

cols_old, cols_new = set(df_old.columns), set(df_new.columns)
missing_in_new = cols_old - cols_new
if missing_in_new:
    errors.append(f"Colonnes MANQUANTES dans la nouvelle version : {missing_in_new}")
extra_in_new = cols_new - cols_old
if extra_in_new:
    print(f"  ℹ️ Colonnes supplémentaires (attendu, via ColumnRegistry) : {extra_in_new}")

common_cols = sorted(cols_old & cols_new)

if not errors:
    errors.extend(compare_sums(df_old, df_new, common_cols))
    errors.extend(compare_averages(df_old, df_new, common_cols))
    errors.extend(compare_cardinality(df_old, df_new, common_cols))
    warnings.extend(compare_top5(df_old, df_new, common_cols))

# =============================================================================
# RÉSULTAT FINAL
# =============================================================================

print("\n" + "=" * 60)
if warnings:
    print(f"⚠️ {len(warnings)} avertissement(s) (non bloquant) :")
    for w in warnings:
        print(f"  - {w}")

if errors:
    error_msg = f"❌ PARALLEL-RUN ÉCHOUÉ ({len(errors)} erreur(s)) :\n" + "\n".join(f"  - {e}" for e in errors)
    print(error_msg)
    raise Exception(error_msg)
else:
    print("✅ PARALLEL-RUN RÉUSSI — toutes les vérifications sont passées")
