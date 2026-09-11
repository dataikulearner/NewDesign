import dataiku
from dataiku import spark as dkuspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType, StringType

spark = SparkSession.builder.getOrCreate()

dataset_old = dataiku.Dataset("SP_agg")
dataset_new = dataiku.Dataset("SP_agg_generic")

df_old = dkuspark.get_dataframe(spark, dataset_old)
df_new = dkuspark.get_dataframe(spark, dataset_new)

errors = []
warnings = []


# =============================================================================
# FONCTION 1 & 2 : Comparaison des SOMMES et MOYENNES (colonnes numériques)
# En un seul job Spark par dataframe (un seul .agg() avec toutes les expressions)
# =============================================================================

def compare_sums_and_averages(df_old, df_new, numeric_cols, tolerance=1e-6):
    """Compare somme totale et moyenne de chaque colonne numérique, en un seul
    passage sur chaque dataframe (agg() unique, pas un job par colonne)."""
    print("\n=== 1. Comparaison des SOMMES et MOYENNES ===")
    local_errors = []

    if not numeric_cols:
        return local_errors

    agg_exprs = []
    for col in numeric_cols:
        agg_exprs.append(F.sum(F.col(col)).alias("sum__" + col))
        agg_exprs.append(F.avg(F.col(col)).alias("avg__" + col))

    row_old = df_old.agg(*agg_exprs).collect()[0].asDict()
    row_new = df_new.agg(*agg_exprs).collect()[0].asDict()

    for col in numeric_cols:
        sum_old, sum_new = row_old["sum__" + col], row_new["sum__" + col]
        sum_old = 0.0 if sum_old is None else sum_old
        sum_new = 0.0 if sum_new is None else sum_new
        diff = abs(sum_old - sum_new)
        rel_diff = diff / abs(sum_old) if sum_old != 0 else diff

        if rel_diff >= tolerance:
            msg = "  \u274c SUM {}: ancien={:,.2f}, nouveau={:,.2f}, écart relatif={:.2%}".format(
                col, sum_old, sum_new, rel_diff)
            print(msg)
            local_errors.append("SUM '{}': écart relatif {:.2%}".format(col, rel_diff))
        else:
            print("  \u2705 SUM {}: {:,.2f} (OK)".format(col, sum_old))

        avg_old, avg_new = row_old["avg__" + col], row_new["avg__" + col]
        avg_old = 0.0 if avg_old is None else avg_old
        avg_new = 0.0 if avg_new is None else avg_new
        diff = abs(avg_old - avg_new)
        rel_diff = diff / abs(avg_old) if avg_old != 0 else diff

        if rel_diff >= tolerance:
            msg = "  \u274c AVG {}: ancien={:,.4f}, nouveau={:,.4f}, écart relatif={:.2%}".format(
                col, avg_old, avg_new, rel_diff)
            print(msg)
            local_errors.append("AVG '{}': écart relatif {:.2%}".format(col, rel_diff))
        else:
            print("  \u2705 AVG {}: {:,.4f} (OK)".format(col, avg_old))

    return local_errors


# =============================================================================
# FONCTION 3 : Comparaison de la CARDINALITÉ (nombre de valeurs distinctes)
# =============================================================================

def compare_cardinality(df_old, df_new, all_cols):
    """Compare le nombre de valeurs distinctes par colonne, en un seul .agg()
    par dataframe via countDistinct."""
    print("\n=== 3. Comparaison de la CARDINALITÉ ===")
    local_errors = []

    agg_exprs = [F.countDistinct(F.col(c)).alias(c) for c in all_cols]
    row_old = df_old.agg(*agg_exprs).collect()[0].asDict()
    row_new = df_new.agg(*agg_exprs).collect()[0].asDict()

    for col in all_cols:
        card_old, card_new = row_old[col], row_new[col]
        if card_old != card_new:
            msg = "  \u274c {}: ancien={} valeurs distinctes, nouveau={} valeurs distinctes".format(
                col, card_old, card_new)
            print(msg)
            local_errors.append("CARDINALITY '{}': {} vs {}".format(col, card_old, card_new))
        else:
            print("  \u2705 {}: {} valeurs distinctes (OK)".format(col, card_old))

    return local_errors


# =============================================================================
# FONCTION 4 : Comparaison du TOP 5 (colonnes catégorielles / texte)
# =============================================================================

def compare_top5(df_old, df_new, categorical_cols, top_n=5):
    """Compare l'ensemble des N valeurs les plus fréquentes pour les colonnes
    catégorielles. WARNING (non bloquant) si les ensembles diffèrent — l'ordre
    exact peut varier en cas d'égalité (ties)."""
    print("\n=== 4. Comparaison du TOP {} (colonnes catégorielles) ===".format(top_n))
    local_warnings = []

    for col in categorical_cols:
        top_old_rows = (df_old.groupBy(col).count()
                        .orderBy(F.desc("count"))
                        .limit(top_n).select(col).collect())
        top_new_rows = (df_new.groupBy(col).count()
                        .orderBy(F.desc("count"))
                        .limit(top_n).select(col).collect())
        top_old = set(r[col] for r in top_old_rows)
        top_new = set(r[col] for r in top_new_rows)

        if top_old != top_new:
            only_old = top_old - top_new
            only_new = top_new - top_old
            msg = "  \u26a0\ufe0f {}: ensembles différents — seulement ancien={}, seulement nouveau={}".format(
                col, only_old, only_new)
            print(msg)
            local_warnings.append("TOP{} '{}': ensembles différents".format(top_n, col))
        else:
            print("  \u2705 {}: top {} identique".format(col, top_n))

    return local_warnings


# =============================================================================
# EXÉCUTION — appel des 4 fonctions + vérifications existantes (shape, colonnes)
# =============================================================================

print("=== 0. Vérifications de base ===")

count_old, count_new = df_old.count(), df_new.count()
if count_old != count_new:
    errors.append("Nombre de lignes différent : ancien={}, nouveau={}".format(count_old, count_new))
else:
    print("  \u2705 Nombre de lignes identique : {:,}".format(count_old))

cols_old, cols_new = set(df_old.columns), set(df_new.columns)
missing_in_new = cols_old - cols_new
if missing_in_new:
    errors.append("Colonnes MANQUANTES dans la nouvelle version : {}".format(missing_in_new))
extra_in_new = cols_new - cols_old
if extra_in_new:
    print("  \u2139\ufe0f Colonnes supplémentaires (attendu, via ColumnRegistry) : {}".format(extra_in_new))

common_cols = sorted(cols_old & cols_new)

if not errors:
    schema_map = {f.name: f.dataType for f in df_old.schema.fields}
    numeric_cols = [c for c in common_cols if isinstance(schema_map.get(c), NumericType)]
    categorical_cols = [c for c in common_cols if isinstance(schema_map.get(c), StringType)]

    errors.extend(compare_sums_and_averages(df_old, df_new, numeric_cols))
    errors.extend(compare_cardinality(df_old, df_new, common_cols))
    warnings.extend(compare_top5(df_old, df_new, categorical_cols))

# =============================================================================
# RÉSULTAT FINAL
# =============================================================================

print("\n" + "=" * 60)
if warnings:
    print("\u26a0\ufe0f {} avertissement(s) (non bloquant) :".format(len(warnings)))
    for w in warnings:
        print("  - {}".format(w))

if errors:
    error_msg = "\u274c PARALLEL-RUN ÉCHOUÉ ({} erreur(s)) :\n".format(len(errors)) + "\n".join(
        "  - {}".format(e) for e in errors)
    print(error_msg)
    raise Exception(error_msg)
else:
    print("\u2705 PARALLEL-RUN RÉUSSI — toutes les vérifications sont passées")
