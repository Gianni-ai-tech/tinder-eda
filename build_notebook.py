"""Construit tinder_eda.ipynb (cellules markdown + code). Exécution : jupyter nbconvert --execute."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Tinder – Qu'est-ce qui donne envie d'un second rendez-vous ?

**Contexte.** Tinder constate une baisse du nombre de matchs. Pour comprendre ce qui rend deux personnes intéressées l'une par l'autre, l'équipe marketing s'appuie sur une expérience de *speed dating* (2002-2004) : rendez-vous de 4 minutes, puis chacun dit secrètement s'il veut revoir l'autre (`dec`) et note son partenaire sur 6 attributs (attirance, sincérité, intelligence, humour, ambition, intérêts communs).

**Question.** Qu'est-ce qui fait dire « oui » à un second rendez-vous ?

**Plan.**
1. Données et qualité
2. Vue d'ensemble : combien de oui, combien de matchs
3. Les notes données à ceux qui reçoivent un oui
4. Le poids réel de chaque attribut, par genre
5. Ce que les gens disent chercher et ce qu'ils choisissent vraiment
6. Intérêts communs ou même origine ?
7. Les gens savent-ils ce qu'ils valent ?
8. Vaut-il mieux être le premier ou le dernier rendez-vous de la soirée ?
9. Conclusions pour Tinder
""")

code("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110

df = pd.read_csv("data/speed_dating.csv", encoding="latin-1")
GENRE = {0: "Femmes", 1: "Hommes"}
ATTR = ["attr", "sinc", "intel", "fun", "amb", "shar"]
NOMS = {"attr": "Attirance", "sinc": "Sincérité", "intel": "Intelligence",
        "fun": "Humour", "amb": "Ambition", "shar": "Intérêts communs"}
df.shape
""")

md("""
## 1. Données et qualité

Chaque ligne est **un rendez-vous vu par un participant** : `iid` est la personne, `pid` son partenaire, `dec` sa décision, `dec_o` celle du partenaire, `match` vaut 1 si les deux ont dit oui. Chaque rendez-vous apparaît donc deux fois, une fois de chaque côté.
""")

code("""
print(f"{len(df):,} lignes, {df.iid.nunique()} participants, {df.wave.nunique()} soirées (waves)")
print(f"Âge médian : {df.drop_duplicates('iid').age.median():.0f} ans")
manquants = df[ATTR + ["dec", "samerace", "imprace", "attr3_1", "order"]].isna().mean().mul(100).round(1)
manquants.rename("% manquant").to_frame().T
""")

md("""
Les notes sont presque complètes, sauf **ambition (8,5 %)** et **intérêts communs (12,7 %)**, que certains participants n'ont pas remplies. Les analyses qui utilisent ces notes ignorent les lignes incomplètes, plutôt que d'inventer des valeurs.
""")

md("## 2. Vue d'ensemble")

code("""
vue = df.groupby(df.gender.map(GENRE)).agg(taux_de_oui=("dec", "mean"), rendez_vous=("dec", "size"))
print(f"Taux de oui global : {df.dec.mean():.1%} — taux de match : {df.match.mean():.1%}")

fig, ax = plt.subplots(figsize=(6, 3.5))
sns.barplot(x=vue.index, y=vue.taux_de_oui * 100, ax=ax, hue=vue.index, legend=False)
ax.set(ylabel="% de « oui »", xlabel="", title="Qui dit oui au second rendez-vous ?")
for i, v in enumerate(vue.taux_de_oui * 100):
    ax.text(i, v + 1, f"{v:.0f} %", ha="center")
plt.show()
vue
""")

md("""
**Lecture.** 42 % des rendez-vous se terminent par un « oui », mais seuls **16,5 % aboutissent à un match** : il faut que les deux soient d'accord. Les hommes disent oui plus souvent (47 %) que les femmes (37 %) : **les femmes sont plus sélectives**, ce sont souvent elles qui décident du match.
""")

md("## 3. Les notes données à ceux qui reçoivent un oui")

code("""
moy = df.groupby("dec")[ATTR].mean().T.rename(index=NOMS, columns={0: "Non", 1: "Oui"})
ax = moy.plot.bar(figsize=(8, 4), rot=0, color=["#c44e52", "#55a868"])
ax.set(ylabel="Note moyenne (sur 10)", title="Note donnée au partenaire selon la décision")
ax.legend(title="Décision")
plt.show()
moy.assign(écart=moy.Oui - moy.Non).round(2).sort_values("écart", ascending=False)
""")

md("""
**Lecture.** Les partenaires qui reçoivent un oui ont de meilleures notes sur **tous** les attributs : c'est un effet de halo, quand on apprécie quelqu'un on le note mieux partout. Mais les écarts ne sont pas égaux : **attirance (+1,9 point), intérêts communs (+1,7) et humour (+1,6)** séparent le plus les oui des non ; sincérité, intelligence et ambition beaucoup moins (+0,7).
""")

code("""
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
for ax, a in zip(axes, ["attr", "shar", "fun"]):
    taux = df.dropna(subset=[a]).groupby(df[a].round())["dec"].mean() * 100
    ax.plot(taux.index, taux.values, marker="o")
    ax.set(title=NOMS[a], xlabel="Note donnée (arrondie)", ylabel="% de oui")
plt.suptitle("Probabilité de dire oui selon la note donnée", y=1.03)
plt.show()
""")

md("""
**Lecture.** La relation est presque linéaire : en dessous de 4/10 d'attirance, quasiment personne ne dit oui ; au-dessus de 8/10, la grande majorité dit oui.
""")

md("""
## 4. Le poids réel de chaque attribut, par genre

Les attributs sont corrélés entre eux (effet de halo), donc la comparaison des moyennes ne dit pas lequel compte *à autres choses égales*. Une **régression logistique sur les notes standardisées** donne ce poids : plus le coefficient est grand, plus l'attribut pèse dans la décision, toutes les autres notes étant fixées.
""")

code("""
def poids(d):
    d = d.dropna(subset=ATTR + ["dec"])
    X = StandardScaler().fit_transform(d[ATTR])
    m = LogisticRegression().fit(X, d.dec)
    return pd.Series(m.coef_[0], index=[NOMS[a] for a in ATTR])

revele = pd.DataFrame({GENRE[g]: poids(df[df.gender == g]) for g in (0, 1)})
ax = revele.sort_values("Hommes").plot.barh(figsize=(7, 4))
ax.axvline(0, color="black", lw=0.8)
ax.set(xlabel="Coefficient (notes standardisées)", title="Poids de chaque attribut dans le « oui »")
plt.show()
revele.round(2)
""")

code("""
corr = df[ATTR].rename(columns=NOMS).corr()
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1)
plt.title("Corrélation entre les notes données")
plt.show()
""")

md("""
**Lecture.**
- **L'attirance est le premier critère pour les deux genres**, nettement plus pour les hommes (1,26) que pour les femmes (0,81).
- Viennent ensuite **les intérêts communs et l'humour**, de poids comparable chez les deux genres.
- L'intelligence compte un peu pour les femmes (0,21), pas pour les hommes (−0,05).
- **Sincérité et ambition ont un poids négatif** une fois les autres notes fixées. Cela ne veut pas dire qu'elles repoussent : ces notes sont très corrélées aux autres (sincérité–intelligence 0,66, intelligence–ambition 0,63), et à attirance égale, une note de sincérité élevée signale plutôt un partenaire « sympa mais pas plus ». Ce sont **les attributs les moins décisifs**, chez les hommes comme chez les femmes ; l'ambition pèse un peu plus négativement pour les femmes que pour les hommes, contrairement à l'intuition.
""")

md("""
## 5. Ce que les gens disent chercher et ce qu'ils choisissent vraiment

Avant la soirée, chaque participant répartit **100 points** entre les 6 attributs selon ce qu'il recherche (`attr1_1` à `shar1_1`). On compare cette importance **déclarée** au poids **révélé** par les décisions (section 4).
""")

code("""
P = ["attr1_1", "sinc1_1", "intel1_1", "fun1_1", "amb1_1", "shar1_1"]
pers = df.drop_duplicates("iid")[["iid", "gender"] + P].dropna()
pers[P] = pers[P].div(pers[P].sum(axis=1), axis=0) * 100   # quelques participants ne totalisent pas 100
declare = pers.groupby(pers.gender.map(GENRE))[P].mean().T
declare.index = [NOMS[p.split("1_1")[0]] for p in P]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
declare.plot.bar(ax=axes[0], rot=30, title="Importance déclarée (points sur 100)").legend(title="")
revele.plot.bar(ax=axes[1], rot=30, title="Poids révélé par les décisions")
axes[1].axhline(0, color="black", lw=0.8)
plt.tight_layout()
plt.show()
declare.round(1)
""")

md("""
**Lecture.**
- **Les femmes disent chercher d'abord l'intelligence (21 points) et la sincérité (18), et placent l'attirance en 3e position (18).** Dans leurs décisions, l'attirance pèse pourtant 4 fois plus que l'intelligence, et la sincérité joue négativement.
- **Les hommes assument davantage l'attirance** (27 points, 1er critère), et leurs décisions le confirment : c'est leur critère dominant.
- **Les intérêts communs sont sous-estimés par tous** : 11 à 13 points déclarés (dernier ou avant-dernier critère), alors qu'ils sont le 2e critère réel.

Conclusion : on sous-estime le poids réel de l'attirance et des intérêts communs, et on surestime celui de la sincérité et de l'ambition.
""")

md("## 6. Intérêts communs ou même origine ?")

code("""
d = df.dropna(subset=["shar"]).copy()
d["tranche"] = pd.cut(d.shar, [0, 3, 5, 7, 10], include_lowest=True, labels=["0-3", "4-5", "6-7", "8-10"])
t_shar = d.groupby("tranche", observed=True).dec.mean() * 100

df["race_importante"] = np.where(df.imprace >= 7, "Origine importante (≥ 7/10)", "Origine peu importante")
t_race = df.groupby(["race_importante", "samerace"]).dec.mean().unstack() * 100
t_race.columns = ["Origines différentes", "Même origine"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
t_shar.plot.bar(ax=axes[0], rot=0, color="#4c72b0", title="% de oui selon les intérêts communs perçus")
t_race.plot.bar(ax=axes[1], rot=0, title="% de oui selon l'origine")
axes[0].set(xlabel="Note « intérêts communs »", ylabel="% de oui"); axes[1].set(xlabel="")
plt.tight_layout()
plt.show()
t_shar.round(1), t_race.round(1)
""")

code("""
dd = df.dropna(subset=["shar", "samerace", "dec"])
m = LogisticRegression().fit(StandardScaler().fit_transform(dd[["shar", "samerace"]]), dd.dec)
print("Coefficients standardisés : intérêts communs = %.2f, même origine = %.2f" % tuple(m.coef_[0]))
""")

md("""
**Lecture.** Les intérêts communs perçus font passer le taux de oui de **14 % à 72 %**. Partager la même origine ne change presque rien en moyenne (41 % contre 43 %), avec un coefficient 100 fois plus faible.

Seule exception : les participants qui déclarent l'origine **importante** (≥ 7/10) disent beaucoup moins oui à une personne d'une autre origine (26 % contre 41 %). C'est une minorité, mais elle compte pour un algorithme de recommandation.

**Les intérêts communs sont donc bien plus importants que l'origine commune.**
""")

md("""
## 7. Les gens savent-ils ce qu'ils valent ?

Chaque participant se note lui-même (`attr3_1`, etc.). On compare cette auto-évaluation à la **moyenne des notes reçues** de ses partenaires (`attr_o`, etc.).
""")

code("""
auto = df.drop_duplicates("iid").set_index("iid")[["attr3_1", "fun3_1", "intel3_1", "sinc3_1", "amb3_1"]]
recu = df.groupby("iid")[["attr_o", "fun_o", "intel_o", "sinc_o", "amb_o"]].mean()
j = auto.join(recu).dropna()

lignes = []
for a, b, nom in [("attr3_1", "attr_o", "Attirance"), ("fun3_1", "fun_o", "Humour"), ("intel3_1", "intel_o", "Intelligence"),
                  ("sinc3_1", "sinc_o", "Sincérité"), ("amb3_1", "amb_o", "Ambition")]:
    r, p = stats.pearsonr(j[a], j[b])
    lignes.append({"attribut": nom, "auto-note": j[a].mean(), "note reçue": j[b].mean(),
                   "% qui se surestiment": (j[a] > j[b]).mean() * 100, "corrélation r": r, "p-valeur": p})
auto_eval = pd.DataFrame(lignes).set_index("attribut")

fig, ax = plt.subplots(figsize=(5.5, 5))
sns.regplot(x=j.attr_o, y=j.attr3_1, ax=ax, scatter_kws={"alpha": 0.4})
ax.plot([2, 10], [2, 10], "--", color="grey", label="Auto-note = note reçue")
ax.set(xlabel="Attirance : note moyenne reçue", ylabel="Attirance : auto-note", title="Se voit-on comme les autres nous voient ?")
ax.legend()
plt.show()
auto_eval.round(2)
""")

md("""
**Lecture.** Non, pas vraiment.
- **7 à 8 personnes sur 10 se surestiment**, en moyenne de 0,8 à 1,3 point selon l'attribut.
- Le lien entre l'auto-note et la note reçue est **faible pour l'attirance et l'humour (r ≈ 0,3)**, et **nul pour l'intelligence et la sincérité** (non significatif).
- Autrement dit, on a une vague idée de son attirance, mais aucune de la façon dont les autres perçoivent sa sincérité ou son intelligence.
""")

md("""
## 8. Premier ou dernier rendez-vous de la soirée ?

`order` donne le rang du rendez-vous dans la soirée. On compare le taux de oui reçu par le **premier** et par le **dernier** rendez-vous de chaque participant, et on regarde la tendance sur toute la soirée.
""")

code("""
dernier = df.groupby(["iid", "wave"]).order.transform("max")
prem, der = df[df.order == 1], df[df.order == dernier]
table = [[prem.dec_o.sum(), len(prem) - prem.dec_o.sum()], [der.dec_o.sum(), len(der) - der.dec_o.sum()]]
p_val = stats.chi2_contingency(table)[1]
print(f"Premier rendez-vous : {prem.dec_o.mean():.1%} de oui — dernier : {der.dec_o.mean():.1%} — test du χ² : p = {p_val:.2f}")

df["position"] = (df.order - 1) / (dernier - 1).replace(0, 1)
df["moment"] = pd.cut(df.position, [-0.01, 0.2, 0.4, 0.6, 0.8, 1], labels=["0-20 %", "20-40 %", "40-60 %", "60-80 %", "80-100 %"])
t_pos = df.groupby("moment", observed=True).dec_o.mean() * 100
ax = t_pos.plot(marker="o", figsize=(7, 3.5), ylim=(30, 55))
ax.set(xlabel="Moment de la soirée", ylabel="% de oui reçus", title="Le moment du rendez-vous change-t-il la réponse ?")
plt.show()
""")

md("""
**Lecture.** Le premier rendez-vous reçoit un peu plus de oui (49 %) que le dernier (45 %), mais **l'écart n'est pas significatif** (p ≈ 0,23, sur 551 participants). Sur l'ensemble de la soirée, le taux reste entre 40 et 43 %. **Le moment du rendez-vous n'a pas d'effet démontrable** : rien à exploiter de ce côté.
""")

md("""
## 9. Conclusions pour Tinder

| Constat | Chiffre clé | Piste pour Tinder |
| --- | --- | --- |
| L'attirance est le 1er critère réel, pour les deux genres | coefficient 1,26 (hommes), 0,81 (femmes) | Mettre les photos au centre du profil, aider à les choisir |
| Les intérêts communs sont le 2e critère réel, mais sous-estimés | 14 % → 72 % de oui | Afficher les centres d'intérêt partagés dès le profil, les pondérer dans l'algorithme |
| L'humour compte autant que les intérêts communs | coefficient ≈ 0,5 | Encourager une bio qui montre la personnalité |
| Ce qu'on déclare ≠ ce qu'on choisit | femmes : intelligence 1re déclarée, 4e réelle | Apprendre les préférences des swipes, pas du questionnaire |
| L'origine commune ne joue presque pas | 41 % contre 43 % | Ne pas la mettre en avant, sauf pour qui la déclare importante |
| Les femmes sont plus sélectives | 37 % de oui contre 47 % | Optimiser d'abord la satisfaction des femmes, qui décident du match |
| Le moment du rendez-vous ne compte pas | p ≈ 0,23 | Pas de levier ici |

**Limites.** Données de 2002-2004, étudiants de Columbia (âge médian 26 ans), rendez-vous en face à face : les résultats sont à valider sur les données réelles de l'application. Les notes sont subjectives et influencées par l'impression générale (effet de halo).
""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "tinder_eda.ipynb")
print("ok", len(C), "cellules")
