# Tinder – Qu'est-ce qui donne envie d'un second rendez-vous ? (Jedha CDSD, bloc 2)

Analyse exploratoire, descriptive et inférentielle d'une expérience de speed dating (8 378 rendez-vous, 551 participants, 2002-2004) pour comprendre ce qui fait dire « oui » à un second rendez-vous.

## Fichiers
- `tinder_eda.ipynb` : notebook exécuté (statistiques descriptives, visualisations, régressions logistiques, tests du χ² et corrélations, interprétations).
- `tinder_eda.html` : le même notebook, lisible dans un navigateur.
- `img/` : les 9 graphiques.
- `build_notebook.py` : génère le notebook.

## Données
`Speed Dating Data.csv` et sa notice, depuis le bucket du cours Jedha, à placer dans `data/speed_dating.csv`.

## Lancer
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn jupyter
jupyter nbconvert --to notebook --execute tinder_eda.ipynb
```

## Principaux résultats
- L'attirance est le 1er critère réel pour les deux genres, devant les intérêts communs et l'humour.
- Les femmes déclarent chercher d'abord l'intelligence, mais décident sur l'attirance.
- Les intérêts communs perçus font passer le taux de oui de 14 % à 72 % ; l'origine commune ne joue presque pas.
- 7 à 8 participants sur 10 se surestiment ; le rang du rendez-vous dans la soirée n'a pas d'effet significatif.
