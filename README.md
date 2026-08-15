# Fine-tuning de CamemBERT : Analyse de sentiment (Allociné)

Fine-tuning de [CamemBERT](https://huggingface.co/camembert-base) pour la classification de sentiment (positif/négatif) sur des critiques de films en français. Ce projet est la suite du [classificateur PyTorch from scratch](https://github.com/narsatGit/camembert-sentiment-allocine) : mêmes données, même split (`seed=42`), pour permettre une comparaison directe entre une approche from-scratch et une approche par fine-tuning d'un modèle pré-entraîné.

## Approche technique

- **Dataset** : [tblard/allocine](https://huggingface.co/datasets/tblard/allocine)
- **Sous-ensemble utilisé** : 2000 critiques d'entraînement, 500 critiques de test (`seed=42`, identique au projet from-scratch)
- **Modèle de base** : `camembert-base`, via `AutoModelForSequenceClassification` (`num_labels=2`)
- **Tokenisation** : tokenizer CamemBERT (SentencePiece/BPE, sous-mots), `truncation=True`, `padding="max_length"`, `max_length=128`
- **Entraînement** : `Trainer` de HuggingFace, 2 époques, batch size 16, `load_best_model_at_end=True` (sélection automatique selon la loss d'évaluation)

## Résultats

| Époque | Loss (eval) | Accuracy (eval) |
|---|---|---|
| 1 | 0.2342 | 91.40% |
| 2 | 0.2548 | 92.60% |

Le modèle **publié** correspond au checkpoint de l'**époque 1** (loss d'évaluation la plus basse : 0.2342 vs 0.2548), sélectionné automatiquement par `load_best_model_at_end=True` , son accuracy est donc **91.40%**, pas celle, légèrement supérieure, de l'époque 2.

## Comparaison avec l'approche from-scratch (modèle PyTorch simple)

| Approche | Accuracy (test, 500 exemples) | Temps d'entraînement (CPU) |
|---|---|---|
| PyTorch from scratch (Bag-of-Words) | 85.80% | Quelques secondes |
| CamemBERT fine-tuné (modèle publié) | 91.40% | ~64 minutes |

Le fine-tuning de CamemBERT améliore l'accuracy de **+5.6 points** par rapport à l'approche from-scratch, au prix d'un temps d'entraînement nettement plus long , cohérent avec l'avantage du pré-entraînement sur de grandes quantités de texte français.

## Modèle publié

Le modèle fine-tuné est disponible sur HuggingFace Hub : [Narsat/camembert-sentiment-allocine](https://huggingface.co/Narsat/camembert-sentiment-allocine)

Utilisation rapide :
```python
from transformers import pipeline

classifieur = pipeline("sentiment-analysis", model="Narsat/camembert-sentiment-allocine")
resultat = classifieur("Ce film est vraiment excellent !")
print(resultat)
# [{'label': 'positif', 'score': 0.9554412364959717}]
```

*Note : le format exact des labels retournés (`positif`/`négatif`), à tester avant de se fier à ce résultat pour une intégration dans une application.*

## Installation et exécution

```bash
git clone https://github.com/narsatGit/camembert-sentiment-allocine-2.git
cd camembert-sentiment-allocine-2
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
python src/train.py
```

## Structure du projet

```
camembert-sentiment-allocine-2/
├── src/
│   ├── preprocessing.py   # Chargement dataset + tokenisation CamemBERT
│   ├── model.py            # Chargement du modèle (AutoModelForSequenceClassification)
│   ├──train.py             # TrainingArguments, Trainer, entraînement, métriques
│   └── publier.py          # Publication du modèle sur HuggingFace Hub
├── resultats/
│   └── metriques.txt       # Accuracy et loss par époque
└── requirements.txt
```