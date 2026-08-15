r"""Script de publication d'un modèle de classification de sentiment sur le Hugging Face Hub.

Ce script charge un point de contrôle (checkpoint) local d'un modèle CamemBERT fine-tuné,
y associe le tokenizer de base, configure le dictionnaire de correspondance des étiquettes (labels),
puis publie l'ensemble sur le profil utilisateur Hugging Face.
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Chemin vers le dossier local contenant les poids du modèle fine-tuné
checkpoint_path = "resultats/checkpoint-125"
# Nom sous lequel le modèle sera publié sur le Hugging Face Hub
nom_modele_hub = "camembert-sentiment-allocine"

# Chargement des poids du modèle sauvegardés localement pendant l'entraînement
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
# Chargement du tokenizer d'origine pour accompagner le modèle lors de la publication
tokenizer = AutoTokenizer.from_pretrained("camembert-base")

# Association des IDs numériques aux noms de classes lisibles pour l'inférence
model.config.id2label = {0: "négatif", 1: "positif"}
model.config.label2id = {"négatif": 0, "positif": 1}

# Envoi des fichiers du modèle (poids, configuration) vers votre dépôt Hugging Face
model.push_to_hub(nom_modele_hub)

# Envoi du tokenizer associé pour permettre une réutilisation directe via pipeline()
tokenizer.push_to_hub(nom_modele_hub)

print(f"Modèle publié : https://huggingface.co/Narsat/{nom_modele_hub}")