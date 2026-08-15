from transformers import AutoModelForSequenceClassification

def charger_modele():
    r"""Charge le modèle CamemBERT pré-entraîné avec une tête de classification de séquence.

    Returns:
        AutoModelForSequenceClassification: Modèle CamemBERT configuré pour la classification binaire.
    """
    model_name = "camembert-base"
    
    # Chargement du modèle avec 2 étiquettes de sortie (0: Négatif, 1: Positif)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    return model