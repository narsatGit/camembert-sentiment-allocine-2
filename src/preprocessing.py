from datasets import load_dataset
from transformers import AutoTokenizer

# Chargement du tokenizer pré-entraîné de CamemBERT
tokenizer = AutoTokenizer.from_pretrained("camembert-base")

def tokeniser_fonction(exemples):
    r"""Tokenise les critiques textuelles pour les adapter à l'entrée de CamemBERT.

    Args:
        exemples (dict): Lot d'exemples contenant la clé 'review'

    Returns:
        dict: Dictionnaire contenant les input_ids et attention_mask sous forme de tokens.
    """
    # Découpage du texte, troncature à 128 tokens et ajout de padding si nécessaire
    return tokenizer(exemples["review"], truncation=True, padding="max_length", max_length=128)


def preparer_donnees():
    r"""Charge le dataset Allociné, extrait un sous-ensemble mélangé et applique la tokenisation

    Returns:
        tuple: (dataset_train_tokenise, dataset_test_tokenise) prêts pour l'entraînement.
    """
    # Chargement du dataset depuis Hugging Face
    dataset = load_dataset("tblard/allocine")
    
    # Sélection aléatoire mais reproductible de 2000 exemples (train) et 500 (test)
    dataset_train_subset = dataset["train"].shuffle(seed=42).select(range(2000))
    dataset_test_subset = dataset["test"].shuffle(seed=42).select(range(500))
    
    # Application de la tokenisation par lots sur l'ensemble des données
    dataset_train_tokenise = dataset_train_subset.map(tokeniser_fonction, batched=True)
    dataset_test_tokenise = dataset_test_subset.map(tokeniser_fonction, batched=True)
    
    return dataset_train_tokenise, dataset_test_tokenise
