from transformers import TrainingArguments, Trainer
import numpy as np
import os
from preprocessing import preparer_donnees
from model import charger_modele
# la fonction de métriques

def calculer_metriques(eval_pred):
    r"""Calcule l'exactitude (accuracy) à partir des prédictions brutes du modèle.

    Args:
        eval_pred (tuple): Contient (logits, labels)

    Returns:
        dict: Dictionnaire contenant la valeur de l'accuracy
    """
    logits =  eval_pred[0]
    labels =  eval_pred[1]
    
    # Récupération de l'indice de la classe ayant la plus forte probabilité (argmax)
    classes_predites = np.argmax(logits, axis=-1)
    accuracy = np.mean(classes_predites == labels)
    return {"accuracy": accuracy}

# objet qui centralise tous les réglages de l'entraînement
def main():
    r"""Fonction principale pour exécuter le fine-tuning avec la classe Trainer de Hugging Face."""
    
    # Configuration des hyperparamètres et options d'entraînement
    arg = TrainingArguments(
        output_dir="./resultats",           # Dossier de sauvegarde
        num_train_epochs=2,                 # Nombre d'époques
        per_device_train_batch_size=16,     # Taille du lot pour l'entraînement
        per_device_eval_batch_size=16,      # Taille du lot pour l'évaluation
        eval_strategy="epoch",              # Évaluation à la fin de chaque époque
        save_strategy="epoch",              # Sauvegarde du modèle à chaque époque
        load_best_model_at_end=True         # Charger la meilleure version du modèle à la fin
    )

    # Préparation des données tokenisées
    dataset_train_tokenise, dataset_test_tokenise = preparer_donnees()
    
    # Initialisation de l'outil Trainer
    trainer = Trainer(
        model = charger_modele(),
        args = arg,
        train_dataset = dataset_train_tokenise,
        eval_dataset = dataset_test_tokenise,
        compute_metrics = calculer_metriques
    )
    
    # Lancement de la boucle d'entraînement
    trainer.train()
    
    # Création du dossier de résultats et sauvegarde des métriques d'évaluation
    os.makedirs("resultats", exist_ok=True)
    with open("resultats/metriques.txt", "w") as f:
        for element in trainer.state.log_history:
            if "eval_accuracy" in element:
                f.write(f"Époque {element['epoch']} - Loss: {element['eval_loss']} - Accuracy: {element['eval_accuracy']}\n")
        
    
if __name__ == "__main__":
    main()