from transformers import AutoModelForSequenceClassification, AutoTokenizer

checkpoint_path = "resultats/checkpoint-125"
nom_modele_hub = "camembert-sentiment-allocine"

model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained("camembert-base")

model.push_to_hub(nom_modele_hub)
tokenizer.push_to_hub(nom_modele_hub)

print(f"Modèle publié : https://huggingface.co/Narsat/{nom_modele_hub}")