from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path= "models/distilbert-sentiment",
    repo_id= "Raghav0511/distilbert-sentiment-imdb",
    repo_type= "model"
)