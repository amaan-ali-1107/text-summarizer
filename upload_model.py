from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="artifacts/model_trainer/pegasus-samsum-model", 
    repo_id="amaan-1107/text-summarizer-model",  
    repo_type="model"
)
