from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path="artifacts/model_trainer/tokenizer",   # your local tokenizer folder
    repo_id="amaan-1107/text-summarizer-tokenizer",    # your HF username/repo name
    repo_type="model"
)
