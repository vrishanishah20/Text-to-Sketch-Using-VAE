from huggingface_hub import HfApi, HfFolder

api = HfApi()
token = HfFolder.get_token()  # Gets locally stored token

# Check if access to a model
try:
    api.model_info("bert-base-uncased", token=token)
    print("Token is valid and has access.")
except Exception as e:
    print("Token access failed:", e)
