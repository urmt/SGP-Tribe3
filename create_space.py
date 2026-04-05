from huggingface_hub import HfApi, create_repo

HF_TOKEN = "your_new_token_here"
HF_USERNAME = "MRTraver"
SPACE_NAME = "sgp-tribe3"

api = HfApi(token=HF_TOKEN)

# Create the Space
create_repo(
    repo_id=f"{HF_USERNAME}/{SPACE_NAME}",
    repo_type="space",
    space_sdk="docker",
    token=HF_TOKEN,
    private=False,
    exist_ok=True,
)
print("Space created!")

# Set HF_TOKEN as secret
api.add_space_secret(
    repo_id=f"{HF_USERNAME}/{SPACE_NAME}",
    key="HF_TOKEN",
    value=HF_TOKEN,
)
print("Secret set!")
