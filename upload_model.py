from huggingface_hub import upload_folder, HfApi, whoami
import os
import sys

# === Configuration ===
REPO_ID = "AndaiMD/brain-unet-model" 
FOLDER_PATH = "brain_unet_model"   
REPO_TYPE = "model"

def main():
    try:
        user = whoami()
        print(f"✅ Authenticated as: {user['name']}")
    except Exception as e:
        print("❌ Not authenticated. Run `huggingface-cli login` first.")
        sys.exit(1)

    if not os.path.exists(FOLDER_PATH):
        print(f"❌ Folder not found: {FOLDER_PATH}")
        sys.exit(1)

    print(f"Uploading `{FOLDER_PATH}` to Hugging Face Hub as `{REPO_ID}` ...")

    upload_folder(
        folder_path=FOLDER_PATH,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
    )

    print(f"✅ Upload complete! View your model at: https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    main()
