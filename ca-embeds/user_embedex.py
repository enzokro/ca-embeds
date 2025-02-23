from typing import Dict
from fastcore.foundation import L
from fastcore.xtras import ls
import modal
from image import image
from volume import vol, REMOTE_DIR

# NOTE: first proper embedding model built on modernBERT
DEFAULT_EMBEDDING_MODEL = "nomic-ai/modernbert-embed-base"
APP_NAME = "community-archive-v1-user"

# initialize Modal app first, so we can use it for function decorators
app = modal.App(APP_NAME)
# Update the app with the configured image
app.image = image

# create the modal class
@app.cls(
    container_idle_timeout=60,
    gpu="A10G",
    image=image,
    volumes={REMOTE_DIR: vol},
    timeout=60*60  # wait for 1 hour since we're only processing one user
)
class UserTweetEmbedder:
    "Embeds Community Archive tweets for a single user."

    @modal.enter()
    def initialize(self):
        print("Preparing empty model...")
        self.model = None
    
    @modal.web_endpoint(method="POST", docs=True)
    def embed_user(self, data: dict) -> Dict:
        """
        Embeds tweets from a single user's archive.
        
        Args:
            data: Dictionary containing:
                - username: str - The username whose archive to process
                - model_name: Optional[str] - Full name of the huggingface embedding model to use
                
        Returns:
            dict: Dictionary containing:
                - username: str - Processed username
                - embeddings: List[List[float]] - List of tweet embeddings
                - metadata: List[dict] - List of tweet metadata
                - status: str - Status of the operation
        """
        import json
        import traceback
        import torch
        import numpy as np
        from sentence_transformers import SentenceTransformer
        from tqdm.auto import tqdm
        from fastcore.basics import Path

        # Get required username
        username = data.get("username")
        if not username:
            return {"status": "error", "message": "Username is required"}

        # speed things up a bit
        torch.set_float32_matmul_precision('high')

        # parse in the args
        self.model_name = data.get("model_name", DEFAULT_EMBEDDING_MODEL)
        print("Using model:", self.model_name)

        # initialize the model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(self.model_name, device=device)
        self.model.encode("moving to device...") # warmup the model

        # setup output directory
        clean_model_name = self.model_name.replace("/", "_") # deal with huggingface model name conventions
        out_dir = Path(REMOTE_DIR) / f'{clean_model_name}_userEmbeds'
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            print(f"Processing {username}...")
            
            # load the user's archive
            archive_path = Path(REMOTE_DIR) / 'data' / username / "archive.json"
            if not archive_path.exists():
                return {"status": "error", "message": f"Archive not found for user {username}"}
                
            with open(archive_path, 'r', encoding='utf-8') as f:
                archive_data = json.load(f)

            # get the account id
            account_id = archive_data['account'][0]['account']['accountId']

            # extract all tweets that are not retweets or replies
            tweets = []
            for item in archive_data.get('tweets', []):
                tweet = item['tweet']
                if tweet.get('full_text', '').startswith('RT'):
                    continue
                if tweet.get('in_reply_to_user_id') and tweet['in_reply_to_user_id'] != account_id:
                    continue
                text = tweet.get("full_text", "")
                if text:  # only include non-empty tweets
                    tweets.append(tweet)

            if not tweets:
                return {
                    "status": "success",
                    "username": username,
                    "embeddings": [],
                    "metadata": [],
                    "message": "No valid tweets found for embedding"
                }

            # create embeddings for all tweets
            texts = [t["full_text"] for t in tweets]
            embeddings = []
            batch_size = 128
            num_batches = (len(texts) + batch_size - 1) // batch_size  # Ceiling division
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(texts))
                batch = texts[start_idx:end_idx]
                batch_embeddings = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
                embeddings.extend(batch_embeddings)
            embeddings = np.array(embeddings)
            
            # prepare metadata for each tweet
            metadata = []
            for tweet in tweets:
                metadata.append({
                    "id": tweet.get("id"),
                    "created_at": tweet.get("created_at"),
                    "text": tweet.get("full_text"),
                    'in_reply_to_user_id': tweet.get('in_reply_to_user_id', ''),
                    'user_mentions': tweet.get('entities', {}).get('user_mentions', [])
                })

            # save the results            
            out_file = out_dir / f"{username}_embeddings.json"
            with open(out_file, 'w') as f:
                json.dump(metadata, f)
            np.save(out_dir / f"{username}_embeddings.npy", embeddings)

            return {
                "status": "success",
                "username": username,
                "embeddings": embeddings.tolist(),
                "metadata": metadata
            }

        except Exception as e:
            error_msg = f"Error processing {username}: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return {
                "status": "error",
                "username": username,
                "message": error_msg
            }

@app.local_entrypoint()
def main(username: str):
    """Local testing entrypoint. Run this to process a single user's archive."""
    embedder = UserTweetEmbedder()
    result = embedder.embed_user.remote({"username": username})
    print(f"Processed {username}:", result["status"])
