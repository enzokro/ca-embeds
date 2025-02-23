from typing import List, Dict, Tuple, Optional
import json
import numpy as np
from fastcore.basics import Path
from fastcore.foundation import L
from fastcore.xtras import ls
import modal
from image import image
from volume import vol, REMOTE_DIR

# NOTE: first proper embedding model built on modernBERT
DEFAULT_EMBEDDING_MODEL = "nomic-ai/modernbert-embed-base"
APP_NAME = "community-archive-v1"

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
    timeout=60*60*23 # wait for 23 hours
)
class TweetEmbedder:
    "Embeds Community Archive tweets."

    @modal.enter()
    def initialize(self):
        print("Preparing empty model...")
        self.model = None
    
    @modal.web_endpoint(method="POST", docs=True)
    def embed_archive(self, data: dict) -> Dict:
        """
        Embeds tweets from all user archives in the directory.
        
        Args:
            data: Dictionary containing:
                - model_name: Optional[str] - Full name of the huggingface embedding model to use
                
        Returns:
            dict: Dictionary containing:
                - processed_users: List[str] - List of successfully processed users
                - failed_users: List[dict] - List of users that failed with error messages
                - status: str - Overall status of the operation
        """
        import json
        import traceback
        import torch
        import numpy as np
        from sentence_transformers import SentenceTransformer
        from tqdm.auto import tqdm
        from fastcore.basics import Path
        from fastcore.xtras import ls

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
        out_dir = Path(REMOTE_DIR) / f'{clean_model_name}'
        out_dir.mkdir(parents=True, exist_ok=True)

        # find all users in the volume directory, assuming they were uploaded to REMOTE_DIR (see volume.py)
        data_dir = Path(REMOTE_DIR) / 'data'
        users = data_dir.ls().filter(lambda x: x.is_dir()).map(lambda x: x.name).sorted()
        
        if not users:
            return {"status": "error", "message": "No user archives found"}

        # keep track of which users succeeded and which fiales
        processed_users = []
        failed_users = []

        # store embeddings for all users in a single file
        all_embeddings = []
        all_metadata = []
        
        # process each user
        for username in tqdm(users, desc="Processing users"):
            try:
                print(f"Processing {username}...")

                # load the current user's archive
                archive_path = data_dir / username / "archive.json"
                with open(archive_path, 'r', encoding='utf-8') as f:
                    archive_data = json.load(f)

                # Extract tweets and metadata
                tweets = []
                metadata = []
                
                for item in archive_data.get('tweets', []):
                    tweet = item['tweet']
                    # NOTE: this is how accountIds are stored in the json
                    account_id = archive_data['account'][0]['account']['accountId']

                    # get the tweet's text and useful metadata
                    text, meta = self._process_tweet(
                        tweet,
                        account_id
                    )
                    # append valid tweets for extraction
                    if text and meta:
                        tweets.append(text)
                        metadata.append(meta)
    
                # mark as failure if we could not get any tweets for this user
                if not tweets:
                    failed_users.append({
                        "username": username,
                        "error": "No valid tweets found"
                    })
                    continue

                # Generate embeddings in batches
                bs = 128 # TODO: tune batch size
                embeddings = self._batch_encode(tweets, bs)

                print(f"Embedding {username} complete, found {len(embeddings)} embeddings")
                
                # Calculate total size in GB (assuming float32 embeddings)
                embedding_size_gb = embeddings.nbytes / (1024**3)
                MAX_SHARD_SIZE_GB = 10

                # add embeddings to the all_embeddings list
                all_embeddings.extend(embeddings)
                all_metadata.extend(metadata)
                
                if embedding_size_gb <= MAX_SHARD_SIZE_GB:
                    # Single shard case - original behavior
                    self._save_shard(embeddings, metadata, username, out_dir, shard_idx=0)   
                    print(f"Saved single shard - Embeddings size: {embedding_size_gb:.2f}GB")

                else:
                    # Multi-shard case
                    # Calculate number of embeddings per shard to stay under size limit
                    embeddings_per_gb = len(embeddings) / embedding_size_gb
                    embeddings_per_shard = int(embeddings_per_gb * MAX_SHARD_SIZE_GB)
                    num_shards = (len(embeddings) + embeddings_per_shard - 1) // embeddings_per_shard
                    
                    print(f"Sharding {len(embeddings)} embeddings into {num_shards} shards")
                    
                    # Create and upload each shard
                    for shard_idx in range(num_shards):
                        start_idx = shard_idx * embeddings_per_shard
                        end_idx = min(start_idx + embeddings_per_shard, len(embeddings))
                        
                        # Shard both embeddings and metadata to maintain alignment
                        shard_embeddings = embeddings[start_idx:end_idx]
                        shard_metadata = metadata[start_idx:end_idx]
                        
                        # Save shards
                        self._save_shard(
                            shard_embeddings,
                            shard_metadata,
                            username,
                            out_dir,
                            shard_idx
                        )

                # Update processed_users with shard information
                processed_users.append({
                    "username": username,
                    "status": "success",
                    "tweet_count": len(tweets),
                    "embedding_shape": embeddings.shape,
                    "total_size_gb": embedding_size_gb,
                    "num_shards": 1 if embedding_size_gb <= MAX_SHARD_SIZE_GB else num_shards
                })

                print(f"Successfully processed and uploaded all shards for {username}")

            except Exception as e:
                failed_users.append({
                    "username": username,
                    "error": str(e),
                    'stacktrace': traceback.format_exc()
                })
                
        # save all embeddings and metadata to a single file
        all_embeddings_path = out_dir / "all_embeddings.npy"
        all_metadata_path = out_dir / "all_metadata.json"
        np.save(all_embeddings_path, all_embeddings)
        with open(all_metadata_path, 'w') as f:
            json.dump(all_metadata, f)

        # persist changes to the volume
        vol.commit()

        return {
            "status": "complete",
            "processed_users": processed_users,
            "failed_users": failed_users,
            "summary": {
                "total_users": len(users),
                "successful": len([u for u in processed_users if u["status"] == "success"]),
                "skipped": len([u for u in processed_users if u["status"] == "skipped"]),
                "failed": len(failed_users)
            }
        }

    def _process_tweet(self, tweet: Dict, account_id: Optional[str] = None) -> Tuple[Optional[str], Optional[Dict]]:
        "Process a single tweet, returning (text, metadata) if valid, (None, None) if should skip"

        # skip retweets and replies for now
        if tweet.get('full_text', '').startswith('RT'): return None, None
        if tweet.get('in_reply_to_user_id') and tweet['in_reply_to_user_id'] != account_id: return None, None
        
        text = tweet.get('full_text', '')
        metadata = {
            'id': tweet.get('id', ''),
            'created_at': tweet.get('created_at', ''),
            'text': text,
            'in_reply_to_user_id': tweet.get('in_reply_to_user_id', ''),
            'user_mentions': tweet.get('entities', {}).get('user_mentions', [])
        }
        return text, metadata
    
    def _batch_encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        "Encode texts in batches, returning numpy array of embeddings"
        embeddings = []
        num_batches = (len(texts) + batch_size - 1) // batch_size  # Ceiling division
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(texts))
            batch = texts[start_idx:end_idx]
            batch_embeddings = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)
    
    def _save_shard(self, embeddings: np.ndarray, metadata: List[Dict], username: str, 
                    out_dir: Path, shard_idx: int = 0) -> float:
        "Save and upload a single shard of embeddings"
        embedding_path = out_dir/f"{username}_shard_{shard_idx}.npy"
        metadata_path = out_dir/f"{username}_shard_{shard_idx}.json"
        
        # save embeddings and metadata
        np.save(embedding_path, embeddings)
        json.dump(metadata, open(metadata_path, 'w'))


@app.local_entrypoint()
def main():
    """Local testing entrypoint. Run this to process all archives."""
    print("Initializing Tweet Embedder...")
    

if __name__ == "__main__":
    main()
