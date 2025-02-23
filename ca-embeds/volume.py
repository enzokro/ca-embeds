"""Push the local community archive to a remote modal volume.

NOTE: Assumes you have the proper modal env vars setup.
"""
import modal

# NOTE: replace this with where you downloaded the CA, following the official instructions
LOCAL_DIR = "<<PATH_TO_YOUR_LOCAL_ARCHIVE>>"
# storing things in the default path
REMOTE_DIR = "/data"
VOL_NAME = "community-archive-v1"

# create app to run this file
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True) 

def upload_data():
    """Upload the local data to the remote volume."""
    with vol.batch_upload() as batch:
        batch.put_directory(LOCAL_DIR, REMOTE_DIR)

# Create or get existing volume for model weights
if __name__ == "__main__":
    upload_data()
