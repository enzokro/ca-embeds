# Extracting Community Archive embeddings with Modal

This project came from the Community Archive hackathon hosted at FractalTech in Februrary 2025. Many folks were building apps with embeddings, so I decided to create embeddings for all of the archive. 

With [modal](https://modal.com/docs/), we can quickly extract embeddings for tons of users using a cloud GPU. This is much better than waiting hours or days for the extractions on our laptops. They offer a very generous $30 in monthly credits that gets us tons of free experiments. The rest of this guide assumes you've created a Modal account and setup the proper API keys. 

As a first pass, I am extracting all tweets and threads from a user. Retweets and replies to other users are excluded, but a one-line code change can fold them in. 

There are two main files: 

- `ca-embeds/archive_embedex.py`   
- `ca-embeds/user_embedex.py`  

The first one extracts embeddings for every user in the archive. The second one extracts embeddings for a single user. 

For now, we assume that you're starting with a local copy of the Community Archive. We then upload this copy to a modal volume, which the cloud GPU pulls and runs from. This could be much cleaner by directly pulling the latest & relevant archives from supabase, but that's left as future work. 

## Environment Setup

Copy the `.example.env` file to `.env`, and fill in the appropriate Modal and HuggingFace values. 


## Uploading your local Community Archive

This repo assumes you've gone through the official [Community Archive local setup](https://github.com/TheExGenesis/community-archive/blob/main/docs/local-setup.md) and have downloaded the archive to a local directory on your device.  

The file in `ca-embeds/volume.py` creates a remote Modal volume and uploads the local archive to it. In this file, you must set the following variables:
- Where the archive lives locally. 
- Where you want to store it in the volume.
- The name of the uploaded volume. 

Once those are set, kick off this command to upload your local archive:
```bash
uv run python ca-embeds/volume.py
```

## Extracting all archive embeddings

Once the volume is up, we can create the Modal cloud GPU instances that will extract embeddings for us. For more details see the excellent Modal docs, but at a high-level:   

- We deploy `ca-embeds/archive_embedex.py` as a web endpoint.  
- We then make a POST request to this endpoint, and it will extract embeddings for all of the archive.   
- - In the POST body, we pass the specific HuggingFace model to use. 
- - **NOTE!** We are using the `sentence-transformers` library, which returns un-normalized embeddings by default.

In the `archive_embedex.py` file, what you do with the embeddings is up to you. By default it will save them to the same remote Modal volume. During the hackathon, I uploaded them to a supabase s3 bucket and served public URLs at this dashboard: [Hackathon Embeddings](https://ca-embeds-dashboard-production.up.railway.app/). I was also able to upload them on `pgVector` using the `vecs` library from supabase. 

There is an ongoing Discord discussion about the best place and schema for stored embeddings, come join and contribute! 

Command to deploy the archive Modal endpoint:
```bash
uv run modal deploy ca-embeds/archive_embedex.py
```

This command returns the URL that you can make a POST request to. See the example notebook in `nbs/` for more details.  

## Extracting a single user's embeddings

To extract a single user's embeddings, we can use `ca-embeds/user_embedex.py`. This is very similar to `archive_embedex.py`:   
- We deploy `ca-embeds/user_embedex.py` as a web endpoint.  
- We then make a POST request to this endpoint, and it will extract and return embeddings for the user.   
- - In the POST body, we pass the specific HuggingFace model to use, and the @-name of the user to extract.  

Command to deploy the user Modal endpoint:
```bash
uv run modal deploy ca-embeds/user_embedex.py
```

This command returns the URL that you can make a POST request to. See the example notebook in `nbs/` for more details.  

## Example POST requests

Here's how easy it is to extract embeddings with a POST request. These are live values I used during the hackathon, to extract the embeddings seen in the Railway dashboard above.  

```python
import requests

# output of the `modal deploy` endpoint above.
MODAL_ENDPOINT = "https://kronresearch--community-archive-v1-tweetembedder-embed-archive.modal.run"

# pick your embedding model of choice
# model_name = "Alibaba-NLP/gte-modernbert-base"
model_name = "nomic-ai/modernbert-embed-base"

# extract the embeddings
r = requests.post(
    MODAL_ENDPOINT,
    json={"model_name": model_name},
)
```

Note that web endpoints might be overkill for your use case. Feel free to run Modal locally for quicker experiments. 

## GPU Docker Images 

The file `ca-embeds/image.py` contains the docker image we use to run the modal endpoints. We use it to build an image with the following:
- CUDA 12.4 devel on Ubuntu 22.04  
- Python 3.12  
- Flash-Attention  

I've tried many cloud GPUs, and the ease + customizability of Modal is by far the best. The above image.py file might seem verbose, but it shows how much power Modal provides. Getting the equivalent setup in other cloud providers is like pulling teeth. 

## Future Work 

- Host a permanent endpoint on the cloud, and use it to extract embeddings for the archive.
- Let users access this endpoint via an easy API. 
- Quick way of extracting embeddings for a given list of tweet ids. 
- Dashboard to explore, cluster, visualize different embeddings.
- Make it trivially easy for users to get up-and-running with embeddings for their apps.