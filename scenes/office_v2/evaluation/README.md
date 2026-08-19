# Evaluation-only boundary

Files in this directory contain privileged simulator ground truth. They are
for offline acceptance tests only. Production containers and product-facing
services must not mount or read this directory; pure-VLM evaluation receives
RGB/RGB-D observations and public robot pose/TF, never object IDs or target
answers from this manifest.
