# DIS 2026 JAX-on-TPU workshop

This is the student repo for the DIS 2026 JAX/TPU workshop.

You will:

1. start a small TPU VM in your assigned Google Cloud project;
2. open Jupyter on that TPU VM;
3. run the notebook;
4. delete the TPU VM before leaving.

The notebook trains a small GPT-2-style model from scratch on Shakespeare text using JAX, Flax, and Optax.

## 0. Set your project variables

Replace `CRSID` with your Cambridge CRSid.

```bash
export CRSID="CRSID"
export PROJECT_ID="dis-2026-tpu-${CRSID}"
export ZONE="us-east5-b"
export ACCELERATOR_TYPE="v6e-1"
export RUNTIME_VERSION="v2-alpha-tpuv6e"
export TPU_NAME="dis-2026-${CRSID}-tpu"
```

Run the preflight check:

```bash
./scripts/student_preflight.sh
```

If it fails, stop and ask for help.

## 1. Create the TPU VM

```bash
gcloud compute tpus tpu-vm create "$TPU_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --accelerator-type="$ACCELERATOR_TYPE" \
  --version="$RUNTIME_VERSION"
```

Check it is ready:

```bash
gcloud compute tpus tpu-vm list \
  --project="$PROJECT_ID" \
  --zone="$ZONE"
```

## 2. Copy this repo to the TPU VM

From your laptop, in this repo:

```bash
tar --exclude='.git' --exclude='data' --exclude='checkpoints' --exclude='outputs' \
  -czf /tmp/dis2026_tpu_workshop.tgz .

gcloud compute tpus tpu-vm scp \
  /tmp/dis2026_tpu_workshop.tgz "${TPU_NAME}:~/dis2026_tpu_workshop.tgz" \
  --project="$PROJECT_ID" \
  --zone="$ZONE"
```

Then SSH into the TPU VM:

```bash
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE"
```

On the TPU VM:

```bash
rm -rf ~/gcp_jax_tpu_lecture
mkdir -p ~/gcp_jax_tpu_lecture
tar -xzf ~/dis2026_tpu_workshop.tgz -C ~/gcp_jax_tpu_lecture
cd ~/gcp_jax_tpu_lecture
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
uv pip install -r requirements-tpu.txt
```

## 3. Start Jupyter

On the TPU VM:

```bash
cd ~/gcp_jax_tpu_lecture
jupyter lab --no-browser --ip=127.0.0.1 --port=8888
```

In a second terminal on your laptop, open the tunnel:

```bash
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  -- -L 8888:127.0.0.1:8888
```

Open the Jupyter URL printed by the TPU VM, usually:

```text
http://127.0.0.1:8888/lab?token=...
```

If port 8888 is already in use on your laptop, use:

```bash
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  -- -L 8889:127.0.0.1:8888
```

Then open `http://127.0.0.1:8889/lab?token=...`.

## 4. Run the notebook

Open:

```text
notebooks/dis_2026_jax_tpu_gpt2_shakespeare.ipynb
```

First check that JAX sees TPU devices:

```python
import jax
jax.devices()
```

## 5. Delete the TPU VM

Before leaving:

```bash
gcloud compute tpus tpu-vm delete "$TPU_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --quiet

gcloud compute tpus tpu-vm list \
  --project="$PROJECT_ID" \
  --zone="$ZONE"
```

The final list should be empty. Idle READY TPUs cost money.
