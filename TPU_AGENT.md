# TPU runbook

Use this when asking Claude Code, Codex, or another agent to run the workshop.

Rules:

- Use only your assigned project: `dis-2026-tpu-${CRSID}`.
- Use `us-east5-b` unless the instructor says otherwise.
- Print the cleanup command before creating a TPU.
- Delete the TPU VM before leaving.
- List TPU VMs after deletion and check that none remain.

Set:

```bash
export CRSID="CRSID"
export PROJECT_ID="dis-2026-tpu-${CRSID}"
export ZONE="us-east5-b"
export ACCELERATOR_TYPE="v6e-1"
export RUNTIME_VERSION="v6e-ubuntu-2404"
export TPU_NAME="dis-2026-${CRSID}-tpu"
```

Preflight:

```bash
./scripts/student_preflight.sh
```

Create:

```bash
gcloud compute tpus tpu-vm create "$TPU_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --accelerator-type="$ACCELERATOR_TYPE" \
  --version="$RUNTIME_VERSION"
```

List:

```bash
gcloud compute tpus tpu-vm list --project="$PROJECT_ID" --zone="$ZONE"
```

SSH:

```bash
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --project="$PROJECT_ID" --zone="$ZONE"
```

Tunnel Jupyter:

```bash
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  -- -L 8888:127.0.0.1:8888
```

Delete:

```bash
gcloud compute tpus tpu-vm delete "$TPU_NAME" --project="$PROJECT_ID" --zone="$ZONE" --quiet
gcloud compute tpus tpu-vm list --project="$PROJECT_ID" --zone="$ZONE"
```

Expected final state: no TPU VMs listed.
