#!/usr/bin/env bash
# Read-only DIS 2026 student TPU preflight.
# This script checks the local gcloud context and the assigned project before any TPU VM is created.
set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID, e.g. dis-2026-tpu-${CRSID:-CRSID}}"
: "${CRSID:?Set CRSID, e.g. abc123}"
: "${ZONE:=us-east5-b}"
: "${ACCELERATOR_TYPE:=v6e-1}"
: "${RUNTIME_VERSION:=v6e-ubuntu-2404}"
: "${TPU_NAME:=dis-2026-${CRSID}-tpu}"
: "${MAX_WORKSHOP_TPUS:=2}"

EXPECTED_PROJECT_ID="dis-2026-tpu-${CRSID}"
APPROVED_TPU_ZONES="us-east5-b"

fail() { echo "FAIL: $*" >&2; exit 1; }
pass() { echo "PASS: $*"; }
section() { printf '\n== %s ==\n' "$*"; }

section "Identity"
gcloud auth list --filter=status:ACTIVE --format='table(account,status)'
ACTIVE_ACCOUNT="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' | head -n1 || true)"
[ -n "$ACTIVE_ACCOUNT" ] || fail "no active gcloud account"
pass "active account: $ACTIVE_ACCOUNT"

section "Assigned project guard"
[ "$PROJECT_ID" = "$EXPECTED_PROJECT_ID" ] || fail "PROJECT_ID must be ${EXPECTED_PROJECT_ID}; got ${PROJECT_ID}"
case "$PROJECT_ID" in
  ""|"dis-2026-tpu"|"cambridge-tpu"|"dis-2026-test-"*)
    fail "controller/shared/test project is not allowed: ${PROJECT_ID}"
    ;;
esac
pass "project id matches assigned CRSID pattern"

gcloud config set project "$PROJECT_ID" >/dev/null

gcloud projects describe "$PROJECT_ID" --format='table(projectId,projectNumber,lifecycleState)'

section "Billing"
gcloud billing projects describe "$PROJECT_ID" --format='table(projectId,billingEnabled,billingAccountName)'
BILLING_ENABLED="$(gcloud billing projects describe "$PROJECT_ID" --format='value(billingEnabled)' || true)"
[ "$BILLING_ENABLED" = "True" ] || [ "$BILLING_ENABLED" = "true" ] || fail "billing is not enabled"
pass "billing enabled"

section "Required APIs"
for api in compute.googleapis.com tpu.googleapis.com; do
  if gcloud services list --enabled --project="$PROJECT_ID" --filter="config.name=${api}" --format='value(config.name)' | grep -qx "$api"; then
    pass "API enabled: $api"
  else
    fail "API not enabled: $api"
  fi
done

section "Approved zone/type/runtime"
case " $APPROVED_TPU_ZONES " in
  *" $ZONE "*) pass "approved zone: $ZONE" ;;
  *) fail "ZONE ${ZONE} is not in approved zones: ${APPROVED_TPU_ZONES}" ;;
esac

gcloud compute tpus tpu-vm accelerator-types list \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --filter="name~${ACCELERATOR_TYPE}" \
  --format='table(name,zone)'

gcloud compute tpus tpu-vm versions list \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --filter="name~${RUNTIME_VERSION}" \
  --format='table(name,zone)'

section "Live TPU count before create"
live_tpus=0
for z in $APPROVED_TPU_ZONES; do
  gcloud compute tpus tpu-vm list \
    --project="$PROJECT_ID" \
    --zone="$z" \
    --filter="acceleratorType=${ACCELERATOR_TYPE}" \
    --format='table(name,state,acceleratorType,health,createTime)'
  count="$(gcloud compute tpus tpu-vm list \
    --project="$PROJECT_ID" \
    --zone="$z" \
    --filter="acceleratorType=${ACCELERATOR_TYPE}" \
    --format='value(name)' | wc -l | tr -d ' ')"
  live_tpus=$((live_tpus + count))
done
[ "$live_tpus" -lt "$MAX_WORKSHOP_TPUS" ] || fail "live ${ACCELERATOR_TYPE} TPU count ${live_tpus} is at/above limit ${MAX_WORKSHOP_TPUS}"
pass "live ${ACCELERATOR_TYPE} TPU count ${live_tpus}/${MAX_WORKSHOP_TPUS}"

section "Cleanup command to save before create"
echo "gcloud compute tpus tpu-vm delete '$TPU_NAME' --project='$PROJECT_ID' --zone='$ZONE' --quiet"

section "Result"
pass "preflight passed. Do not create a TPU until the instructor confirms the project budget/quota cap was provisioned."
