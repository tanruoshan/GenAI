#!/usr/bin/env bash
# pod.sh -- run the BART fine-tuning and repair on a RunPod GPU pod (runs on the laptop).
#
# A slim version of the RunPod REST workflow of Simon's thesis. The pod gets exactly the committed code
# (git archive HEAD, no GitHub access needed on the pod) plus the two git-ignored data files, whose hashes
# are checked there against runs/*.sha256.
#
#   bash scripts/pod/pod.sh price               # A40 price and account balance (creates nothing)
#   bash scripts/pod/pod.sh create [--go]       # dry run without --go
#   bash scripts/pod/pod.sh status  ID
#   bash scripts/pod/pod.sh push    ID          # code + data to /workspace/GenAI, hash check, pip install
#   bash scripts/pod/pod.sh run     ID "CMD"    # CMD in the background on the pod, log in runs/pod_run.log
#   bash scripts/pod/pod.sh log     ID
#   bash scripts/pod/pod.sh fetch   ID          # models/bart_ft_v1, runs/bart_*, runs/preds/bart_*, the log
#   bash scripts/pod/pod.sh terminate ID
#
# The key: RUNPOD_API_KEY from .env of this repo, else from the thesis' experiments/.env (read only). It is
# never printed and never sent to the pod.
set -euo pipefail
cd "$(dirname "$0")/../.."

API=https://rest.runpod.io/v1
GQL=https://api.runpod.io/graphql
GPU="NVIDIA A40"
IMAGE="runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
REMOTE=/workspace/GenAI
KNOWN_HOSTS="$HOME/.ssh/known_hosts_genai_pods"
SSH_OPTS=(-o "UserKnownHostsFile=$KNOWN_HOSTS" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15
          -o ServerAliveInterval=30 -o BatchMode=yes)

die() { echo "FATAL: $*" >&2; exit 1; }

key() {
  local f
  for f in .env /Users/simonm/dev/Bachelorarbeit/experiments/.env; do
    if [ -f "$f" ] && grep -q '^RUNPOD_API_KEY=' "$f"; then
      grep '^RUNPOD_API_KEY=' "$f" | head -1 | cut -d= -f2- | tr -d '"'"'"' \r'
      return 0
    fi
  done
  die "RUNPOD_API_KEY not found"
}

api() {  # api METHOD PATH [BODY]
  local tmp code
  tmp="$(mktemp)"
  code="$(curl -sS -X "$1" -m 60 -o "$tmp" -w '%{http_code}' -H "Authorization: Bearer $(key)" \
          ${3:+-H 'Content-Type: application/json' -d "$3"} "$API$2")"
  case "$code" in 2*) cat "$tmp"; rm -f "$tmp" ;; *) echo "HTTP $code: $(cat "$tmp")" >&2; rm -f "$tmp"; exit 1 ;; esac
}

gql() { curl -sS -m 30 -H 'Content-Type: application/json' -H "Authorization: Bearer $(key)" -d "$1" "$GQL"; }

endpoint() {  # prints "IP PORT"
  local j; j="$(api GET "/pods/$1")"
  printf '%s' "$j" | jq -r '"\(.publicIp // "") \(.portMappings["22"] // "")"'
}

on_pod() {  # on_pod ID CMD
  local ip port; read -r ip port <<< "$(endpoint "$1")"
  [ -n "$ip" ] && [ -n "$port" ] || die "pod $1 has no SSH endpoint yet (still starting?)"
  ssh "${SSH_OPTS[@]}" -p "$port" "root@$ip" "$2"
}

rsync_to() {  # rsync_to ID SRC... DEST (DEST relative to REMOTE)
  local id="$1"; shift
  local ip port; read -r ip port <<< "$(endpoint "$id")"
  local dest="${!#}"; set -- "${@:1:$#-1}"
  rsync -az -e "ssh ${SSH_OPTS[*]} -p $port" "$@" "root@$ip:$REMOTE/$dest"
}

case "${1:-}" in
  price)
    gql '{"query":"query { gpuTypes(input: {id: \"NVIDIA A40\"}) { id securePrice communityPrice } myself { clientBalance } }"}' | jq .
    ;;
  create)
    body="$(jq -cn --arg n "genai-bart-$(date -u +%m%d-%H%M)" --arg i "$IMAGE" --arg g "$GPU" \
      '{name:$n, imageName:$i, gpuTypeIds:[$g], gpuCount:1, cloudType:"SECURE", volumeInGb:20,
        containerDiskInGb:30, volumeMountPath:"/workspace", ports:["22/tcp"], supportPublicIp:true}')"
    if [ "${2:-}" != "--go" ]; then echo "DRY RUN, add --go to create:"; printf '%s\n' "$body" | jq .; exit 0; fi
    api POST /pods "$body" | jq '{id, costPerHr, desiredStatus}'
    ;;
  status)
    api GET "/pods/$2" | jq '{id, name, desiredStatus, costPerHr, publicIp, portMappings}'
    ;;
  push)
    [ -z "$(git status --porcelain -- src scripts configs runs/*.sha256 requirements.txt)" ] \
      || die "uncommitted changes in src, scripts, configs or hashes: commit first, the pod runs HEAD"
    on_pod "$2" "mkdir -p $REMOTE/data/processed"
    git archive --format=tar HEAD src scripts configs reports runs/bert_train_pool.sha256 runs/corrupted_v2.sha256 \
        requirements.txt pyproject.toml | on_pod "$2" "tar -x -C $REMOTE"
    rsync_to "$2" data/processed/bert_train_pool.jsonl data/processed/corrupted_v2.jsonl data/processed/
    on_pod "$2" "cd $REMOTE && echo \"$(git rev-parse HEAD)\" > POD_COMMIT \
      && sha256sum data/processed/bert_train_pool.jsonl | cut -d' ' -f1 | grep -qx \$(cut -d' ' -f1 runs/bert_train_pool.sha256) \
      && sha256sum data/processed/corrupted_v2.jsonl | cut -d' ' -f1 | grep -qx \$(cut -d' ' -f1 runs/corrupted_v2.sha256) \
      && echo 'data hashes OK' \
      && grep -v '^torch==' requirements.txt | pip install -q -r /dev/stdin && pip install -q -e . --no-deps \
      && python -c 'import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))'"
    ;;
  run)
    on_pod "$2" "cd $REMOTE && mkdir -p runs && nohup bash -c 'PYTHONUTF8=1 timeout 3h $3; echo \"EXIT \$?\"' >> runs/pod_run.log 2>&1 < /dev/null & echo started"
    ;;
  log)
    on_pod "$2" "tail -n ${3:-25} $REMOTE/runs/pod_run.log | tr '\r' '\n' | tail -n ${3:-25}"
    ;;
  fetch)
    ip_port="$(endpoint "$2")"; read -r ip port <<< "$ip_port"
    mkdir -p models runs/preds
    rsync -az -e "ssh ${SSH_OPTS[*]} -p $port" "root@$ip:$REMOTE/models/bart_ft_v1" models/ || true
    rsync -az -e "ssh ${SSH_OPTS[*]} -p $port" --include='bart_*' --include='pod_run.log' --exclude='*' \
      "root@$ip:$REMOTE/runs/" runs/
    rsync -az -e "ssh ${SSH_OPTS[*]} -p $port" --include='bart_*' --exclude='*' "root@$ip:$REMOTE/runs/preds/" runs/preds/ || true
    ls -la models/bart_ft_v1 runs/bart_* runs/preds/bart_* 2>/dev/null || true
    ;;
  terminate)
    api DELETE "/pods/$2" > /dev/null && echo "terminated $2"
    ;;
  *)
    sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
    ;;
esac
