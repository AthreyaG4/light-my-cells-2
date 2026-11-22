#!/usr/bin/env bash
# Run the whole pipeline in order, stopping with a message as soon as a check fails.
#
#   bash run_all.sh               validate, reconcile, build the CSV, download, split, upload (keeps local images)
#   bash run_all.sh --delete      same, but delete each study's images after it is uploaded (frees disk)
#   bash run_all.sh --dry-run     run every check up to the upload and show what would be downloaded; download/upload nothing
#
# Order: validate -> fetch the full stacks reconcile needs -> reconcile -> build the CSV -> download the
# CSV's files -> make the train/validation split -> upload. (reconcile needs image files, and the CSV needs
# reconcile, so the stacks come first. The upload names each shard after its split.)
#
# Re-running is safe: downloads skip files already present, uploads skip studies already in the repo.
# Needs bash (Git Bash or WSL on Windows) and uv. HF_TOKEN goes in .env; REPO is set in upload_to_hub.py.

set -euo pipefail
cd "$(dirname "$0")"
unset VIRTUAL_ENV  # avoids uv's "VIRTUAL_ENV does not match" warning

EXPECTED_STUDIES=87
SLACK_GB=2  # keep this much disk free beyond the download
TOTAL_STEPS=7

DRY=0
DELETE=""
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY=1 ;;
    --delete) DELETE="--delete" ;;
    *) echo "unknown option: $arg (use --dry-run and/or --delete)"; exit 2 ;;
  esac
done

PY="uv run python"
OUT="$(mktemp)"
trap 'rm -f "$OUT"' EXIT
STEP="starting"
STEP_START=$SECONDS

# each step announces itself, and the previous one is reported as passed (with its run time) when the next begins
done_step() { [ "$STEP" = "starting" ] || echo "--> PASSED: $STEP ($((SECONDS - STEP_START)) s)"; }
step() { done_step; STEP="$1"; STEP_START=$SECONDS; echo; echo "=== $1 ==="; }
fail() { echo; echo "STOPPED at: $STEP"; echo "Reason: $*"; exit 1; }
# a script failing (non-zero exit) also stops the run, with the step it was in
trap 'echo; echo "STOPPED at: $STEP (command failed, see the output above)"' ERR
free_gb() { df -k . | awk 'NR==2 {printf "%d", $4 / 1048576}'; }

# ---------------------------------------------------------------- 0. preflight
step "0/${TOTAL_STEPS} preflight"
command -v uv >/dev/null || fail "uv is not installed or not on PATH"
[ -d "data/Study_info_filenames_and_optimal focus" ] || fail "data/Study_info_filenames_and_optimal focus (the TSVs) is missing"
grep -qiE '^[[:space:]]*hf_token[[:space:]]*=' .env 2>/dev/null || fail ".env has no HF_TOKEN line"
grep -q '^REPO = "your-username' upload_to_hub.py && fail "set REPO in upload_to_hub.py first"
# check the token works now rather than after a 30 GB download (the token itself is never printed)
$PY -c "
from dotenv import load_dotenv
from huggingface_hub import HfApi
load_dotenv('.env')
HfApi().whoami()
print('Hugging Face token OK')
" || fail "the Hugging Face token in .env was rejected or could not be checked"
echo "check ok: uv found, TSV folder present, .env has a token line, REPO is set, token accepted"

# ---------------------------------------------------------------- 1. validate
step "1/${TOTAL_STEPS} data_validation.py (TSVs against each other)"
$PY data_validation.py | tee "$OUT"
grep -q "^Checked ${EXPECTED_STUDIES} studies" "$OUT" || fail "expected ${EXPECTED_STUDIES} studies in the TSV folder"
out_of_range=$(sed -n 's/^\[best_z outside the listed slices\]  \([0-9][0-9]*\) channel cases.*/\1/p' "$OUT")
[ "${out_of_range:-missing}" = "0" ] || fail "best_z values point outside the listed slices (${out_of_range:-count not found}); the best_z rules no longer hold, check before going on"
to_compute=$(sed -n 's/^BEST Z STILL TO COMPUTE (run reconcile.py): \([0-9][0-9]*\) channel cases.*/\1/p' "$OUT")
[ -n "$to_compute" ] || fail "could not read the 'still to compute' count from the validator"
echo "check ok: ${EXPECTED_STUDIES} studies read, no best_z outside the listed slices, ${to_compute} channel cases still need a computed best z"

# ---------------------------------------------------------------- 2. stacks for reconcile
# Channels with files but no best_z (Study 59 PC) need their whole z-stack on disk to be scored.
# This has to happen before create_pairs.py, which otherwise leaves those rows out, and before the
# pair download, which only fetches what the CSV lists.
step "2/${TOTAL_STEPS} download_data.py --stacks (full z-stacks that reconcile.py must score)"
if [ "$to_compute" = "0" ]; then
  echo "Nothing to score: every best z is in the TSVs or already in computed_best_focus.csv."
elif [ "$DRY" = 1 ]; then
  $PY download_data.py --stacks --dry-run
  echo; echo "Dry run stops here: $to_compute channel cases need those stacks on disk before reconcile.py,"
  echo "create_pairs.py and the pair download can be checked. Nothing was downloaded."
  exit 0
else
  $PY download_data.py --stacks
  echo "check ok: stacks downloaded, each file verified against its CRC"
fi

# ---------------------------------------------------------------- 3. reconcile
step "3/${TOTAL_STEPS} reconcile.py (best focus for channels the TSV leaves out)"
$PY reconcile.py | tee "$OUT"
if grep -q "^Failed " "$OUT"; then fail "reconcile.py could not process some stacks (see 'Failed' lines)"; fi
$PY data_validation.py > "$OUT"
to_compute=$(sed -n 's/^BEST Z STILL TO COMPUTE (run reconcile.py): \([0-9][0-9]*\) channel cases.*/\1/p' "$OUT")
[ "${to_compute:-missing}" = "0" ] || fail "${to_compute:-?} channel cases still have no best z. Their stack files are not all on disk or could not be read; run: python download_data.py --stacks, then re-run"
echo "check ok: reconcile finished, 0 channel cases left to compute"

# ---------------------------------------------------------------- 4. pairs
step "4/${TOTAL_STEPS} create_pairs.py (input_output_pairs.csv)"
$PY create_pairs.py | tee "$OUT"
if grep -q "^Left out" "$OUT"; then fail "create_pairs.py left rows out because a best z is missing (see 'Left out' lines)"; fi
rows=$(( $(wc -l < input_output_pairs.csv) - 1 ))
[ "$rows" -gt 0 ] || fail "input_output_pairs.csv has no rows"
echo "input_output_pairs.csv: $rows rows"
if [ -n "$DELETE" ]; then
  $PY download_data.py --prune-stacks  # drop the stack slices the CSV does not use
fi
echo "check ok: no rows left out, CSV written"

# ---------------------------------------------------------------- 5. download
step "5/${TOTAL_STEPS} download_data.py (only the files in the CSV, only for studies not yet in the Hub repo)"
# Studies already uploaded are skipped, so a re-run after --delete does not fetch them again.
PENDING=$($PY - <<'EOF'
import pandas as pd
from huggingface_hub import HfApi
from upload_to_hub import studies_in_repo

studies = sorted(pd.read_csv("input_output_pairs.csv")["study"].unique())
done = studies_in_repo(HfApi())
print(" ".join(s for s in studies if s not in done))
EOF
) || fail "could not list the studies already in the Hub repo"
if [ -z "$PENDING" ]; then
  echo "Every study is already in the Hub repo; nothing to download."
else
  echo "Studies still to download and upload: $(echo "$PENDING" | wc -w)"
  $PY download_data.py --dry-run $PENDING | tee "$OUT"
  if grep -q "^PROBLEM" "$OUT"; then fail "some files in the CSV are not in the zips (see 'PROBLEM' lines)"; fi
  need_gb=$(sed -n 's/^Would fetch \([0-9.]*\) GB.*/\1/p' "$OUT")
  [ -n "$need_gb" ] || fail "could not read the download size from the dry run"
  have_gb=$(free_gb)
  echo "To download: ${need_gb} GB (compressed). Free disk: ${have_gb} GB."
  if awk -v need="$need_gb" -v have="$have_gb" -v slack="$SLACK_GB" 'BEGIN { exit !(need + slack > have) }'; then
    if [ "$DRY" = 1 ]; then
      echo "WARNING: not enough free disk for this download (a real run would stop here)."
    else
      fail "not enough free disk: need ${need_gb} GB (+${SLACK_GB} GB spare), have ${have_gb} GB. Free some space or download in batches (python download_data.py Study_01 Study_02 ...)"
    fi
  fi
  if [ "$DRY" = 1 ]; then
    echo "Dry run: skipping the download."
  else
    echo "Downloading now (this is the long step; each study is listed as it starts)..."
    $PY download_data.py $PENDING
    # every file those studies need must now exist and be non-empty
    $PY - $PENDING <<'EOF' || fail "some files are missing after the download (listed above); re-run to retry"
import sys
from pathlib import Path
import pandas as pd
from download_data import needed_files

pairs = pd.read_csv("input_output_pairs.csv")
bad = [f"{s}/{n}" for s in sys.argv[1:] for n in needed_files(pairs, s)
       if not (Path("data") / s / n).exists() or (Path("data") / s / n).stat().st_size == 0]
print(f"download check: {len(bad)} missing or empty files")
for b in bad[:20]:
    print("  ", b)
sys.exit(1 if bad else 0)
EOF
    echo "check ok: every file the CSV needs for these studies is on disk and non-empty"
  fi
fi

# ---------------------------------------------------------------- 6. splits
# Needs only the CSV, so it also runs in a dry run. Whole studies are held out for validation.
step "6/${TOTAL_STEPS} create_splits.py (train / validation, whole studies held out)"
$PY create_splits.py | tee "$OUT"
$PY - <<'EOF' || fail "splits.csv failed its checks (listed above)"
import sys
import pandas as pd

ORGANELLES = ["nucleus", "mitochondria", "tubulin", "actin"]
pairs = pd.read_csv("input_output_pairs.csv")
splits = pd.read_csv("splits.csv")
problems = []

images = pairs.drop_duplicates(["study", "image_id"])
if len(splits) != len(images) or splits.duplicated(["study", "image_id"]).any():
    problems.append(f"splits.csv has {len(splits)} rows for {len(images)} images (or duplicates)")
if set(splits["split"]) != {"train", "validation"}:
    problems.append(f"expected both 'train' and 'validation', found {sorted(set(splits['split']))}")
both = set(splits[splits.split == "train"].study) & set(splits[splits.split == "validation"].study)
if both:
    problems.append(f"studies on both sides of the split: {sorted(both)[:5]}")
merged = pairs.merge(splits, on=["study", "image_id"], how="left")
if merged["split"].isna().any():
    problems.append(f"{int(merged['split'].isna().sum())} pair rows have no split")
two = merged.groupby(["study", "image_id"])["split"].nunique()
if (two > 1).any():
    problems.append("an image's rows are split across train and validation")
for o in ORGANELLES:
    n_val = int((merged[f"{o}_file"].notna() & (merged.split == "validation")).sum())
    if n_val == 0:
        problems.append(f"no validation images for {o}")

print(f"split check: {len(problems)} problem(s)")
for p in problems:
    print("  ", p)
sys.exit(1 if problems else 0)
EOF
echo "check ok: every image has one split, no study on both sides, two-input images together, every organelle has validation images"

if [ "$DRY" = 1 ]; then
  done_step
  echo; echo "Dry run finished: every check up to the upload passed. Nothing was downloaded or uploaded."
  exit 0
fi

# ---------------------------------------------------------------- 7. upload
step "7/${TOTAL_STEPS} upload_to_hub.py (private dataset, one shard per study named after its split)"
echo "Uploading now (each batch is announced when its commit finishes)..."
$PY upload_to_hub.py $DELETE
# every study must now be in the repo under the split splits.csv gives it
$PY - <<'EOF' || fail "the Hub repo is missing shards or has them under the wrong split (listed above); re-run to retry"
import sys
from huggingface_hub import HfApi
from upload_to_hub import REPO, studies_in_repo, study_splits

splits = study_splits()
in_repo = studies_in_repo(HfApi())
missing = [s for s in splits if s not in in_repo]
wrong = [s for s in splits if s in in_repo and in_repo[s] != splits[s]]
print(f"upload check: {len(splits) - len(missing)} of {len(splits)} studies are in {REPO}, {len(wrong)} under the wrong split")
for s in missing:
    print("  missing", s)
for s in wrong:
    print(f"  {s} is in the repo as {in_repo[s]} but splits.csv says {splits[s]}")
sys.exit(1 if missing or wrong else 0)
EOF
echo "check ok: every study has its shard in the Hub repo under the right split"

done_step
echo; echo "All steps finished (total ${SECONDS} s)."
