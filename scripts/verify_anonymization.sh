#!/usr/bin/env bash
set -euo pipefail

if ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"; then
  IN_GIT_REPO=1
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  IN_GIT_REPO=0
fi
cd "$ROOT"

FAIL=0

IDENTITY_PATTERN='talha|chafekar|talha1503|talha15032|zihan|zihanwang|zihanwang314|northwestern|zhejiang|dongqianyu|lixinyuan|saberrr|zju|\blxy\b|jinhui|junqiu|fangjing|zhijie|gaoning|yejinhui|jye624|hkust|fudan|sust|pku\.edu|gmail\.com|outlook\.com|163\.com'
PATH_PATTERN='(/Users/|/home/[A-Za-z0-9_.-]+|/mnt/[A-Za-z0-9_.-]+|/gpfs/|/project/|/pfs/users/)'
PUBLIC_LINK_PATTERN='latency-sensitive-bench|anonymous-latency-bench|anonymous-iclr|github\.com/ZihanWang314|git@github\.com:ZihanWang314|wandb\.ai/saberrr-zju|starvla\.github\.io|huggingface\.co/StarVLA|arxiv\.org/abs/2604'
EMAIL_PATTERN='[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'

run_search() {
  local pattern="$1"
  local ignore_case="${2:-0}"

  if [[ "$IN_GIT_REPO" -eq 1 ]]; then
    if [[ "$ignore_case" -eq 1 ]]; then
      git ls-files -z ':!:scripts/verify_anonymization.sh' ':!:*.zip' ':!:*.tar' ':!:*.tar.gz' |
        xargs -0 rg -n --text -i -e "$pattern"
    else
      git ls-files -z ':!:scripts/verify_anonymization.sh' ':!:*.zip' ':!:*.tar' ':!:*.tar.gz' |
        xargs -0 rg -n --text -e "$pattern"
    fi
  else
    if [[ "$ignore_case" -eq 1 ]]; then
      rg -n --text -i --hidden --no-ignore --glob '!.git/**' --glob '!*.zip' --glob '!*.tar' --glob '!*.tar.gz' --glob '!scripts/verify_anonymization.sh' -e "$pattern" .
    else
      rg -n --text --hidden --no-ignore --glob '!.git/**' --glob '!*.zip' --glob '!*.tar' --glob '!*.tar.gz' --glob '!scripts/verify_anonymization.sh' -e "$pattern" .
    fi
  fi
}

check_no_matches() {
  local label="$1"
  local pattern="$2"
  local ignore_case="${3:-0}"

  echo
  echo "== ${label} =="

  set +e
  run_search "$pattern" "$ignore_case"
  local status=$?
  set -e

  if [[ "$status" -eq 0 ]]; then
    echo "FAIL: matches found above."
    FAIL=1
  elif [[ "$status" -eq 1 ]]; then
    echo "OK: no matches."
  else
    echo "ERROR: search failed with status ${status}."
    FAIL=1
  fi
}

check_no_matches "Known identity strings" "$IDENTITY_PATTERN" 1
check_no_matches "Personal/local absolute paths" "$PATH_PATTERN" 0
check_no_matches "Public repo/model/citation identifiers" "$PUBLIC_LINK_PATTERN" 1

echo
echo "== Email-like strings =="
set +e
EMAIL_HITS="$(run_search "$EMAIL_PATTERN" 0 | grep -v 'git@github.com')"
email_status=$?
set -e
if [[ "$email_status" -eq 0 ]]; then
  echo "$EMAIL_HITS"
  echo "FAIL: email-like matches found above."
  FAIL=1
elif [[ "$email_status" -eq 1 ]]; then
  echo "OK: no email-like matches other than generic git@github.com SSH rewrites."
else
  echo "ERROR: email check failed with status ${email_status}."
  FAIL=1
fi

echo
echo "== Git whitespace check =="
if [[ "$IN_GIT_REPO" -eq 0 ]]; then
  echo "OK: skipped outside a git repository."
elif git diff --check; then
  echo "OK: git diff --check passed."
else
  echo "FAIL: git diff --check found issues."
  FAIL=1
fi

echo
echo "== Summary =="
if [[ "$FAIL" -eq 0 ]]; then
  echo "Anonymization verification passed."
else
  echo "Anonymization verification failed. Review the matches above."
fi

exit "$FAIL"
