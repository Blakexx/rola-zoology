#!/usr/bin/env bash
# Tripwire for the public mirror: fail the sync if anything private-shaped
# is present in the export tree. $1 = export dir.
set -euo pipefail
EXPORT="$1"
fail=0
# 1) no PDFs except the project's own note
if find "$EXPORT" -name '*.pdf' ! -path '*paper_configs*' | grep -q .; then
  echo "TRIPWIRE: unexpected PDF in export:"; find "$EXPORT" -name '*.pdf'; fail=1
fi
# 2) no agent/workflow-private files
for p in .claude CLAUDE.md; do
  [ -e "$EXPORT/$p" ] && { echo "TRIPWIRE: $p present in export"; fail=1; }
done
# 3) no cloud identifiers (pattern, not the literal — keep the literal out of this script too):
#    GCP project ids in this project match 'project-[0-9a-f-]{20,}'; buckets carry the same stem.
if grep -rIlE 'project-[0-9a-f]{8}-[0-9a-f]{4}' "$EXPORT" --exclude-dir=.git | grep -q .; then
  echo "TRIPWIRE: cloud project identifier found:"; grep -rIlE 'project-[0-9a-f]{8}-[0-9a-f]{4}' "$EXPORT" --exclude-dir=.git; fail=1
fi
exit $fail
