#!/usr/bin/env bash
set -euo pipefail

DEST=${DEST:-data}
mkdir -p "${DEST}"

echo "Collections and licenses are listed in glass-sources.txt."
echo "Resolve each url, accept its terms, and stage the archives under ${DEST}/<tag>."
echo
grep -vE '^\s*#' glass-sources.txt | awk -F'|' 'NF>=4 {gsub(/ /,"",$1); print $1" -> "$4}'
echo
echo "figshare and grand-challenge gate automated downloaders; fetch those from a browser session."
