#!/bin/bash
curl -L "https://www.dropbox.com/scl/fi/dfs6hs02arab7owdffnoa/MedClinSum.zip?rlkey=fdr8fdjterrygi265d5avw8r6&st=15350uet&dl=1" -o "archive.zip"
unzip "archive.zip"
rm "archive.zip"

curl -L "https://www.dropbox.com/scl/fi/8bkjdlgh5w51p0kqtur66/MultiClinSum_rationale.zip?rlkey=qtsjwexzigshfyds7nkn1pod0&st=rw68uwcf&dl=1" -o "archive.zip"
unzip "archive.zip"
rm "archive.zip"
