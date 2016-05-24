#!/bin/bash

USER=$1
URL="/1.1/statuses/user_timeline.json?screen_name=${USER}&count=200"
TMP=$(mktemp)
OUT=$1

while :; do
  if [ -f "$OUT" ]; then
    MAXID=$(( $(tail -1 "$OUT" | awk '$0=$1') - 1 ))
    echo "$MAXID"
    twurl "${URL}&max_id=${MAXID}" > "$TMP"
  else
    twurl "${URL}" > "$TMP"
  fi

  jq -r '.[] | "\(.id_str)\t\(.in_reply_to_status_id_str)\t\(.text|sub("\n";"改行文字"))"' "$TMP" >> "$OUT"
  sleep 15
done
rm "$TMP"
