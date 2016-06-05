#!/bin/bash

USER=$1
cat "$1" |awk '$0=$2' | grep -v null | while read id; do
  if [ ! -f "inreplyto/$id" ]; then
    twurl "/1.1/statuses/show.json?id=${id}" | jq -r .text > "inreplyto/$id"
    echo "$id"; cat "inreplyto/$id"
    sleep 10
  fi
done
