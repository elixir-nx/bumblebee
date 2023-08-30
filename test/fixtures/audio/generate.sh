#!/bin/bash

cd "$(dirname "$0")"

for source in $(ls **/*.{wav,mp3}); do
  name="${source%.*}"
  ffmpeg -i $source -ac 1 -ar 16000 -f f32le -hide_banner -loglevel quiet "${name}_pcm_f32le_16000.bin"
done
