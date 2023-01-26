#!/bin/bash

for source in $(ls *.wav); do
  id="${source%.wav}"
  ffmpeg -i "${id}.wav" -ac 1 -ar 16000 -f f32le -hide_banner -loglevel quiet "${id}_pcm_f32le_16000.bin"
done
