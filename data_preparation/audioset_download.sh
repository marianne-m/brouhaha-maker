#!/bin/bash

# This script downloads the wavfile of Audioset from YouTube.
# The files are clipped and re-sampled to 16kHz.
#
# Usage : ./audioset_download.sh metadata.csv

METADATA_FILE=$1
PARTITION=$(echo $METADATA_FILE | cut -d'.' -f1)
SAMPLE_RATE=16000

mkdir $PARTITION

# download_and_clip(videoID, startTime, endTime)
download_and_clip() {
  echo "Downloading $1 ($2 to $3)..."
  outname="$PARTITION/$1_$2"

  if [ -f "${outname}.wav" ]; then
    echo "Already downloaded."
    return
  fi

  yt-dlp https://youtube.com/watch?v=$1 \
    --cookies youtube.com_cookies.txt \
    --quiet --extract-audio --audio-format wav \
    --output "${outname}.%(ext)s"

  if [ $? -eq 0 ]; then
    # If we don't pipe `yes`, ffmpeg seems to steal a
    # character from stdin. I have no idea why.
    yes | ffmpeg -loglevel quiet -i "${outname}.wav" -ar $SAMPLE_RATE \
      -ss "$2" -to "$3" "${outname}_out.wav"
    mv ${outname}_out.wav $outname.wav
  else
    # Give the user a chance to Ctrl+C.
    sleep 1
  fi
}

grep -v '^#' $METADATA_FILE | while read line
do
  download_and_clip $(echo "$line" | sed -E 's/, / /g')
done