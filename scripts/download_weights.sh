#!/bin/bash
fileid="1iPC-ot1lYG4udkVHFc4TMOVWt_cImZL7"
filename="models/xray_model.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" --create-dirs -o ${filename}
rm cookie

cd models
unzip -a xray_model.zip
rm xray_model.zip
cd ..
