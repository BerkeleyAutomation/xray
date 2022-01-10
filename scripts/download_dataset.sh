#!/bin/bash
fileid="1Ft_Y9DcWCaE1tfgej8JYiNbLffrXMPEs"
filename="datasets/images.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" --create-dirs -o ${filename}
rm cookie

cd datasets
unzip -a images.zip
mv images example_dataset
rm images.zip
cd ..
