#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# BEDLAM checkpoints
echo -e "\nYou need to register at https://bedlam.is.tue.mpg.de/"
read -p "Username (BEDLAM):" username
read -p "Password (BEDLAM):" password
username=$(urle $username)
password=$(urle $password)

# Download training labels
mkdir -p data/bedlam/download/
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=bedlam_labels/all_npz_12_training.zip' -O 'data/bedlam/download/all_npz_12_training.zip' --no-check-certificate --continue
unzip data/bedlam/download/all_npz_12_training.zip -d data/bedlam/download/
rm -rf data/bedlam/download/all_npz_12_training/agora*

# Download validation labels
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=bedlam_labels/all_npz_12_validation.zip' -O 'data/bedlam/download/all_npz_12_validation.zip' --no-check-certificate --continue
unzip data/bedlam/download/all_npz_12_validation.zip -d data/bedlam/download/

# 30 sequence labals in total
mkdir -p data/bedlam/processed_labels/
mv data/bedlam/download/all_npz_12_training/* data/bedlam/processed_labels/
mv data/bedlam/download/all_npz_12_validation/* data/bedlam/processed_labels/
rm -rf data/bedlam/download/all_npz_12_training.zip
rm -rf data/bedlam/download/all_npz_12_validation.zip