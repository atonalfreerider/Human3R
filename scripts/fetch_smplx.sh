#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# # SMPL-X model
echo -e "\nYou need to register at https://smpl-x.is.tue.mpg.de"
read -p "Username (SMPL-X):" username
read -p "Password (SMPL-X):" password
username=$(urle $username)
password=$(urle $password)

mkdir -p src/models
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip' -O './src/models/smplx.zip' --no-check-certificate --continue
unzip src/models/smplx.zip -d src/models/smplx
mv src/models/smplx/models/smplx/* src/models/smplx/
rm -rf src/models/smplx/models
rm -rf src/models/smplx.zip


# SMPL Male and Female model
echo -e "\nYou need to register at https://smpl.is.tue.mpg.de"
read -p "Username (SMPL):" username
read -p "Password (SMPL):" password
username=$(urle $username)
password=$(urle $password)

mkdir -p src/models/smpl
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.1.0.zip' -O './src/models/smpl/smpl.zip' --no-check-certificate --continue
unzip src/models/smpl/smpl.zip -d src/models/smpl/smpl
mv src/models/smpl/smpl/SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl src/models/smpl/SMPL_NEUTRAL.pkl
mv src/models/smpl/smpl/SMPL_python_v.1.1.0/smpl/models/basicmodel_f_lbs_10_207_0_v1.1.0.pkl src/models/smpl/SMPL_FEMALE.pkl
mv src/models/smpl/smpl/SMPL_python_v.1.1.0/smpl/models/basicmodel_m_lbs_10_207_0_v1.1.0.pkl src/models/smpl/SMPL_MALE.pkl
rm -rf src/models/smpl/smpl
rm -rf src/models/smpl/smpl.zip

# Supplementary SMPL and SMPL-X files
gdown --folder -O ./src/models/ https://drive.google.com/drive/folders/1JU7CuU2rKkwD7WWjvSZJKpQFFk_Z6NL7?usp=share_link
mv src/models/J_regressor_h36m.npy src/models/smpl/J_regressor_h36m.npy
mv src/models/smplx2smpl.pkl src/models/smplx/smplx2smpl.pkl