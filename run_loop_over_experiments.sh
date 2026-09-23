#!/bin/bash

ATTACKS=(neurotoxin) #BAs: neurotoxin badnets scaling iba dba // MPAs: krum, trim, gauss 
DEFENSES=(none)   #BDs: weakdp clip deepsight mmad krum ubar scclip dfldual balance abalance trim spp flame
DATASETS=(har) #cifar10 cifar100 mnist femnist har gtsrb fashionmnist nslkdd unsw_nb15 nbaiot

#krum flame mmad spp deepsight clip weakdp 


# Notes: 
# HAR dataset must be downloaded manually and saved in data/har/
# HAR is not CV dataset run it with Neurotoxin, Badnets or Scaling
# Download
# wget https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip
# unzip human+activity+recognition+using+smartphones.zip
# unzip UCI\ HAR\ Dataset.zip
# cp -r UCI\ HAR\ Dataset/*  har
# Repeat the same process for the other datasets if needed


for atk in "${ATTACKS[@]}"; do
  for def in "${DEFENSES[@]}"; do
    for data in "${DATASETS[@]}"; do
      ./run_experiment.sh "$atk" "$def" "$data" "decentralized" #centralized
    done
  done
done
