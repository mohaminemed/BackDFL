#!/bin/bash

ATTACKS=(iba) # a3fl neurotoxin) # BAs: neurotoxin badnets scaling iba dba // MPAs: krum, trim, gauss  neurotoxin 
DEFENSES=(abalance flame) #trim spp weakdp clip  deepsight mmad krum ubar scclip dfldual balance abalance trim spp clip weakdp flame abalance none
DATASETS=(fashionmnist) #cifar10 mnist femnist har


# Notes: 
# HAR dataset must be downloaded manually and saved in data/har/
# HAR is not CV dataset run it with Neurotoxin, Badnets or Scaling
# Download
# wget https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip
# unzip human+activity+recognition+using+smartphones.zip
# unzip UCI\ HAR\ Dataset.zip
# cp -r UCI\ HAR\ Dataset/*  har


for atk in "${ATTACKS[@]}"; do
  for def in "${DEFENSES[@]}"; do
    for data in "${DATASETS[@]}"; do
      ./run_experiment.sh "$atk" "$def" "$data" "decentralized" #centralized
    done
  done
done
