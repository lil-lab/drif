#!/usr/bin/env bash

results_folder=$1
names=( "fpv_" "action_" "D_u" "D_u_r" "F_c" "F_w" "S_w" "R_w" "R_w")
for name in "${names[@]}"
do
	#ffmpeg -start_number 0 -framerate 10 -i ${results_folder}/${name}%d.png -vcodec libx264 -acodec aac -qp 0 ${results_folder}/${name}.mp4
    ffmpeg -start_number 0 -framerate 5 -i ${results_folder}/${name}%d.png -c:v libx264 -preset slow  -profile:v high -level:v 4.0 -pix_fmt yuv420p -crf 22 -codec:a aac -vf scale="iw*4:ih*4" ${results_folder}/${name}.mp4
done