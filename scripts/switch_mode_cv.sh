#!/bin/bash

settings_file="/home/valts/Documents/AirSim/settings.json"
cv_ettings_file="./airsim_settings/settings_cv_overhead.json"

rm $settings_file
cp $cv_ettings_file $settings_file
