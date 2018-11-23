#!/bin/bash

settings_file="/home/valts/Documents/AirSim/settings.json"
fly_settings_file="./settings_fly.json"

rm $settings_file
cp $fly_settings_file $settings_file
