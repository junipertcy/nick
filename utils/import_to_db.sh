#!/bin/bash
for filename in ./*.json; do
    mongoimport --db nick --collection tickets --file tickets/"$filename" --type json
done
