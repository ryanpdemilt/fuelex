
export GEO_FILE="./data/geo/conus_landfire_zones_zones/us_lf_zones.shp"
export OUT_FILE="./data/"

mkdir -p $OUT_FILE

uv run fuelex sample \
     --geometry $GEO_FILE \
     --group_names superzone \
     --img_size 128 \
     --seed 2000 \
     --outfile $OUT_FILE \
     --samples_per_group 10 \
     --dataset landfire \
     --splits test val train \
     --split-sizes 0.2 0.1 0.7 \
     --expname fuelex \