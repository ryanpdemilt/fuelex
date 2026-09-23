
export GEO_FILE="/home/rpdemilt/SIG/fuels/fuelex/data/geo/conus_landfire_zones_zones/us_lf_zones.shp"
export OUT_FILE="/home/rpdemilt/SIG/fuels/fuelex/data/fuelex_dataset/"

uv run fuelex sample \
     --geometry $GEO_FILE \
     --group_names superzone \
     --img_size 128 \
     --seed 2000 \
     --outfile $OUT_FILE \
     --samples_per_group 3000 \
     --dataset landfire \
     --splits test val train \
     --split-sizes 0.2 0.1 0.7 \
     --expname fuelex