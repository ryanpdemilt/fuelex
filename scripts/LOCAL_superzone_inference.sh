EXP_DIR="/home/rpdemilt/SIG/fuels/fuelex/output/20260929_174647_b027fc_seg_segformer_fbfm40_Southern_Appalachians_2024_128px/"
GEO_DIR="/home/rpdemilt/SIG/fuels/fuelex/data/geo/conus_landfire_zones_zones/us_lf_zones.shp"
OUT_DIR="/home/rpdemilt/SIG/fuels/fuelex/maps"
LOG_DIR="/home/rpdemilt/SIG/fuels/fuelex/logs"

uv run fuelex inference \
    --exp-dir $EXP_DIR \
    --device cuda:0 \
    --geometry-path $GEO_DIR \
    --group-name ZONE_NAME\
    --groups 57 \
    --year 2025 \
    --out-path $OUT_DIR \
    --scale 30 \
    --crs EPSG:5070 \
    --source ee \
    --project pyregence-ee \
    --batch-size 64 \
    --band-groups 4 \
    --padding 5 \
    --log-dir $LOG_DIR\
    --n-jobs 8 \
