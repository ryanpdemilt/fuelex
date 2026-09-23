export CACHE_DIR="/home/rpdemilt/SIG/fuels/fuelex/tmp/cache/"
export WORK_DIR="/home/rpdemilt/SIG/fuels/fuelex/data/fuelex_dataset/"
export GEO_FILE="/home/rpdemilt/SIG/fuels/fuelex/data/fuelex_dataset/fuelex_superzone_sampling_256px_3000im.geojson"

uv run fuelex retrieve \
    --dataset aef \
    --geometry $GEO_FILE \
    --year 2024 \
    --groups "Pacific South" "Great Basin" \
    --cache-dir $CACHE_DIR \
    --cache-size 20 \
    --work-dir $WORK_DIR \
    --dst-scale 30 \
    --dst-crs EPSG:5070 \
    --source source.coop \
    --clean \
    --n-jobs 4