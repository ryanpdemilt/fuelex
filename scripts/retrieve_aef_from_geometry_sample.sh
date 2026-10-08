export CACHE_DIR="/home/rdemilt/redsky/fuelex/data/cache/"
export WORK_DIR="/mnt/data/rdemilt/fuelex_sc/"
export GEO_FILE="/home/rdemilt/redsky/fuelex/data/"

uv run fuelex retrieve \
    --dataset aef \
    --geometry $GEO_FILE/fuelex_test_superzone_sampling_128px_3000im.geojson $GEO_FILE/fuelex_val_superzone_sampling_128px_3000im.geojson $GEO_FILE/fuelex_train_superzone_sampling_128px_3000im.geojson \
    --year 2024 \
    --groups "Great Lakes"\
    --cache-dir $CACHE_DIR \
    --cache-size 10 \
    --work-dir $WORK_DIR \
    --dst-scale 30 \
    --dst-crs EPSG:5070 \
    --source source.coop \
    --clean \
    --n-jobs 8