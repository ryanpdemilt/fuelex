export WORK_DIR="/home/rpdemilt/SIG/fuels/fuelex/data/fuelex_dataset/"
export GEO_FILE="/home/rpdemilt/SIG/fuels/fuelex/data/fuelex_dataset/fuelex_superzone_sampling_256px_3000im.geojson"
export LANDFIRE_SRC="/home/rpdemilt/SIG/fuels/fuelex/data/landfire"

uv run fuelex retrieve \
    --geometry $GEO_FILE \
    --groups "Northwest" "Pacific South" "Great Basin" \
    --work-dir $WORK_DIR \
    --cache-dir $LANDFIRE_SRC \
    --dataset fbfm40 \
    --year 2025 \
    --n-jobs 1