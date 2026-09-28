HYDRA_FULL_ERROR=1 uv run fuelex train \
    --config-path ../../configs/ \
    --config-name train.yaml \
    criterion=cross_entropy \
    dataset=aef_fbfm40_Northwest_superzone_dataset_128px \
    lr_scheduler=multi_step_lr \
    model=seg_unet_resnet34 \
    optimizer=sgd \
    preprocessing=standard_augs \
    task=image_segmentation \