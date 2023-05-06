# Euler
custom GPT full fine-tune (domain adaptation) based on run_clm.py

## How to run
```bash
python run_clm.py \
    --model_name_or_path {MODEL_NAME} \
    --model_revision {MODEL_BRANCH} \
    --cache_dir {CACHE_PATH} \
    --torch_dtype auto \
    --train_file {DATA_PATH} \
    --validation_split_percentage 0 \
    --preprocessing_num_workers 64 \
    --dataloader_drop_last True \
    --per_device_train_batch_size 8 \
    --do_train \
    --num_train_epochs 1 \
    --save_strategy no \
    --report_to none \
    --output_dir {OUTPUT_PATH}
```

