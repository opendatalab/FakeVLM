python scripts/eval.py \
  --model_path /path/to/your/checkpoint \
  --val_batch_size 16 \
  --workers 16 \
  --output_path results/fakevlm.json \
  --test_json_file "/path/to/your/test.json" \
  --data_base_test "path/to/your/test_images" \