# proofwriter,gsm8k,mathqa,logicnli,logiqa,mmlu_redux,arc_c
# ${MODELS_ROOT}/Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554
# ${MODELS_ROOT}/Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
python3 ${SEC5_ROOT}/infer/infer.py \
  --model ${MODELS_ROOT}/Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --tasks arc_c \
  --n_samples 5 \
  --max_new_tokens 1024 \
  --reflection \
  --out_dir ${SEC5_ROOT}/infer/output/eval_arc_c_v1