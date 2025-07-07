#########
# ...existing hpc setup for sbatch...

module load cuda/12.3

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
start=$SECONDS

### Conda ENV
echo "Activating VLA environment"
source ~/anaconda3/bin/activate
conda activate openvla

echo "Running finetuning script for OpenVLA"

## path export
export PYTHONPATH=/home/<user>/openvla/rlds_dataset_builder/:$PYTHONPATH

## add your API key for Weights & Biases setup
export WANDB_API_KEY="your_wandb_api_key_here"



torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir <PATH TO BASE DATASETS DIR> \
  --dataset_name bridge_orig \
  --run_root_dir <PATH TO LOG/CHECKPOINT DIR> \
  --adapter_tmp_dir <PATH TO TEMPORARY DIR TO SAVE ADAPTER WEIGHTS> \
  --lora_rank 32 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug <True or False> \
  --wandb_project <PROJECT> \
  --wandb_entity <ENTITY> \
  --save_steps <NUMBER OF GRADIENT STEPS PER CHECKPOINT SAVE>

echo "Deactivating openvla environment"
# deactivate
conda deactivate

duration=$(( SECONDS - start ))

echo "Completed in $duration s"

echo "Done."
