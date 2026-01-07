WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $WORKDIR
LOG_DIR="$WORKDIR/.log/judge"
mkdir -p $LOG_DIR
DATE=`date +%Y%m%d_%H%M%S`
LOG_FILE="$LOG_DIR/$DATE.log"
echo "log file: ${LOG_FILE}"

MODEL_PATH=YOUR_PATH/Qwen/Qwen3-Omni-30B-A3B-Instruct
conda activate qwen3_caption_vllm
python src/FG-CE.py -p $MODEL_PATH >> $LOG_FILE 2>&1
