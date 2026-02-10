#!/bin/bash
# Training monitoring script
# Usage: ./monitor_training.sh [optional: reward_threshold]

REWARD_THRESHOLD=${1:-1.5}
LOG_DIR="/home/mlangtry/IsaacLabLego/logs/rl_games/franka_stack"

echo "=========================================="
echo "Isaac Lab Training Monitor"
echo "=========================================="
echo ""

# Find the latest run
LATEST_RUN=$(ls -td $LOG_DIR/2026-* 2>/dev/null | head -1)

if [ -z "$LATEST_RUN" ]; then
    echo "❌ No training runs found"
    exit 1
fi

echo "📁 Latest run: $(basename $LATEST_RUN)"
echo ""

# Check if screen session is running
if screen -ls | grep -q isaac_training; then
    echo "✅ Training is RUNNING in screen session 'isaac_training'"
    echo "   To attach: screen -r isaac_training"
    echo "   To detach: Ctrl+A then D"
else
    echo "⚠️  Screen session 'isaac_training' not found"
fi
echo ""

# Get latest checkpoints
echo "📊 Latest Checkpoints:"
ls -lht $LATEST_RUN/nn/*.pth 2>/dev/null | head -5 | awk '{print "   "$9" - "$5" ("$6" "$7" "$8")"}'
echo ""

# Get best checkpoint
BEST_CHECKPOINT=$(ls -lht $LATEST_RUN/nn/franka_stack.pth 2>/dev/null)
if [ -n "$BEST_CHECKPOINT" ]; then
    echo "🏆 Best checkpoint: franka_stack.pth"
    echo "   $(echo $BEST_CHECKPOINT | awk '{print $5" ("$6" "$7" "$8")"}')"
    echo ""
fi

# Extract reward from latest checkpoint filename
LATEST_CHECKPOINT=$(ls -t $LATEST_RUN/nn/last_*.pth 2>/dev/null | head -1)
if [ -n "$LATEST_CHECKPOINT" ]; then
    EPOCH=$(echo $LATEST_CHECKPOINT | grep -oP 'ep_\K[0-9]+')
    REWARD=$(echo $LATEST_CHECKPOINT | grep -oP 'rew_\K[0-9.]+')

    echo "📈 Latest Epoch: $EPOCH"
    echo "📈 Latest Reward: $REWARD"

    # Check if reward exceeds threshold
    if (( $(echo "$REWARD > $REWARD_THRESHOLD" | bc -l) )); then
        echo ""
        echo "🎉 MILESTONE REACHED! Reward $REWARD > threshold $REWARD_THRESHOLD"
    fi
else
    echo "⚠️  No checkpoints found yet"
fi

echo ""
echo "=========================================="
echo "Recent Training Log (last 10 lines):"
echo "=========================================="
tail -10 $(ls -t /tmp/training_resume_*.log 2>/dev/null | head -1) 2>/dev/null || echo "No log file found"
