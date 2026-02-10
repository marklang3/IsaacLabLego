#!/bin/bash
# Monitor LEGO precision stacking training progress (50k epochs)
# Usage: ./monitor_lego_training.sh

LOG_DIR="/home/mlangtry/IsaacLabLego/logs/rl_games/franka_stack"
LATEST_RUN=$(ls -t $LOG_DIR | head -1)
LOG_FILE="/home/mlangtry/IsaacLabLego/logs/lego_training_50k.log"

echo "=========================================================="
echo "LEGO 2-Cube Precision Stacking Training Monitor"
echo "=========================================================="
echo ""

# Check process status
if pgrep -f "Isaac-Stack-2Cube-LEGO" > /dev/null; then
    echo "✅ Training process is RUNNING"
else
    echo "❌ Training process is NOT running"
fi
echo ""

# Get latest checkpoint
LATEST_CHECKPOINT=$(ls -t $LOG_DIR/$LATEST_RUN/nn/*.pth 2>/dev/null | grep -v "^franka_stack.pth$" | head -1)
if [ -f "$LATEST_CHECKPOINT" ]; then
    CHECKPOINT_NAME=$(basename "$LATEST_CHECKPOINT")
    EPOCH=$(echo "$CHECKPOINT_NAME" | grep -oP 'ep_\K[0-9]+')
    REWARD=$(echo "$CHECKPOINT_NAME" | grep -oP 'rew_\K[0-9.]+')
    echo "📊 Latest Checkpoint:"
    echo "   Epoch: $EPOCH / 50000 ($(echo "scale=1; $EPOCH*100/50000" | bc)%)"
    echo "   Reward: $REWARD"
    echo "   Time: $(stat -c %y "$LATEST_CHECKPOINT" | cut -d'.' -f1)"
else
    echo "⏳ No checkpoints saved yet (saves every 100 epochs)"
fi
echo ""

# Show recent training progress from log
if [ -f "$LOG_FILE" ]; then
    echo "📈 Recent Training Progress:"
    tail -20 "$LOG_FILE" | grep "fps total:" | tail -5
else
    echo "⚠️  Log file not found: $LOG_FILE"
fi
echo ""

# Performance stats
if [ -f "$LOG_FILE" ]; then
    LATEST_FPS=$(tail -20 "$LOG_FILE" | grep "fps total:" | tail -1 | grep -oP 'fps total: \K[0-9]+')
    LATEST_EPOCH=$(tail -20 "$LOG_FILE" | grep "epoch:" | tail -1 | grep -oP 'epoch: \K[0-9]+')
    if [ ! -z "$LATEST_FPS" ] && [ ! -z "$LATEST_EPOCH" ]; then
        echo "⚡ Performance:"
        echo "   FPS: $LATEST_FPS"
        echo "   Current Epoch: $LATEST_EPOCH / 50000"

        # Estimate time remaining
        EPOCHS_LEFT=$((50000 - LATEST_EPOCH))
        FRAMES_PER_EPOCH=131072
        FRAMES_LEFT=$((EPOCHS_LEFT * FRAMES_PER_EPOCH))
        SECONDS_LEFT=$((FRAMES_LEFT / LATEST_FPS))
        HOURS_LEFT=$((SECONDS_LEFT / 3600))
        echo "   ETA: ~$HOURS_LEFT hours remaining"
    fi
fi
echo ""
echo "=========================================================="
echo "Run: watch -n 60 ./monitor_lego_training.sh"
echo "Attach to session: screen -r lego_training"
echo "=========================================================="
