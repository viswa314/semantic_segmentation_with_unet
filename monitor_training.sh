#!/bin/bash
# Training Monitor Script

echo "===================================================================================="
echo "U-Net Training Monitor"
echo "===================================================================================="
echo ""

# Check if training is running
if pgrep -f "train_cityscapes.py" > /dev/null; then
    echo "✓ Training is RUNNING"
    echo ""
    
    # Show process info
    echo "Process Info:"
    ps aux | grep "train_cityscapes.py" | grep -v grep | head -1
    echo ""
    
    # Show latest progress
    echo "Latest Progress:"
    tail -5 /home/viswa/unet/training.log
    echo ""
    
    # Check for checkpoints
    if [ -d "/home/viswa/unet/checkpoints" ]; then
        echo "Saved Checkpoints:"
        ls -lh /home/viswa/unet/checkpoints/ 2>/dev/null | tail -5
    fi
else
    echo "✗ Training is NOT running"
    echo ""
    echo "Last 20 lines of log:"
    tail -20 /home/viswa/unet/training.log
fi

echo ""
echo "===================================================================================="
echo "To monitor in real-time: tail -f /home/viswa/unet/training.log"
echo "To stop training: pkill -f train_cityscapes.py"
echo "===================================================================================="
