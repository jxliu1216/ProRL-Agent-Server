#!/usr/bin/env bash
# Cleanup all training, rollout, and related processes.
set -euo pipefail

echo "=== Cleaning up training processes ==="

# 1. Stop Ray (manages SGLang engines + Megatron training workers)
echo "Stopping Ray..."
ray stop --force 2>/dev/null || true
sleep 1

# 2. Kill all apptainer instances (Polar runtime containers)
echo "Stopping Apptainer instances..."
apptainer instance list 2>/dev/null | awk 'NR>1 {print $1}' | xargs -r -I{} apptainer instance stop {} 2>/dev/null || true
sleep 1

# 3. Detach stale loop devices left by apptainer
echo "Detaching stale loop devices..."
losetup -a 2>/dev/null | grep -oP '/dev/loop\d+' | while read dev; do
    if ! mount 2>/dev/null | grep -q "$dev"; then
        losetup -d "$dev" 2>/dev/null || true
    fi
done
sleep 1

# 4. Kill Polar services
echo "Stopping Polar services..."
pkill -f "polar serve" 2>/dev/null || true
sleep 1

# 5. Kill torchrun (weight conversion / Megatron)
echo "Stopping torchrun..."
pkill -f "torchrun" 2>/dev/null || true
sleep 1

# 6. Kill SGLang engines
echo "Stopping SGLang..."
pkill -f "sglang" 2>/dev/null || true
sleep 1

# 7. Kill any remaining uvicorn servers (Polar rollouts)
echo "Stopping uvicorn..."
pkill -f "uvicorn" 2>/dev/null || true
sleep 1

# 8. Kill any remaining Python training/rollout processes
echo "Stopping remaining Python processes..."
pkill -f "train_async\|rollout\|megatron" 2>/dev/null || true
sleep 1

# 9. Verify cleanup
echo ""
echo "=== Cleanup verification ==="
echo -n "Ray processes:     "; ps aux | grep -c "[r]ay" || echo 0
echo -n "Apptainer instances: "; apptainer instance list 2>/dev/null | tail -n +2 | wc -l || echo 0
echo -n "Polar processes:   "; ps aux | grep -c "[p]olar" || echo 0
echo -n "SGLang processes:  "; ps aux | grep -c "[s]glang" || echo 0
echo -n "Torchrun processes:"; ps aux | grep -c "[t]orchrun" || echo 0
echo -n "Python training:   "; ps aux | grep -cE "[t]rain_async|[m]egatron" || echo 0
echo -n "Loop devices:      "; losetup -a 2>/dev/null | wc -l || echo 0

echo ""
echo "Done."
