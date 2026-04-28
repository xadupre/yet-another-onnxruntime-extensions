rm dump_test* -rf
rm onnxruntime_profile*
rm eval*.onnx
rm plot*.onnx
rm plot*.png
rm temp*.onnx
rm test*.data
rm test*.onnx
rm dummy*.onnx
rm onnx*.md

if [ "$1" == "--all" ]; then
    rm yaourt/ortops/fused_kernel/cuda/*.so
    rm yaourt/ortops/sparse/cpu/*.so
    rm yaourt/ortops/sparse/cpu_v1/*.so
    rm -rf dist/
    rm -rf _build/
    rm -rf build/
    rm -rf .ruff_cache/
    rm -rf .pytest_cache/
    rm -rf docs/_build/
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
fi
