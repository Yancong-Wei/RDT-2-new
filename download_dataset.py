# 使用 huggingface_hub 直接下载
from huggingface_hub import snapshot_download

# 下载整个数据集
snapshot_download(
    repo_id="robotics-diffusion-transformer/BimanualUR5eExample",
    repo_type="dataset",
    local_dir="./pretrained_models/example_data"
)