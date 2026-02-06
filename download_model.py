from modelscope import snapshot_download

# 下载模型
snapshot_download(
    'Qwen/Qwen3-TTS-12Hz-0.6B-Base',
    local_dir='./Qwen3-TTS-12Hz-0.6B-Base'
)
print("模型下载完成！")
