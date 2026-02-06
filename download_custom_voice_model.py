from modelscope import snapshot_download

# 下载模型
snapshot_download(
    'Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice',
    local_dir='./Qwen3-TTS-12Hz-0.6B-CustomVoice模型下载'
)
print("模型下载完成！")
