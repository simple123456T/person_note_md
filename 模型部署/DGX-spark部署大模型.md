---
title: "DGX-spark部署大模型"
created: "2026-08-10 10:32:47"
updated: "2026-08-13 23:44:21"
folder: "模型部署"
---

DGX-Spark部署 `Qwen3.6-35B-A3B-NVFP4`

镜像文件下载
- vllm-openai:v0.26.0-aarch64 
- `docker pull vllm/vllm-openai:v0.26.0-aarch64`

模型下载
- `hf download nvidia/Qwen3.6-35B-A3B-NVFP4 --local-dir ./`

执行脚本
``` start.sh
#!/bin/bash
docker rm -f Qwen3.6-35B-A3B-NVFP4_vllm 2>/dev/null

docker run -d \
  --name Qwen3.6-35B-A3B-NVFP4_vllm \
  --runtime=nvidia \
  --gpus all \
  --ipc=host \
  --network=host \
  --privileged \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --restart=unless-stopped \
  -e VLLM_LOGGING_LEVEL=INFO \
  -e PYTHONUNBUFFERED=1 \
  -v /home/admin/models/Qwen3.6-35B-A3B-NVFP4:/model \
  vllm/vllm-openai:v0.26.0-aarch64 \
  --model /model \
  --served-model-name Qwen3.6-35B-A3B-NVFP4 \
  --host 0.0.0.0 \
  --port 35667 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --kv-cache-dtype fp8  \
  --attention-backend flashinfer  \
  --moe-backend marlin \
  --kv-cache-memory 75130096538 \
  --max-model-len 262144 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192  \
  --enable-chunked-prefill \
  --async-scheduling \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'  \
  --load-format fastsafetensors \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice

```
 特别说明：
```
--gpu-memory-utilization 0.8  \
这个参数在DGX spark 上没有生效，实际设置这个参数后效果是会将内存占满，原因是在这个vllm-openai:v0.26.0-aarch64镜像下在 GB10 统一内存平台上，vLLM 的 CUDA Graph 显存估算测出了一个负值（-20.04 GiB）——这是该平台上内存统计的已知怪癖，导致 vLLM 误以为还有更多余量
```

官方推荐部署参数
``` 
vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4  \
   --host 0.0.0.0 \
   --port 8000 \
   --tensor-parallel-size 1 \
   --trust-remote-code \
   --kv-cache-dtype fp8  \
   --attention-backend flashinfer  \
   --moe-backend marlin \
   --gpu-memory-utilization 0.4  \
   --max-model-len 262144 \
   --max-num-seqs 4 \
   --max-num-batched-tokens 8192  \
   --enable-chunked-prefill \
   --async-scheduling \
   --enable-prefix-caching \
   --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'  \
   --load-format fastsafetensors \
   --reasoning-parser qwen3 \
   --tool-call-parser qwen3_xml \
   --enable-auto-tool-choice
```


