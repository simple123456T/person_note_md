---
title: "minicpm-v4.6"
created: "2026-09-07 09:50:33"
updated: "2026-09-07 17:12:43"
folder: "模型部署/198服务器模型部署"
---

# 198服务器 MiniCPM-V-4.6 部署 脚本

1.3B 参数量模型占用 7G 显存

```
docker rm -f vllm-minicpm-v46 2>/dev/null || true

docker run -d \
  --gpus '"device=7"' \
  -p 36523:8000 \
  -v /home/admin/models/MiniCPM-V-4.6:/models \
  --name vllm-minicpm-v46 \
  vllm/vllm-openai:v0.24.0 \
  --model /models \
  --served-model-name MiniCPM-V-4.6 \
  --trust-remote-code \
  --max-model-len 2048 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.4 \
  --max-num-seqs 32 \
  --limit-mm-per-prompt '{"image": 1}'
```

# 测试脚本

```
curl --location --request POST 'http://10.118.21.198:36523/v1/chat/completions' \
--header 'Content-Type: application/json' \
--data-raw '{
  "model": "MiniCPM-V-4.6",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "请描述这张图片的内容。"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "https://img-blog.csdnimg.cn/fcc22710385e4edabccf2451d5f64a99.jpeg"
          }
        }
      ]
    }
  ],
  "max_tokens": 1000,
  "temperature": 0.7
}'
```
