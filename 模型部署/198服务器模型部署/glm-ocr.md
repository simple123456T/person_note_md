---
title: "glm-ocr"
created: "2026-09-07 09:38:56"
updated: "2026-09-07 09:53:01"
folder: "模型部署/198服务器模型部署"
---

# 198服务器 GLM-OCR-0.9B 部署 脚本

0.9B 参数量模型占用 6G 显存

```
#!/bin/bash

docker rm -f vllm-glm-ocr 2>/dev/null || true

docker run -d \
  --name vllm-glm-ocr \
  --restart unless-stopped \
  --gpus '"device=7"' \
  -p 35663:8080 \
  -v /home/admin/models/GLM-OCR:/models/GLM-OCR \
  --entrypoint /bin/bash \
  vllm/vllm-openai:nightly \
  -c '
    exec vllm serve /models/GLM-OCR \
      --port 8080 \
      --served-model-name glm-ocr \
      --speculative-config '"'"'{"method": "mtp", "num_speculative_tokens": 3}'"'"' \
      --gpu-memory-utilization 0.25 \
      --max-model-len 4086 \
      --dtype half \
      --trust-remote-code
```

