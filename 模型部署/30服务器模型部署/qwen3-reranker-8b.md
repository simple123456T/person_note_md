---
title: "qwen3-reranker-8b"
created: "2026-09-07 10:44:28"
updated: "2026-09-10 14:42:14"
folder: "模型部署/30服务器模型部署"
---

# qwen3-embedding-8b 模型部署

`
启动脚本地址：/home/admin/middleware/vllm-Qwen3-Reranker-8B/start.sh
`
`
模型地址：/home/admin/models/Qwen3-Reranker-8B
`


```
docker rm -f vllm-Qwen3-Reranker-8B 2>/dev/null || true

docker run -d \
  --name vllm-Qwen3-Reranker-8B \
  --gpus '"device=2"' \
  --ipc=host \
  -p 32142:8000 \
  -v /home/admin/models/Qwen3-Reranker-8B:/models/Qwen3-Reranker-8B:ro \
  vllm/vllm-openai:v0.24.0 \
  /models/Qwen3-Reranker-8B \
  --runner pooling \
  --chat-template /models/Qwen3-Reranker-8B/chat_template.jinja \
  --max-model-len 8192 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.95 \
  --dtype float16 \
  --attention-backend TRITON_ATTN \
  --served-model-name ph-Qwen-Reranker-8B \
  --hf-overrides '{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}'
```


# 测试脚本

```
curl --location --request POST 'http://192.168.2.30:32142/v1/rerank' \
--header 'Content-Type: application/json' \
--data-raw '{
  "model": "ph-Qwen-Reranker-8B",
  "query": "美洲杯夺冠次数最多的国家",
  "documents": [
    "乌拉圭和阿根廷以15次冠军，并列美洲杯历史夺冠次数第一。",
    "首届美洲杯于1916年举办，乌拉圭拿下首届冠军。",
    "近年美洲杯赛事节奏调整，恢复为四年举办一次。",
    "阿根廷曾夺得2022年卡塔尔世界杯冠军。",
    "巴西是南美洲传统足球强国，大赛成绩稳定。",
    "欧洲杯每四年举办一次，是欧洲最高级别足球赛事。",
    "西班牙在2024年欧洲杯展现了全新战术体系。",
    "世界杯是全球影响力最大的足球赛事。",
    "姆巴佩是法国国家队核心前锋。",
    "梅西曾多次代表阿根廷参加国际大赛。"
  ],
  "top_n": 5,
  "return_documents": true
}'

```