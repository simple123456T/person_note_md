---
title: "LLMops平台"
created: "2026-09-17 14:48:56"
updated: "2026-10-08 13:16:34"
folder: "工作相关"
---

198的big平台在请求积压巨大的情况下，删除请求队列表能缓解
```
# 要分开执行
TRUNCATE TABLE `gpu_metrics`;
TRUNCATE TABLE `queue_requests`;
```

先停掉后台服务，然后清空数据表，在重启服务。

