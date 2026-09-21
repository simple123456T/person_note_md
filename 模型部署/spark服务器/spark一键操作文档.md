---
title: "spark一键操作文档"
created: "2026-09-21 10:11:29"
updated: "2026-09-21 10:11:42"
folder: "模型部署/spark服务器"
---

在 /home/admin/ 下建 modelctl ，内容即

```
#!/usr/bin/env bash
# =============================================================================
# modelctl — 本机三个模型服务的统一启停入口
#
#   modelctl start [服务]        启动（不给服务名=三个都起，已在运行的跳过）
#   modelctl stop  [服务]        停止（保留容器）
#   modelctl restart [服务]      停止后重新启动
#   modelctl status [服务]       查看状态与就绪情况
#   modelctl logs  [服务]        跟踪日志（不给服务名=各打印最后 30 行）
#
#   服务名：embedding | reranker | qwen（也接受 embed / rerank / llm 简写）
#   退出码：只要有服务没起来就是 1，方便 `modelctl start && ...` 串联判断
#
#   例：modelctl start            # 三个一起
#       modelctl restart qwen     # 只重启 SGLang
#
# 安装：sudo install -m 0755 modelctl /usr/local/bin/modelctl
# =============================================================================
set -uo pipefail

# 字段：名称|容器名|启动脚本|就绪探针|docker stop 超时(s)|就绪最长等待(s)
SERVICES=(
  "embedding|llama-qwen3-embedding-8b|/home/admin/middleware/llama-Qwen3-Embedding-8B-GGUF/start.sh|http://127.0.0.1:36226/health|15|300"
  "reranker|llama-qwen3-reranker-8b|/home/admin/middleware/llama-Qwen3-Reranker-8B-GGUF/start.sh|http://127.0.0.1:36227/health|15|300"
  "qwen|sglang-server-qwen|/home/admin/middleware/sglang-RadixArk-Qwen3.8-27B-NVFP4/sglang_start.sh|http://127.0.0.1:9210/v1/models|60|1800"
)

LOG_DIR="${LOG_DIR:-$HOME/model-logs}"
TARGETS=()
declare -A SCRIPT_PID=()   # 启动脚本的 PID，用来判断它是否已经自己退出
FAILED=0

info() { printf '\033[36m[INFO]\033[0m  %s\n' "$*"; }
err()  { printf '\033[31m[ERR]\033[0m   %s\n' "$*" >&2; }
die()  { err "$*"; exit 1; }

running()      { docker ps --format '{{.Names}}' | grep -qx "$1"; }
container_id() { docker container inspect -f '{{.Id}}' "$1" 2>/dev/null || true; }
# llama.cpp 的 /health 和 SGLang 的 /v1/models 都是就绪返回 200
ready()        { [[ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 "$1" 2>/dev/null || true)" == "200" ]]; }

# 服务名简写归一
normalize() {
  case "$1" in
    embed|emb|embedding)      echo 'embedding' ;;
    rerank|rer|reranker)      echo 'reranker' ;;
    llm|sglang|qwen)          echo 'qwen' ;;
    *)                        echo "$1" ;;
  esac
}

# 按可选的服务名筛出 TARGETS（不传服务名 = 全部）
select_services() {
  local want='' entry name
  [[ -n "${1:-}" ]] && want="$(normalize "$1")"
  TARGETS=()
  for entry in "${SERVICES[@]}"; do
    name="${entry%%|*}"
    [[ -z "$want" || "$name" == "$want" ]] && TARGETS+=("$entry")
  done
  (( ${#TARGETS[@]} > 0 )) \
    || die "未知服务: ${1:-}（可选：embedding | reranker | qwen）"
}

for_each_target() {
  local fn="$1" entry
  for entry in "${TARGETS[@]}"; do "$fn" "$entry"; done
}

# --------------------------------- start -------------------------------------
start_one() {
  local name container script probe stop_to ready_to
  IFS='|' read -r name container script probe stop_to ready_to <<< "$1"
  [[ -f "$script" ]] || die "找不到启动脚本: $script"
  if running "$container"; then
    info "$name 已在运行，跳过"
    return 0
  fi
  mkdir -p "$LOG_DIR"
  # 关键：先删掉残留的"已停止"容器。否则下面的就绪等待会把它误判成"新容器已退出"，
  # 还会把它的历史日志当成本次失败原因打印出来（曾真的踩到过）。
  docker rm -f "$container" >/dev/null 2>&1 || true
  info "$name 拉起中 → 日志 $LOG_DIR/$name.log"
  # 后台拉起：qwen 首次要编译 kernel（脚本内部最长等 1800s），不能阻塞另外两个。
  # nohup + 重定向 + </dev/null 保证 ssh 断开不会因 SIGHUP 中断，且 $! 就是脚本 PID。
  # 多传的 "start" 参数会被两个 llama 的 start.sh 忽略。
  nohup bash "$script" start >"$LOG_DIR/$name.log" 2>&1 </dev/null &
  SCRIPT_PID["$name"]=$!
}

wait_targets() {
  local entry name container script probe stop_to ready_to waited stopped_secs pid failed
  for entry in "${TARGETS[@]}"; do
    IFS='|' read -r name container script probe stop_to ready_to <<< "$entry"
    printf '等待 %s 就绪' "$name"
    waited=0; stopped_secs=0; failed=0

    while ! ready "$probe"; do
      if [[ -z "$(container_id "$container")" ]]; then
        # 新容器还没建出来。如果启动脚本已经自己退出，说明这次真的没起来
        # （模型目录不存在、镜像缺失之类），不必干等到超时。
        pid="${SCRIPT_PID[$name]:-}"
        if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
          printf '\n'; err "$name 启动脚本已退出且没有创建出容器，脚本日志尾部："
          tail -n 40 "$LOG_DIR/$name.log" 2>/dev/null || true
          failed=1; break
        fi
      elif ! running "$container"; then
        # 容器存在却不在运行：连续 15s 才判失败，容忍 --restart unless-stopped
        # 自动重启期间出现的短暂 Restarting 状态。
        stopped_secs=$(( stopped_secs + 5 ))
        if (( stopped_secs >= 15 )); then
          printf '\n'; err "$name 容器已停止，最后 40 行日志："
          docker logs --tail 40 "$container" 2>&1 || true
          failed=1; break
        fi
      else
        stopped_secs=0
      fi

      if (( waited >= ready_to )); then
        printf '\n'; err "$name 等待超时（${ready_to}s），脚本日志尾部："
        tail -n 20 "$LOG_DIR/$name.log" 2>/dev/null || true
        failed=1; break
      fi
      printf '.'; sleep 5; waited=$(( waited + 5 ))
    done

    if (( failed )); then FAILED=1; else printf ' 就绪（%ss）\n' "$waited"; fi
  done
}

# ---------------------------------- stop -------------------------------------
stop_one() {
  local name container script probe stop_to ready_to
  IFS='|' read -r name container script probe stop_to ready_to <<< "$1"
  if ! docker container inspect "$container" >/dev/null 2>&1; then
    info "$name 容器不存在，跳过"
    return 0
  fi
  if ! running "$container"; then
    info "$name 已处于停止状态，跳过"
    return 0
  fi
  info "$name 停止（优雅退出超时 ${stop_to}s，容器保留）"
  docker stop -t "$stop_to" "$container" >/dev/null 2>&1 || true
}

# --------------------------------- status ------------------------------------
do_status() {
  select_services "${1:-}"
  local entry name container script probe stop_to ready_to state health
  printf '%-10s %-28s %-10s %s\n' '服务' '容器' '状态' '就绪'
  for entry in "${TARGETS[@]}"; do
    IFS='|' read -r name container script probe stop_to ready_to <<< "$entry"
    state="$(docker container inspect -f '{{.State.Status}}' "$container" 2>/dev/null || true)"
    case "$state" in
      running) state='运行中' ;;
      exited)  state='已停止' ;;
      created) state='未启动' ;;
      '')      state='不存在' ;;
    esac
    health='未就绪'; ready "$probe" && health='就绪'
    printf '%-10s %-28s %-10s %s\n' "$name" "$container" "$state" "$health"
  done
}

# ---------------------------------- logs -------------------------------------
do_logs() {
  select_services "${1:-}"
  local entry name container script probe stop_to ready_to
  if [[ -n "${1:-}" ]]; then
    IFS='|' read -r name container script probe stop_to ready_to <<< "${TARGETS[0]}"
    docker container inspect "$container" >/dev/null 2>&1 || die "$container 不存在"
    exec docker logs -f --tail 200 "$container"
  fi
  for entry in "${TARGETS[@]}"; do
    IFS='|' read -r name container script probe stop_to ready_to <<< "$entry"
    printf '\n===== %s（%s）最后 30 行 =====\n' "$name" "$container"
    docker logs --tail 30 "$container" 2>&1 || true
  done
}

# ---------------------------------- 入口 -------------------------------------
CMD="${1:-status}"; shift || true
case "$CMD" in
  start|up)
    select_services "${1:-}"; for_each_target start_one; wait_targets; do_status "${1:-}" ;;
  stop|down)
    select_services "${1:-}"; for_each_target stop_one ;;
  restart)
    select_services "${1:-}"; for_each_target stop_one; for_each_target start_one
    wait_targets; do_status "${1:-}" ;;
  status|ps)
    do_status "${1:-}" ;;
  logs)
    do_logs "${1:-}" ;;
  -h|--help)
    awk 'NR==1 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "$0" ;;
  *)
    die "用法：modelctl {start|stop|restart|status|logs} [embedding|reranker|qwen]" ;;
esac
exit "$FAILED"
```

给 modelctl 配置权限:
chmod 0755 /home/admin/modelctl

装到 PATH (软链接的方式)
sudo ln -sfn /home/admin/modelctl /usr/local/bin/modelctl

modelctl start 实测全量启动；
再实测 modelctl restart qwen 验证单服务粒度。

验证在任何目录都能用
cd /tmp && modelctl status

实测启停
modelctl start               # 三个一起起
modelctl start qwen          # 单独启动qwen
modelctl start embedding     # 单独启动embedding
modelctl start reranker      # 单独启动reranker

modelctl status              # 查看三个的状态
modelctl status qwen         # 单独查看qwen
modelctl status embedding    # 单独查看embedding
modelctl status reranker     # 单独查看reranker

modelctl stop                # 三个一起停
modelctl stop qwen           # 单独停止qwen
modelctl stop embedding      # 单独停止embedding
modelctl stop reranker       # 单独停止reranker

modelctl restart qwen        # 只重启qwen
modelctl restart embedding   # 只重启embedding
modelctl restart reranker    # 只重启reranker

modelctl logs qwen        # 只打印qwen最后 30 行日志
modelctl logs embedding   # 只打印embedding最后 30 行日志
modelctl logs reranker    # 只打印reranker最后 30 行日志