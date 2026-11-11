#!/usr/bin/env bash
# container_who.sh - 汇报当前运行容器及其 owner 和宿主机相关信息。
#
# 依赖: docker、awk、ps、stat；docker 不可免 sudo 时尝试 sudo -n docker。
# 用法:
#   ./container_who.sh            # 汇总 + 简表（owner/名字/镜像/设备/CPU/MEM/时长）
#   ./container_who.sh --table    # 同上简表，无汇总行，方便 grep/awk
#   ./container_who.sh --full     # 全字段明细表（含 command/ports/mounts/compose）
#   ./container_who.sh --json     # JSON Lines（全字段）
#   ./container_who.sh --help
set -o pipefail

MODE="all"
case "${1:-}" in
  ""|--all) MODE="all" ;;
  --table) MODE="table" ;;
  --full) MODE="full" ;;
  --json) MODE="json" ;;
  -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
  *) echo "unknown arg: $1" >&2; exit 2 ;;
esac

if [ -t 1 ]; then
  B=$'\e[1m'; D=$'\e[2m'; R=$'\e[0m'; C=$'\e[36m'
else
  B=""; D=""; R=""; C=""
fi

docker_cmd() {
  command -v docker >/dev/null || return 1
  docker "$@" 2>/dev/null && return 0
  sudo -n docker "$@" 2>/dev/null
}

command -v docker >/dev/null || { echo "docker not found" >&2; exit 1; }

HOST=$(hostname)
NOW=$(date '+%Y-%m-%d %H:%M:%S %z')
NOW_EPOCH=$(date +%s)
IDS=$(docker_cmd ps -q)

clean_field() {
  # 保证记录仍然是一行 TSV，避免命令或 label 破坏后续解析。
  tr '\t\r\n' '   ' <<<"$1" | sed 's/[[:space:]][[:space:]]*/ /g; s/^ //; s/ $//'
}

inspect_field() {
  docker_cmd inspect --format "$1" "$2"
}

stat_owner_walkup() {                     # $1=path -> 属主；本级 stat 权限不够就逐级上溯
  # 不少共享目录是 750（属主 rwx，其他人连 stat 子路径都 Permission denied），但目录条目
  # 本身（走它的父目录）通常还能 stat 到，所以按 dirname 逐级上溯找第一个能 stat 成功的层级。
  local p="$1" u
  while [ -n "$p" ] && [ "$p" != "/" ] && [ "$p" != "." ]; do
    u=$(stat -c %U "$p" 2>/dev/null) && { printf '%s' "$u"; return 0; }
    p=$(dirname "$p")
  done
  return 1
}
human_duration() {                        # $1=秒数 -> "3d4h" / "2h15m" / "15m" / "42s"
  local total="$1" d h m s
  d=$((total/86400)); total=$((total%86400))
  h=$((total/3600));  total=$((total%3600))
  m=$((total/60));    s=$((total%60))
  if   [ "$d" -gt 0 ]; then printf '%dd%dh' "$d" "$h"
  elif [ "$h" -gt 0 ]; then printf '%dh%dm' "$h" "$m"
  elif [ "$m" -gt 0 ]; then printf '%dm' "$m"
  else printf '%ds' "$s"
  fi
}

# docker stats 一次性批量取全部容器的 CPU/MEM，避免逐容器调用（N 次 vs 1 次）。
declare -A CPU_OF MEM_OF
while IFS=$'\t' read -r sid cpu memusage; do
  [ -z "$sid" ] && continue
  CPU_OF["$sid"]="$cpu"
  MEM_OF["$sid"]="${memusage%% / *}"        # "128.1GiB / 1.474TiB" -> 只留已用量
done < <(docker_cmd stats --no-stream --format '{{.ID}}	{{.CPUPerc}}	{{.MemUsage}}')

records=""
while read -r cid; do
  [ -z "$cid" ] && continue

  name=$(inspect_field '{{.Name}}' "$cid"); name=${name#/}
  image=$(inspect_field '{{.Config.Image}}' "$cid")
  status=$(inspect_field '{{.State.Status}}' "$cid")
  host_pid=$(inspect_field '{{.State.Pid}}' "$cid")
  created=$(inspect_field '{{.Created}}' "$cid")
  started=$(inspect_field '{{.State.StartedAt}}' "$cid")
  runtime_user=$(inspect_field '{{.Config.User}}' "$cid"); [ -z "$runtime_user" ] && runtime_user="root"
  path=$(inspect_field '{{.Path}}' "$cid")
  args=$(inspect_field '{{range .Args}}{{printf "%s " .}}{{end}}' "$cid")
  command_line=$(clean_field "$path $args")
  ports=$(inspect_field '{{range $p, $conf := .NetworkSettings.Ports}}{{if $conf}}{{range $conf}}{{.HostIp}}:{{.HostPort}}->{{$p}} {{end}}{{else}}{{$p}} {{end}}{{end}}' "$cid")
  mounts=$(inspect_field '{{range .Mounts}}{{.Source}}:{{.Destination}} ({{.Mode}}) {{end}}' "$cid")
  mount_sources=$(inspect_field '{{range .Mounts}}{{.Source}}{{"\n"}}{{end}}' "$cid")
  devices=$(inspect_field '{{range .HostConfig.Devices}}{{.PathOnHost}}{{"\n"}}{{end}}' "$cid" | \
    sed -n 's#.*/##p' | grep -vE '^(xpuctrl|davinci_manager|hisi_hdc|fuse|tun)$' | sort -V | paste -sd, -)
  [ -z "$devices" ] && devices="-"
  project=$(inspect_field '{{index .Config.Labels "com.docker.compose.project"}}' "$cid")
  service=$(inspect_field '{{index .Config.Labels "com.docker.compose.service"}}' "$cid")
  label_owner=$(inspect_field '{{index .Config.Labels "owner"}}' "$cid")
  [ "$label_owner" = "<no value>" ] && label_owner=""
  [ "$project" = "<no value>" ] && project=""
  [ "$service" = "<no value>" ] && service=""
  [ -z "$ports" ] && ports="-"
  [ -z "$mounts" ] && mounts="-"
  [ -z "$project" ] && project="-"
  [ -z "$service" ] && service="-"

  owner=""
  if [ -n "$label_owner" ]; then
    owner="$label_owner"
  else
    owner=$(printf '%s\n' "$mounts" | grep -oE '/home/[^/: ]+' | head -1 | cut -d/ -f3)
    if [ -z "$owner" ]; then
      while read -r source; do
        [ -z "$source" ] && continue
        mount_owner=$(stat_owner_walkup "$source") || continue
        if [ -n "$mount_owner" ] && [ "$mount_owner" != "root" ]; then
          owner="$mount_owner"
          break
        fi
      done <<<"$mount_sources"
    fi
  fi
  host_user="-"
  if [[ "$host_pid" =~ ^[0-9]+$ ]] && [ "$host_pid" -gt 0 ]; then
    host_user=$(stat -c %U "/proc/$host_pid" 2>/dev/null || echo "-")
  fi
  [ -z "$owner" ] && [ "$host_user" != "root" ] && [ "$host_user" != "-" ] && owner="$host_user"
  [ -z "$owner" ] && owner="unknown"

  cpu="${CPU_OF[${cid:0:12}]:--}"
  mem="${MEM_OF[${cid:0:12}]:--}"
  started_epoch=$(date -d "$started" +%s 2>/dev/null)
  if [ -n "$started_epoch" ]; then
    uptime_s=$((NOW_EPOCH - started_epoch)); [ "$uptime_s" -lt 0 ] && uptime_s=0
    uptime_h=$(human_duration "$uptime_s")
  else
    uptime_s=0; uptime_h="-"
  fi

  printf -v line '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' \
    "${cid:0:12}" "$(clean_field "$name")" "$(clean_field "$owner")" \
    "$(clean_field "$image")" "$(clean_field "$status")" "$host_pid" \
    "$(clean_field "$runtime_user")" "$(clean_field "$created")" \
    "$(clean_field "$started")" "$(clean_field "$command_line")" \
    "$(clean_field "$ports")" "$(clean_field "$mounts")" \
    "$(clean_field "$project")" "$(clean_field "$service")" \
    "$devices" "$cpu" "$mem" "$uptime_h" "$uptime_s"
  records+="$line"$'\n'
done <<<"$IDS"
records=$(printf '%s' "$records" | sed '/^$/d')

# 简表：owner / 名字 / 镜像 / 设备 / CPU / MEM / 时长 / ID —— 默认视图 + --table，
# 长字段做截断（IMAGE 保留尾部，版本 tag 通常在最后；其余保留头部），完整数据走 --json/--full。
print_table() {
  {
    printf 'OWNER\tNAME\tIMAGE\tDEVICES\tCPU\tMEM\tUPTIME\tID\n'
    printf '%s\n' "$records" | sort -t $'\t' -k3,3 -k2,2 | awk -F'\t' -v OFS='\t' '
      function trunc_head(s,n) { return length(s)>n ? substr(s,1,n-1) "…" : s }
      function trunc_tail(s,n) { return length(s)>n ? "…" substr(s,length(s)-n+2) : s }
      { print $3, trunc_head($2,28), trunc_tail($4,40), $15, $16, $17, $18, $1 }'
  } | if command -v column >/dev/null; then column -t -s $'\t'; else cat; fi
}

# 全字段明细表：含完整 command/ports/mounts/compose 信息。
print_full() {
  {
    printf 'CONTAINER_ID\tNAME\tOWNER\tIMAGE\tSTATUS\tHOST_PID\tRUNTIME_USER\tCREATED\tSTARTED\tCOMMAND\tPORTS\tMOUNTS\tCOMPOSE_PROJECT\tCOMPOSE_SERVICE\tDEVICES\tCPU\tMEM\tUPTIME\n'
    printf '%s\n' "$records" | awk -F'\t' -v OFS='\t' '{ NF=18; print }'
  } | if command -v column >/dev/null; then column -t -s $'\t'; else cat; fi
}

json_escape() {
  awk 'BEGIN { ORS="" } { gsub(/\\/, "\\\\"); gsub(/"/, "\\\""); printf "%s", $0 }'
}

print_json() {
  printf '%s\n' "$records" | while IFS=$'\t' read -r id name owner image status pid runtime_user created started command ports mounts project service devices cpu mem uptime_h uptime_s; do
    [ -z "$id" ] && continue
    printf '{"container_id":"%s","name":"%s","owner":"%s","image":"%s","status":"%s","host_pid":%s,"runtime_user":"%s","created":"%s","started":"%s","command":"%s","ports":"%s","mounts":"%s","compose_project":"%s","compose_service":"%s","devices":"%s","cpu":"%s","mem":"%s","uptime":"%s","uptime_seconds":%s}\n' \
      "$(printf '%s' "$id" | json_escape)" "$(printf '%s' "$name" | json_escape)" \
      "$(printf '%s' "$owner" | json_escape)" "$(printf '%s' "$image" | json_escape)" \
      "$(printf '%s' "$status" | json_escape)" "${pid:-0}" \
      "$(printf '%s' "$runtime_user" | json_escape)" "$(printf '%s' "$created" | json_escape)" \
      "$(printf '%s' "$started" | json_escape)" "$(printf '%s' "$command" | json_escape)" \
      "$(printf '%s' "$ports" | json_escape)" "$(printf '%s' "$mounts" | json_escape)" \
      "$(printf '%s' "$project" | json_escape)" "$(printf '%s' "$service" | json_escape)" \
      "$(printf '%s' "$devices" | json_escape)" "$(printf '%s' "$cpu" | json_escape)" \
      "$(printf '%s' "$mem" | json_escape)" "$(printf '%s' "$uptime_h" | json_escape)" \
      "${uptime_s:-0}"
  done
}

if [ "$MODE" = "json" ]; then
  print_json
  exit 0
fi

if [ "$MODE" = "table" ]; then
  print_table
  exit 0
fi

if [ "$MODE" = "full" ]; then
  print_full
  exit 0
fi

count=$(printf '%s\n' "$records" | grep -c . || true)
echo "${B}活动容器概览${R}  @ ${HOST}   ${NOW}"
if [ "$count" -eq 0 ]; then
  echo "  当前没有运行中的容器"
  exit 0
fi
printf '  容器数: %s\n\n' "$count"
print_table