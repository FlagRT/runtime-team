#!/usr/bin/env bash
# container_who.sh - 汇报当前运行容器及其 owner 和宿主机相关信息。
#
# 依赖: docker、awk、ps、stat；docker 不可免 sudo 时尝试 sudo -n docker。
# 用法:
#   ./container_who.sh            # 汇总 + 明细
#   ./container_who.sh --table    # TSV 明细表
#   ./container_who.sh --json     # JSON Lines
#   ./container_who.sh --help
set -o pipefail

MODE="all"
case "${1:-}" in
  ""|--all) MODE="all" ;;
  --table) MODE="table" ;;
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
IDS=$(docker_cmd ps -q)

clean_field() {
  # 保证记录仍然是一行 TSV，避免命令或 label 破坏后续解析。
  tr '\t\r\n' '   ' <<<"$1" | sed 's/[[:space:]][[:space:]]*/ /g; s/^ //; s/ $//'
}

inspect_field() {
  docker_cmd inspect --format "$1" "$2"
}

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
        mount_owner=$(stat -c %U "$source" 2>/dev/null || true)
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

  printf -v line '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' \
    "${cid:0:12}" "$(clean_field "$name")" "$(clean_field "$owner")" \
    "$(clean_field "$image")" "$(clean_field "$status")" "$host_pid" \
    "$(clean_field "$runtime_user")" "$(clean_field "$created")" \
    "$(clean_field "$started")" "$(clean_field "$command_line")" \
    "$(clean_field "$ports")" "$(clean_field "$mounts")" \
    "$(clean_field "$project")" "$(clean_field "$service")"
  records+="$line"$'\n'
done <<<"$IDS"
records=$(printf '%s' "$records" | sed '/^$/d')

print_table() {
  {
    printf 'CONTAINER_ID\tNAME\tOWNER\tIMAGE\tSTATUS\tHOST_PID\tRUNTIME_USER\tCREATED\tSTARTED\tCOMMAND\tPORTS\tMOUNTS\tCOMPOSE_PROJECT\tCOMPOSE_SERVICE\n'
    printf '%s\n' "$records"
  } | if command -v column >/dev/null; then column -t -s $'\t'; else cat; fi
}

json_escape() {
  awk 'BEGIN { ORS="" } { gsub(/\\/, "\\\\"); gsub(/"/, "\\\""); printf "%s", $0 }'
}

print_json() {
  printf '%s\n' "$records" | while IFS=$'\t' read -r id name owner image status pid runtime_user created started command ports mounts project service; do
    [ -z "$id" ] && continue
    printf '{"container_id":"%s","name":"%s","owner":"%s","image":"%s","status":"%s","host_pid":%s,"runtime_user":"%s","created":"%s","started":"%s","command":"%s","ports":"%s","mounts":"%s","compose_project":"%s","compose_service":"%s"}\n' \
      "$(printf '%s' "$id" | json_escape)" "$(printf '%s' "$name" | json_escape)" \
      "$(printf '%s' "$owner" | json_escape)" "$(printf '%s' "$image" | json_escape)" \
      "$(printf '%s' "$status" | json_escape)" "${pid:-0}" \
      "$(printf '%s' "$runtime_user" | json_escape)" "$(printf '%s' "$created" | json_escape)" \
      "$(printf '%s' "$started" | json_escape)" "$(printf '%s' "$command" | json_escape)" \
      "$(printf '%s' "$ports" | json_escape)" "$(printf '%s' "$mounts" | json_escape)" \
      "$(printf '%s' "$project" | json_escape)" "$(printf '%s' "$service" | json_escape)"
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

count=$(printf '%s\n' "$records" | grep -c . || true)
echo "${B}活动容器概览${R}  @ ${HOST}   ${NOW}"
if [ "$count" -eq 0 ]; then
  echo "  当前没有运行中的容器"
  exit 0
fi
printf '  容器数: %s\n\n' "$count"
print_table