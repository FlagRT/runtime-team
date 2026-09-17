#!/usr/bin/env bash
# npu-who.sh — 汇报当前 Ascend NPU 资源占用，并把每个占用进程回溯到
#              宿主机用户 / 启动脚本 / 容器 / 具体任务。
#
# 依赖: npu-smi, awk, ps；可选: docker（能免 sudo 最好）、getent。
# 用法:
#   ./npu-who.sh            # 设备概览 + 归属树 + 明细表
#   ./npu-who.sh --tree     # 只看归属树
#   ./npu-who.sh --table    # 只看明细表（tab 分隔，方便 grep/awk）
#   ./npu-who.sh --json     # 明细以 JSON 行输出
#
# 原理（已在生产机验证）:
#   npu-smi 进程表给出宿主机 PID
#     → /proc/<pid>/cgroup           得到 docker 容器 ID
#     → docker inspect .Mounts       绑定挂载里的 /home/<user>/... 即发起人
#     → ps 找该用户仍在运行的 run.py / docker run 等启动进程
#     → 沿 ppid 上溯到 sshd / cron / systemd 会话入口
set -o pipefail

MODE="all"
case "${1:-}" in
  --tree)  MODE="tree" ;;
  --table) MODE="table" ;;
  --json)  MODE="json" ;;
  ""|--all) MODE="all" ;;
  -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
  *) echo "unknown arg: $1" >&2; exit 2 ;;
esac

if [ -t 1 ]; then B=$'\e[1m'; D=$'\e[2m'; R=$'\e[0m'; Y=$'\e[33m'; C=$'\e[36m'; else B=""; D=""; R=""; Y=""; C=""; fi

command -v npu-smi >/dev/null || { echo "npu-smi not found" >&2; exit 1; }

docker_cmd() {
  command -v docker >/dev/null || return 1
  docker "$@" 2>/dev/null && return 0
  sudo -n docker "$@" 2>/dev/null
}

HOST=$(hostname)
NOW=$(date '+%Y-%m-%d %H:%M:%S %z')

# ---------------------------------------------------------------------------
# 1) 解析 npu-smi info  ->  DEV / PROC 行
# ---------------------------------------------------------------------------
SMI=$(npu-smi info 2>/dev/null)
PARSED=$(printf '%s\n' "$SMI" | awk '
  /^\| +[0-9]+ +Ascend/ {                       # 设备行1: NPU 号
     s=$0; sub(/^\| +/,"",s); split(s,a," "); cur=a[1]; next
  }
  /^\| +[0-9]+ +[0-9]+ +\| +[0-9A-Fa-f]+:[0-9A-Fa-f]/ {   # 设备行2: 芯片/BusId/占用
     n=split($0,f,"|")
     split(f[2],g," "); chip=g[1]; phy=g[2]
     split(f[3],h," "); bus=h[1]
     split(f[4],k," "); ai=k[1]
     hbm_cur=k[length(k)-2]; hbm_tot=k[length(k)]
     print "DEV\t" cur "\t" chip "\t" phy "\t" bus "\t" ai "\t" hbm_cur "\t" hbm_tot
     next
  }
  /^\| +[0-9]+ +[0-9]+ +\| +[0-9]+ +\| +/ {     # 进程行
     n=split($0,f,"|")
     split(f[2],g," "); pnpu=g[1]; pchip=g[2]
     split(f[3],h," "); pid=h[1]
     split(f[4],nm," "); pname=nm[1]
     split(f[5],mm," "); pmem=mm[1]
     print "PROC\t" pnpu "\t" pchip "\t" pid "\t" pname "\t" pmem
  }
')

DEVROWS=$(printf '%s\n' "$PARSED" | awk -F'\t' '$1=="DEV"')
PROCROWS=$(printf '%s\n' "$PARSED" | awk -F'\t' '$1=="PROC"')

# ---------------------------------------------------------------------------
# 2) 逐进程回溯归属
#    输出记录(TSV): npu chip pid pname mem hostuser owner cid12 cname \
#                   launch_pid launch_user launch_cmd entry task
# ---------------------------------------------------------------------------
resolve_launcher() {                      # $1=owner  $2=uid  -> "pid<TAB>user<TAB>cmd"
  # 注意: `ps -o user=` 会把 >8 字符用户名截断成 kzhang5+，必须按 uid 匹配
  ps -ww -eo pid=,uid=,args= 2>/dev/null | awk -v o="$1" -v u="$2" '
    $2==u && $0 ~ /(^| )([^ ]*run\.py|torchrun|mpirun)( |$)|flagperf|docker +run|compose +up/ {
      pid=$1; $1=$2=""; sub(/^ +/,"")
      print pid "\t" o "\t" $0; exit
    }'
}
entry_ancestor() {                        # $1=pid -> "会话入口  [via: a ← b ← c]"
  local p="$1" hop pp pa comm chain="" entry=""
  for hop in $(seq 1 16); do
    [ -z "$p" ] && break
    [ "$p" -le 1 ] 2>/dev/null && break
    IFS=$'\t' read -r pp pa < <(ps -o ppid=,args= -p "$p" 2>/dev/null | awk '{pp=$1;$1="";sub(/^ +/,"");print pp"\t"$0}')
    [ -z "$pp" ] && break
    comm=$(basename "${pa%% *}"); [ -n "$chain" ] && chain="$chain ← "; chain="$chain$comm"
    case "$pa" in
      *sshd:*|*CRON*|*" cron "*|*"systemd --user"*|*"login --"*|*" tmux"*|*screen*)
        entry="${pa:0:80}"; break ;;
    esac
    p="$pp"
  done
  [ -z "$entry" ] && entry="(未定位到登录会话)"
  echo "${entry}   [via: ${chain}]"
}

RECORDS=""
while IFS=$'\t' read -r _ npu chip pid pname mem; do
  [ -z "$pid" ] && continue
  hostuser=$(stat -c %U "/proc/$pid" 2>/dev/null || echo '?')
  task=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | sed 's/ *$//')
  [ -z "$task" ] && task=$(ps -o args= -p "$pid" 2>/dev/null)
  [ -z "$task" ] && task="(process gone)"

  cg=$(cat "/proc/$pid/cgroup" 2>/dev/null)
  cid=$(printf '%s\n' "$cg" | grep -oE '[0-9a-f]{64}' | head -1)
  [ -z "$cid" ] && cid=$(printf '%s\n' "$cg" | grep -oE '(docker|kubepods|containerd)[-/][^ ]+' | head -1)

  cname="-"; owner=""; mounts=""; lbl=""
  if [ -n "$cid" ] && [ ${#cid} -eq 64 ]; then
    cname=$(docker_cmd inspect --format '{{.Name}}' "$cid"); cname=${cname#/}
    [ -z "$cname" ] && cname="${cid:0:12}"
    mounts=$(docker_cmd inspect --format '{{range .Mounts}}{{.Source}}{{"\n"}}{{end}}' "$cid")
    lbl=$(docker_cmd inspect --format '{{index .Config.Labels "owner"}}' "$cid")
  elif [ -n "$cid" ]; then
    cname="${cid:0:20}"
  fi

  if [ -n "$mounts" ]; then
    owner=$(printf '%s\n' "$mounts" | grep -oE '/home/[^/]+' | head -1 | cut -d/ -f3)
    if [ -z "$owner" ]; then
      while read -r m; do
        [ -z "$m" ] && continue
        u=$(stat -c %U "$m" 2>/dev/null) || continue
        [ "$u" != "root" ] && { owner="$u"; break; }
      done <<<"$mounts"
    fi
  fi
  [ -z "$owner" ] && [ -n "$lbl" ] && [ "$lbl" != "<no value>" ] && owner="$lbl"
  { [ -z "$owner" ] || [ "$owner" = "root" ]; } && [ "$hostuser" != "root" ] && owner="$hostuser"
  [ -z "$owner" ] && owner="unknown"

  lpid="-"; luser="-"; lcmd="(launcher not found — 可能已退出)"; entry="-"
  if [ "$owner" != "unknown" ] && [ "$owner" != "root" ]; then
    ouid=$(id -u "$owner" 2>/dev/null)
    IFS=$'\t' read -r lpid luser lcmd < <(resolve_launcher "$owner" "${ouid:--1}")
    [ -z "$lpid" ] && { lpid="-"; luser="-"; lcmd="(launcher not found — 可能已退出)"; }
    [ "$lpid" != "-" ] && entry=$(entry_ancestor "$lpid")
  fi

  printf -v line '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' \
    "$npu" "$chip" "$pid" "$pname" "$mem" "$hostuser" "$owner" "${cid:0:12}" "$cname" \
    "$lpid" "$luser" "$lcmd" "$entry" "$task"
  RECORDS+="$line"$'\n'
done <<<"$PROCROWS"

RECORDS=$(printf '%s' "$RECORDS" | sed '/^$/d')

# ---------------------------------------------------------------------------
# 3) 输出
# ---------------------------------------------------------------------------
NPROC=$(printf '%s\n' "$RECORDS" | grep -c . || true)

print_devsummary() {
  echo "${B}NPU 设备概览${R}  @ ${HOST}   ${NOW}"
  if [ -z "$DEVROWS" ]; then echo "  (无法解析 npu-smi 设备表)"; return; fi
  printf '  %-4s %-5s %-6s %-15s %-7s %-16s %s\n' 'NPU' 'CHIP' 'PHY-ID' 'BUS-ID' 'AICore' 'HBM(MB)' '进程'
  local busy=0 idle=0
  while IFS=$'\t' read -r _ npu chip phy bus ai hc ht; do
    n=$(printf '%s\n' "$RECORDS" | awk -F'\t' -v a="$npu" -v b="$chip" '$1==a&&$2==b' | wc -l | tr -d ' ')
    tag=""; [ "$n" -gt 0 ] && { tag="${Y}${n} proc${R}"; busy=$((busy+1)); } || { tag="${D}idle${R}"; idle=$((idle+1)); }
    printf '  %-4s %-5s %-6s %-15s %-6s%% %-16s %b\n' "$npu" "$chip" "$phy" "$bus" "$ai" "$hc/$ht" "$tag"
  done <<<"$DEVROWS"
  echo "  ${D}合计: ${busy} 芯片有进程 / $((busy+idle)) 芯片${R}"
}

print_tree() {
  echo "${B}资源归属树${R}   (占用进程 ${NPROC} 个)"
  [ -z "$RECORDS" ] && { echo "  ✅ 当前没有进程占用 NPU"; return; }
  local owners
  owners=$(printf '%s\n' "$RECORDS" | awk -F'\t' '{print $7}' | sort -u)
  while read -r o; do
    [ -z "$o" ] && continue
    uid=$(id -u "$o" 2>/dev/null || echo '?')
    sub=$(printf '%s\n' "$RECORDS" | awk -F'\t' -v o="$o" '$7==o')
    cnt=$(printf '%s\n' "$sub" | wc -l | tr -d ' ')
    mb=$(printf '%s\n' "$sub" | awk -F'\t' '{s+=$5} END{print s+0}')
    npus=$(printf '%s\n' "$sub" | awk -F'\t' '{print $1}' | sort -un | paste -sd, -)
    echo
    echo "${B}● owner=${o}${R} (uid ${uid})  —  ${cnt} proc, ${mb} MB, NPU ${npus}"
    # 启动进程 / 会话入口（按 launch_pid 去重）
    printf '%s\n' "$sub" | awk -F'\t' '{print $10"\t"$11"\t"$12"\t"$13}' | sort -u | \
    while IFS=$'\t' read -r lpid luser lcmd entry; do
      if [ "$lpid" = "-" ]; then
        echo "   ${D}↳ 启动进程: ${lcmd}${R}"
      else
        echo "   ${C}↳ 启动进程:${R} ${lcmd}"
        echo "     ${D}host PID ${lpid} (user ${luser})   会话入口: ${entry}${R}"
      fi
    done
    # 容器 -> 进程
    printf '%s\n' "$sub" | awk -F'\t' '{print $9"\t"$8}' | sort -u | \
    while IFS=$'\t' read -r cname cid12; do
      echo "   ├─ 容器 ${cname}  ${D}[${cid12}]${R}"
      printf '%s\n' "$sub" | awk -F'\t' -v c="$cname" '$9==c' | \
      while IFS=$'\t' read -r npu chip pid pname mem hu ow ci cn lp lu lc en task; do
        echo "   │   NPU ${npu}/chip ${chip}  PID ${pid}  ${mem} MB  ${pname}  ${D}(host uid=${hu})${R}"
        echo "   │     ${D}task: ${task}${R}"
      done
    done
  done <<<"$owners"
}

print_table() {
  {
    printf 'NPU\tCHIP\tPID\tMEM_MB\tOWNER\tLAUNCH_PID\tCONTAINER\tTASK\n'
    printf '%s\n' "$RECORDS" | awk -F'\t' '{
      t=$14; if (length(t)>70) t=substr(t,1,67)"...";
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",$1,$2,$3,$5,$7,$10,$9,t
    }'
  } | column -t -s $'\t'
}

print_json() {
  printf '%s\n' "$RECORDS" | awk -F'\t' '{
    gsub(/"/,"\\\"");
    printf "{\"npu\":%s,\"chip\":%s,\"pid\":%s,\"proc\":\"%s\",\"mem_mb\":%s,\"host_user\":\"%s\",\"owner\":\"%s\",\"container\":\"%s\",\"container_id\":\"%s\",\"launch_pid\":\"%s\",\"launch_user\":\"%s\",\"launch_cmd\":\"%s\",\"session_entry\":\"%s\",\"task\":\"%s\"}\n",
    $1,$2,$3,$4,$5,$6,$7,$9,$8,$10,$11,$12,$13,$14
  }'
}

case "$MODE" in
  all)   print_devsummary; echo; print_tree; echo; echo "${B}明细表${R}"; print_table ;;
  tree)  print_tree ;;
  table) print_table ;;
  json)  print_json ;;
esac
