#!/usr/bin/env bash
# npu-who.sh — 汇报当前加速卡资源占用，并把每个占用进程回溯到
#              宿主机用户 / 启动脚本 / 容器 / 具体任务。
#
# 支持后端（自动探测，谁在 PATH 里就用谁；两者都在时优先 Ascend）:
#   - Ascend NPU  : `npu-smi info`             （如 910B/910C，行式表格，NPU 下辖多 Chip）
#   - 昆仑芯 XPU  : `xpu-smi`（P800，nvidia-smi 风格三行式表格，一卡一 die，无 Chip 细分）
#   可用 NPU_WHO_BACKEND=ascend|kunlun 强制指定（两个工具都装的机器上用得到）。
#
# 依赖: npu-smi 或 xpu-smi, awk, ps；可选: docker（能免 sudo 最好）、getent。
# 用法:
#   ./npu-who.sh            # 设备概览 + 归属树 + 明细表
#   ./npu-who.sh --tree     # 只看归属树
#   ./npu-who.sh --table    # 只看明细表（tab 分隔，方便 grep/awk）
#   ./npu-who.sh --json     # 明细以 JSON 行输出
#
# 原理（已在生产机验证）:
#   设备工具的进程表给出宿主机 PID
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
  -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
  *) echo "unknown arg: $1" >&2; exit 2 ;;
esac

if [ -t 1 ]; then B=$'\e[1m'; D=$'\e[2m'; R=$'\e[0m'; Y=$'\e[33m'; C=$'\e[36m'; else B=""; D=""; R=""; Y=""; C=""; fi

# ---------------------------------------------------------------------------
# 0) 探测本机加速卡后端
# ---------------------------------------------------------------------------
BACKEND="${NPU_WHO_BACKEND:-}"
if [ -z "$BACKEND" ]; then
  if command -v npu-smi >/dev/null 2>&1; then
    BACKEND="ascend"
  elif command -v xpu-smi >/dev/null 2>&1; then
    BACKEND="kunlun"
  fi
fi
case "$BACKEND" in
  ascend) command -v npu-smi >/dev/null || { echo "NPU_WHO_BACKEND=ascend 但未找到 npu-smi" >&2; exit 1; } ;;
  kunlun) command -v xpu-smi >/dev/null || { echo "NPU_WHO_BACKEND=kunlun 但未找到 xpu-smi" >&2; exit 1; } ;;
  *) echo "既没有 npu-smi（Ascend）也没有 xpu-smi（昆仑芯 P800）——本机不像是一台加速卡节点" >&2; exit 1 ;;
esac

case "$BACKEND" in
  ascend) DEVLABEL="NPU"; UTILLABEL="AICore"; MEMLABEL="HBM(MB)" ;;
  kunlun) DEVLABEL="XPU"; UTILLABEL="XPU-Util"; MEMLABEL="MEM(MB)" ;;
esac
devtag() {                                # $1=npu $2=chip -> 展示用设备标签
  if [ "$BACKEND" = "ascend" ]; then echo "${DEVLABEL} $1/chip $2"; else echo "${DEVLABEL} $1"; fi
}

docker_cmd() {
  command -v docker >/dev/null || return 1
  docker "$@" 2>/dev/null && return 0
  sudo -n docker "$@" 2>/dev/null
}

HOST=$(hostname)
NOW=$(date '+%Y-%m-%d %H:%M:%S %z')

# ---------------------------------------------------------------------------
# 1) 解析设备工具输出  ->  统一的 DEV / PROC 行
#    DEV\tnpu\tchip\tphy\tbus\tutil%\tmem_used_mb\tmem_total_mb
#    PROC\tnpu\tchip\tpid\tpname\tmem_mb
#    Ascend 一个 NPU 下有 0/1 两个 Chip（chip≠phy，phy 是全局 davinci 号）；
#    昆仑芯 P800 一卡一 die，没有 Chip 细分，固定 chip=0、phy=npu。
# ---------------------------------------------------------------------------
if [ "$BACKEND" = "ascend" ]; then
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
else
  # 昆仑芯 xpu-smi：nvidia-smi 风格，设备块占 3 行物理文本：
  #   行1 = | XPU  Name  Persistence-M | Bus-Id  Disp.A | ECC |
  #   行2 = | Fan Temp Perf Pwr | usedMiB / totalMiB | Util% Compute-M |
  #   行3 = |                   | L3-Usage           | SR-IOV M.       |
  # 进程表: | XPU  XI-ID  CI-ID  PID  Type  Process-name  Mem |
  SMI=$(xpu-smi 2>/dev/null)
  PARSED=$(printf '%s\n' "$SMI" | awk '
    BEGIN { want_mem=0 }
    /^\| *[0-9]+ +[A-Za-z]/ && /[0-9A-Fa-f]+:[0-9A-Fa-f]+:[0-9A-Fa-f]+\.[0-9]/ {  # 设备行1
       n=split($0,f,"|")
       split(f[2],g," "); npu=g[1]
       split(f[3],h," "); bus=h[1]
       pend_npu=npu; pend_bus=bus; want_mem=1
       next
    }
    want_mem==1 {                                 # 设备行2：紧跟在行1后面
       n=split($0,f,"|")
       split(f[3],mm,"/")
       used=mm[1]; gsub(/[^0-9]/,"",used)
       total=mm[2]; gsub(/[^0-9]/,"",total)
       split(f[4],uu," "); ai=uu[1]; gsub(/%/,"",ai)
       print "DEV\t" pend_npu "\t0\t" pend_npu "\t" pend_bus "\t" ai "\t" used "\t" total
       want_mem=0
       next
    }
    /^\| *[0-9]+ +[A-Za-z0-9\/]+ +[A-Za-z0-9\/]+ +[0-9]+ +/ {   # 进程行
       n=split($0,f,"|")
       m=split(f[2],tok," ")
       pid=tok[4]; pmem=tok[m]; gsub(/[^0-9]/,"",pmem)
       pname=tok[m-1]
       print "PROC\t" tok[1] "\t0\t" pid "\t" pname "\t" pmem
    }
  ')
fi

DEVROWS=$(printf '%s\n' "$PARSED" | awk -F'\t' '$1=="DEV"')
PROCROWS=$(printf '%s\n' "$PARSED" | awk -F'\t' '$1=="PROC"')

# ---------------------------------------------------------------------------
# 2) 逐进程回溯归属
#    输出记录(TSV): npu chip pid pname mem hostuser owner cid12 cname \
#                   launch_pid launch_user launch_cmd entry task
# ---------------------------------------------------------------------------
stat_owner_walkup() {                     # $1=path -> owning用户；本级 stat 权限不够就逐级上溯
  # 本机不少共享目录是 750（如 /data1/<name> 属主 rwx，其他人连 stat 子路径都 Permission denied），
  # 但目录条目本身（走它的父目录）通常还能 stat 到，所以按 dirname 逐级上溯找第一个能 stat 成功的层级。
  local p="$1" u
  while [ -n "$p" ] && [ "$p" != "/" ] && [ "$p" != "." ]; do
    u=$(stat -c %U "$p" 2>/dev/null) && { printf '%s' "$u"; return 0; }
    p=$(dirname "$p")
  done
  return 1
}
resolve_launcher() {                      # $1=owner  $2=uid  $3=mounts(换行分隔，可空)  -> "pid<TAB>user<TAB>cmd"
  # 注意: `ps -o user=` 会把 >8 字符用户名截断成 kzhang5+，必须按 uid 匹配
  #
  # 同一用户常常同时跑好几个容器/任务，仅按 uid 匹配启动进程会把毫不相关的启动命令
  # 张冠李戴地挂到别的容器上（例如两个并发 docker run 只有 --pull=never 之类通用参数
  # 相似，实际服务的是不同容器）。有挂载路径信息时，要求候选命令行把这个容器的全部
  # bind-mount Source 路径都包含到，才当作确认匹配；否则（没有挂载信息可比对，比如
  # 裸进程没走 docker）退回旧逻辑：只有唯一候选时才敢用，多个候选就宁可报"未找到"。
  local mounts="$3"
  ps -ww -eo pid=,uid=,args= 2>/dev/null | awk -v o="$1" -v u="$2" -v mounts="$mounts" '
    BEGIN { nm = split(mounts, marr, "\n") }
    $2==u && $0 ~ /(^| )([^ ]*run\.py|torchrun|mpirun)( |$)|flagperf|docker +run|compose +up/ {
      pid=$1; $1=$2=""; sub(/^ +/,"")
      cmd=$0
      n++
      cand_pid[n]=pid; cand_cmd[n]=cmd
      score=0
      for (i=1;i<=nm;i++) { if (marr[i] != "" && index(cmd, marr[i]) > 0) score++ }
      cand_score[n]=score
      if (nm>0 && score==nm && !full_i) full_i=n
    }
    END {
      if (n==0) { exit }
      if (nm>0) {
        if (full_i) print cand_pid[full_i] "\t" o "\t" cand_cmd[full_i]
        exit
      }
      if (n==1) print cand_pid[1] "\t" o "\t" cand_cmd[1]
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
        u=$(stat_owner_walkup "$m") || continue
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
    IFS=$'\t' read -r lpid luser lcmd < <(resolve_launcher "$owner" "${ouid:--1}" "$mounts")
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
  echo "${B}${DEVLABEL} 设备概览${R}  @ ${HOST}   ${NOW}  ${D}[backend: ${BACKEND}]${R}"
  if [ -z "$DEVROWS" ]; then echo "  (无法解析设备表)"; return; fi
  printf '  %-4s %-5s %-6s %-15s %-7s %-16s %s\n' "$DEVLABEL" 'CHIP' 'PHY-ID' 'BUS-ID' "$UTILLABEL" "$MEMLABEL" '进程'
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
  [ -z "$RECORDS" ] && { echo "  ✅ 当前没有进程占用 ${DEVLABEL}"; return; }
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
    echo "${B}● owner=${o}${R} (uid ${uid})  —  ${cnt} proc, ${mb} MB, ${DEVLABEL} ${npus}"
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
        echo "   │   $(devtag "$npu" "$chip")  PID ${pid}  ${mem} MB  ${pname}  ${D}(host uid=${hu})${R}"
        echo "   │     ${D}task: ${task}${R}"
      done
    done
  done <<<"$owners"
}

print_table() {
  {
    printf '%s\tCHIP\tPID\tMEM_MB\tOWNER\tLAUNCH_PID\tCONTAINER\tTASK\n' "$DEVLABEL"
    printf '%s\n' "$RECORDS" | awk -F'\t' '{
      t=$14; if (length(t)>70) t=substr(t,1,67)"...";
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",$1,$2,$3,$5,$7,$10,$9,t
    }'
  } | column -t -s $'\t'
}

print_json() {
  printf '%s\n' "$RECORDS" | awk -F'\t' -v backend="$BACKEND" '{
    gsub(/"/,"\\\"");
    printf "{\"backend\":\"%s\",\"npu\":%s,\"chip\":%s,\"pid\":%s,\"proc\":\"%s\",\"mem_mb\":%s,\"host_user\":\"%s\",\"owner\":\"%s\",\"container\":\"%s\",\"container_id\":\"%s\",\"launch_pid\":\"%s\",\"launch_user\":\"%s\",\"launch_cmd\":\"%s\",\"session_entry\":\"%s\",\"task\":\"%s\"}\n",
    backend,$1,$2,$3,$4,$5,$6,$7,$9,$8,$10,$11,$12,$13,$14
  }'
}

case "$MODE" in
  all)   print_devsummary; echo; print_tree; echo; echo "${B}明细表${R}"; print_table ;;
  tree)  print_tree ;;
  table) print_table ;;
  json)  print_json ;;
esac
