#!/usr/bin/env bash
# Non-interactive helpers for the ELSC Slurm cluster (ssh alias elsc-cluster), safe to drive from a phone session: every
# call is BatchMode (never waits for a password), quiet (no login banner), gives up after 20 s if the cluster is down and
# after 2 min if a remote command hangs.
#
#   cluster.sh status                  per job name: running / pending tasks, GPUs and CPUs in use, why tasks pend
#   cluster.sh watch [-i SEC] ID...    one line per change in the jobs' state counts (array, task or raw job IDs);
#                                      exits once none is queued and prints the final sacct states
#                                      (exit 0 all COMPLETED, 3 otherwise or unknown ID, 2 cluster unreachable)
#   cluster.sh log ID [LINES]          tail of each of the job's logs (Slurm stdout/stderr and logs named after the ID,
#                                      newest 4); default 40 lines; progress bars are split into lines
#   cluster.sh push [-n] SRC... DEST   rsync repo paths (relative to the repo root) to ~/DEST on the cluster, keeping the
#                                      repo layout; refuses main/DATA and its copies (DATA_light, DATA_MANIFEST): code
#                                      only, rebuild checkpoints on the cluster; skips checkpoints, archives, caches and
#                                      files over 5 MB (skips are named); -n = dry run
#   cluster.sh pull [-f] SRC [DEST]    rsync ~/SRC (globs * ? [ ]) into DEST (repo-relative; default
#                                      main/DATA/experimental/cluster_runs), keeping SRC's path below DEST (a /./ in SRC
#                                      keeps only the part after it); refuses more than 5 GB without -f
#
# Anything else: ssh elsc-cluster '...'. To be told when jobs end, run `cluster.sh watch ID` as a background task: its
# exit wakes the session, which can then send a phone notification. Pushing into ~/cgw changes the code that queued
# jobs start with.
set -uo pipefail

HOST=${CLUSTER_HOST:-elsc-cluster}
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 -o LogLevel=ERROR)
ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../.." && pwd)

die() { echo "cluster.sh: $*" >&2; exit 1; }
is_id() { [[ $1 =~ ^[0-9]+(_[0-9]+)?$ ]]; }
# rsh CMD: run CMD on the cluster; stdout in OUT, the start of stderr in ERR, returns the exit status
rsh() {
    local errf rc
    errf=$(mktemp) || return 1
    OUT=$(timeout 120 "${SSH[@]}" -n "$HOST" "$1" 2>"$errf"); rc=$?
    ERR=$(head -c 300 "$errf" | tr '\n' ' '); rm -f "$errf"
    return $rc
}

status() {
    rsh 'squeue -u "$USER" -h -r -o "%j|%T|%C|%b|%r"' || die "cluster unreachable: $ERR"
    [[ -n $OUT ]] || { echo "no jobs queued"; return 0; }
    awk -F'|' '
        NF < 5 { next }
        { n = $1; names[n]
          if ($2 == "RUNNING") { run[n]++; cpu[n] += $3; if ($4 ~ /gpu/) { k = split($4, g, ":"); gpu[n] += g[k] } }
          else if ($2 == "PENDING") { pend[n]++; if (!((n, $5) in sr)) { sr[n, $5]; why[n] = why[n] (why[n] == "" ? "" : ",") $5 } }
          else { oth[n]++; if (!((n, $2) in ss)) { ss[n, $2]; ost[n] = ost[n] (ost[n] == "" ? "" : ",") $2 } } }
        END { for (n in names) {
                  printf "%-16s run %3d  pend %3d  gpu %2d  cpu %4d%s%s\n", n, run[n], pend[n], gpu[n], cpu[n],
                         (why[n] == "" ? "" : "  (" why[n] ")"), (oth[n] ? "  other " oth[n] " (" ost[n] ")" : "") | "sort"
                  R += run[n]; P += pend[n]; G += gpu[n]; C += cpu[n] }
              close("sort")
              printf "%-16s run %3d  pend %3d  gpu %2d  cpu %4d\n", "TOTAL", R, P, G, C }' <<<"$OUT"
}

watch() {
    local interval=120
    if [[ ${1:-} == -i ]]; then interval=${2:-}; shift; (( $# )) && shift; fi
    [[ $interval =~ ^[0-9]+$ ]] && (( 10#$interval >= 10 )) || die "watch: -i takes whole seconds >= 10"
    interval=$((10#$interval))
    (( $# )) || die "watch: give one or more job IDs"
    local id cur prev="-" fails=0 tries ok=0
    for id in "$@"; do is_id "$id" || die "watch: bad job ID '$id'"; done
    while :; do
        if rsh 'squeue -u "$USER" -h -r -o "%i %A %T"'; then
            (( fails >= 3 )) && echo "$(date +%H:%M) cluster reachable again"
            fails=0
            cur=$(awk -v ids="$*" 'BEGIN { k = split(ids, a, " "); for (i = 1; i <= k; i++) want[a[i]] }
                      NF >= 3 { b = $1; sub(/_.*/, "", b); if (($1 in want) || ($2 in want) || (b in want)) print $3 }' <<<"$OUT" |
                  sort | uniq -c | awk '{printf "%s%d %s", (NR > 1 ? ", " : ""), $1, $2}')
            if [[ $cur != "$prev" ]]; then echo "$(date +%H:%M) ${cur:-none queued}"; prev=$cur; fi
            [[ -z $cur ]] && break
        else
            fails=$((fails + 1))
            (( fails == 3 )) && echo "$(date +%H:%M) cluster unreachable for $fails polls: $ERR"
            (( fails >= 30 )) && { echo "$(date +%H:%M) giving up: cluster unreachable for $fails polls"; return 2; }
        fi
        sleep "$interval"
    done
    # accounting can lag the queue: retry while sacct is unreachable or still shows live states
    for tries in 1 2 3 4; do
        if rsh "sacct -X -n -P -j $(IFS=,; echo "$*") -o JobID,JobName,State,ExitCode,Elapsed"; then
            ok=1
            awk -F'|' 'NF >= 5 && $3 ~ /^(PENDING|RUNNING|REQUEUED|SUSPENDED|COMPLETING|CONFIGURING|RESIZING)/ {live = 1}
                       END {exit !live}' <<<"$OUT" || break
        fi
        (( tries < 4 )) && sleep 30
    done
    (( ok )) || { echo "sacct unreachable: $ERR"; return 2; }
    awk -F'|' '
        NF < 5 { next }
        { st = $3; sub(/ .*/, "", st); w = 1
          if ($1 ~ /_\[/) { r = $1; sub(/.*_\[/, "", r); sub(/[%\]].*/, "", r); w = 0
              k = split(r, p, ","); for (i = 1; i <= k; i++) w += (split(p[i], q, "-") == 2) ? q[2] - q[1] + 1 : 1 }
          n[st] += w; rows++
          if (st != "COMPLETED" && ++b <= 12) bad = bad sprintf("\n  %s %s: %s, exit %s, after %s", $1, $2, $3, $4, $5) }
        END { if (!rows) { print "final: sacct has no record of these jobs"; exit 3 }
              printf "final:"; for (s in n) printf " %d %s", n[s], s
              if (b > 12) bad = bad sprintf("\n  ... and %d more rows", b - 12)
              print bad; exit (b > 0 ? 3 : 0) }' <<<"$OUT"
}

log() {
    local id=${1:-} n=${2:-40} rc
    is_id "$id" || die "log: bad job ID '$id'"
    [[ $n =~ ^[0-9]+$ ]] || die "log: LINES must be a whole number"
    rsh "id=$id n=$((10#$n))"'
        files=$( { scontrol show job "$id" 2>/dev/null | sed -n "s/^ *Std\(Out\|Err\)=//p"
                   ls -d ~/cgw/logs/*-"$id"[._]* ~/cgw/*-"$id"[._]* ~/*-"$id"[._]* 2>/dev/null
                 } | awk "!seen[\$0]++" | xargs -r ls -t -- 2>/dev/null )
        [ -n "$files" ] || { echo "no log found for job $id"; exit 1; }
        k=$(printf "%s\n" "$files" | wc -l)
        for f in $(printf "%s\n" "$files" | head -4); do
            echo "== $f"; tail -c 200000 "$f" | tr "\r" "\n" | tail -n "$n" | awk 1
        done
        [ "$k" -le 4 ] || echo "== and $((k - 4)) older log files"'
    rc=$?
    [[ -n $OUT ]] && printf '%s\n' "$OUT"
    [[ $rc == 0 || -n $OUT ]] || die "cluster unreachable: $ERR"
    return $rc
}

push() {
    local dry=()
    [[ ${1:-} == -n ]] && { dry=(--dry-run); shift; }
    (( $# >= 2 )) || die "push: usage: push [-n] SRC... DEST"
    local dest=${!#} rels=() s abs rel
    [[ $dest =~ ^[A-Za-z0-9._][A-Za-z0-9._/-]*$ && $dest != *..* ]] ||
        die "push: DEST must be a relative path under the cluster home, got '$dest'"
    for s in "${@:1:$#-1}"; do
        abs=$(cd "$ROOT" && realpath -e -- "$s") || die "push: no such path in the repo: $s"
        [[ $abs == "$ROOT"/* ]] || die "push: $s is not inside the repo (name folders, not the repo root)"
        rel=${abs#"$ROOT"/}
        [[ /$rel/ =~ /DATA[^/]*/ ]] &&
            die "push: refusing $s: main/DATA and its copies never go to the cluster (code only; rebuild checkpoints there)"
        [[ /$rel/ == */.git/* ]] && die "push: refusing $s: .git"
        rels+=("./$rel")
    done
    [[ $dest == cgw || $dest == cgw/* ]] && echo "note: queued jobs start with the code in ~/cgw; this changes it" >&2
    cd "$ROOT" && rsync -azR --mkpath --timeout=120 --itemize-changes --info=skip1 "${dry[@]}" -e "${SSH[*]}" \
        --max-size=5MB --exclude='DATA*' --exclude=.git/ --exclude=__pycache__/ --exclude=figure_bin/ \
        --exclude='*.pth' --exclude='*.pt' --exclude='*.ckpt' --exclude='*.zip' --exclude='*.gif' \
        -- "${rels[@]}" "$HOST:$dest/"
}

pull() {
    local force=0 args=() a ok='^[]A-Za-z0-9._/*?[+=@%][]A-Za-z0-9._/*?[+=@%-]*$'
    for a in "$@"; do if [[ $a == -f ]]; then force=1; else args+=("$a"); fi; done
    (( ${#args[@]} == 1 || ${#args[@]} == 2 )) || die "pull: usage: pull [-f] SRC [DEST]"
    local src=${args[0]#\~/} dest=${args[1]:-main/DATA/experimental/cluster_runs}
    [[ $src =~ $ok && $src != *..* ]] || die "pull: SRC may hold only path characters and the globs * ? [ ], got '${args[0]}'"
    [[ $dest != -* ]] || die "pull: DEST may not start with '-'"
    [[ $dest == /* ]] || dest=$ROOT/$dest
    rsh "du -scb -- $src 2>/dev/null | tail -1 | cut -f1" || die "pull: cluster unreachable: $ERR"
    [[ $OUT =~ ^[0-9]+$ ]] && (( OUT > 0 )) || die "pull: nothing on the cluster matches $src"
    (( OUT <= 5000000000 || force )) || die "pull: $src is $((OUT / 1000000)) MB; pass -f to pull it anyway"
    mkdir -p -- "$dest" || die "pull: cannot create $dest"
    rsync -azR --timeout=120 --itemize-changes -e "${SSH[*]}" -- "$HOST:$src" "$dest/"
}

case ${1:-} in
    status|watch|log|push|pull) cmd=$1; shift; "$cmd" "$@" ;;
    *) sed -n '2,/^set -uo/{/^set -uo/d; s/^# \{0,1\}//; p}' "${BASH_SOURCE[0]}"; exit 1 ;;
esac
