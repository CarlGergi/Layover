#!/usr/bin/env bash
# Run the whole GPU session on a rented x86_64 Linux machine, from this one, over SSH.
#
#   scripts/remote_gpu.sh user@host [-p port] [-i key] [options]
#
# Order: preflight, sync, then detached on the machine setup smoke baselines gpu-tests sanitize
# traces reproduce, then results/ and traces/ come back to results/gpu-<timestamp>/. Smoke comes
# right after setup and before the baselines, so a broken setup costs minutes. Its results and
# the refined hours of the plan come back and print as soon as it ends.
#
#   --stage NAME            run only this stage, even if done (also preflight or sync alone)
#   --from NAME             run from this stage on, even if done
#   --stages "A B ..."      run only these stages, in the usual order, skipping any already done
#   --continue-on-failure   let later stages run after a failed one
#   --plan NAME             run this plan of the matrix config (default: the config's default plan);
#                           smoke prices the same plan, and a change of plan runs smoke again
#   --max-hours H           stop the matrix after H hours, and stop the session after smoke if its
#                           refined low estimate is already over that. --plan and --max-hours stay
#                           with the session on a resume or a restore unless given again
#   --continue-over-budget  go on past smoke even though its low estimate is over --max-hours
#   --stop-after-smoke      stop after smoke, to read its refined hours before paying for the rest;
#                           the same command without this flag goes on from there
#   --detach                start the run and return; nothing here needs to stay awake. Come back
#                           with --status, or --fetch-only to copy results here
#   --no-auto-shutdown      leave the machine running when the run ends (it shuts down by default)
#   --max-session-hours H   dead-man limit: the machine shuts down this long after the run starts,
#                           whatever is happening (default: 2 with --stop-after-smoke, --max-hours
#                           plus 6, or 48)
#   --leave-running         with --fetch-only: do not shut down a machine whose run has ended
#   --runpod                the machine is a RunPod pod, a container: it stops itself with runpodctl,
#                           the run directory defaults to /workspace/layover-run on the volume disk,
#                           which a stop keeps, and the caches go there too (see below)
#   --restore-from DIR      on a new machine: go on with the session in DIR, results/gpu-<timestamp>.
#                           Setup and baselines rerun, as they live on the machine. Everything in
#                           DIR's results/ and traces/ goes up, new results come back to DIR, and
#                           stages done at this same commit are skipped
#   --spot-notice           watch the instance metadata for a spot interruption notice; on one,
#                           log it and copy results back at once and then every 10 s
#   --status                check in on the session; changes nothing
#   --fetch-only            copy results back and print the time table
#   --dry-run               print what would run; contacts nothing
#   --hourly-usd RATE       print the session's cost at this rate
#   --min-disk-gb N         free disk Docker's filesystem must have (default: 200)
#   --min-gpu-mib N         GPU memory the machine must have (default: 22000)
#   --min-ram-gb N          host RAM the machine must have (default: 64; 0 skips the check)
#   --allow-iam-role        go ahead even though the instance has an IAM role, in the traces stage too
#   --remote-dir DIR        run directory on the machine (default: layover-run, under home)
#   --ssh-option KEY=VALUE  extra ssh -o option, repeatable
#   --fetch-every SECONDS   copy partial results back this often (default: 600)
#   --fetch-every-runs S    the same during traces and reproduce, if that is sooner (default: 60)
#   --poll SECONDS          how often to look for new log output (default: 5)
#   --reconnect-timeout S   how long to keep retrying a lost connection (default: 1800)
#   --no-secret-scan        skip the check for credentials in the files to sync
#
# The work runs detached, so an SSH drop or a closed laptop costs nothing: re-running the same
# command attaches to it, or resumes from the first stage not done. This script never deletes
# anything on this machine.
#
# The machine never bills idle. When the run ends, finished, over budget, or failed, it shuts
# itself down a minute later, which gives a watching session time for its last copy. A shutdown
# timer set when the run starts covers a hang or a killed runner. On AWS an instance's own
# shutdown stops it by default and its disk keeps the results; check that the instance's shutdown
# behavior is Stop, not Terminate, or those results are lost with it. To fetch later, start the
# instance and run --fetch-only, which shuts it down again afterwards.
#
# A RunPod pod is a container, where shutdown does nothing, so with --runpod a timer on the pod
# asks RunPod to stop it through runpodctl and the pod's own API key, which RunPod gives every
# pod. A stop ends the GPU's billing and wipes the container disk, while /workspace survives: the
# run directory, the environments and the model weights live there. The toolkit lives on the
# container disk, so a run on a restarted pod does setup again. Nothing here needs Docker unless
# the traces stage is in the run, so a container can run every other stage. Delete the pod once
# the results are here: a stopped pod still bills for its volume disk.
#
# Spot instances. rsync sends only what changed, so results can come back every minute during the
# stages that run for hours, and run_matrix.py writes each run whole as it ends. So a run that ended
# a minute and one copy before the instance vanished is on this machine, which leaves the run in
# flight, and at worst one that ended in that last gap, as all that is lost. AWS's two-minute notice
# covers that gap. This holds only while this script is running and this machine is online: what
# ended while nothing was fetching is lost with the instance. When a connection stays down, this
# prints the command that resumes on a new instance; run that, and the session directory keeps
# accumulating there.
#
# Exit codes: 0 every planned stage done, 1 connection lost or other error, 2 bad usage,
# 10 preflight, 11 sync, 12 setup, 13 gpu-tests, 14 sanitize, 15 baselines, 16 traces,
# 17 reproduce, 18 fetch, 19 smoke, 20 over budget, 130 interrupted. With several failures, the first in stage order;
# a failed fetch comes before all of them, since the results are then still only over there.
set -u -o pipefail

STAGES="setup smoke baselines gpu-tests sanitize traces reproduce"
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STATE="$ROOT/.layover-run"

# ---------------------------------------------------------------------------------------
# The remote side: the machine runs this same file, synced, through the __ subcommands.
# ---------------------------------------------------------------------------------------

stage_command() {
    case "$1" in
        setup) echo "bash scripts/setup_gpu.sh" ;;
        smoke) echo "make gpu-smoke" ;;
        gpu-tests) echo "make test-gpu" ;;
        sanitize) echo "make sanitize" ;;
        # Launches each baseline once, so a broken one fails here and not hours into the matrix.
        baselines) echo "bash scripts/setup_baselines.sh && .venv/bin/python bench/run_matrix.py --preflight-only --preflight-kinds vllm sglang --out results/matrix_preflight" ;;
        traces) echo "make traces" ;;
        reproduce) echo "make reproduce" ;;
    esac
}

runner_alive() {
    local pid
    pid=$(cat "$STATE/runner.pid" 2>/dev/null) || return 1
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null || return 1
    # A reboot can hand the pid to something else. Minimal images have no ps to tell.
    command -v ps >/dev/null 2>&1 || return 0
    case "$(ps -p "$pid" -o args= 2>/dev/null)" in
        *__runner*) return 0 ;;
    esac
    return 1
}

# What this session was asked for, as "key value" lines. A restore takes it from a directory, so it
# is only ever read this way, never sourced.
setting() { sed -n "s/^$1 //p" "$STATE/settings" 2>/dev/null; }

# make hands this to a shell inside a recipe, so it has to survive a second parse.
matrix_args() {
    local args=""
    [ -z "$1" ] || args="--plan $(printf '%q' "$1") "
    [ -z "$2" ] || args="$args--max-hours $(printf '%q' "$2")"
    printf '%s' "${args% }"
}

export_stage_settings() {
    local plan max
    plan=$(setting plan) max=$(setting max_hours)
    # Unset rather than empty: make's ?= takes an empty variable as already set.
    unset MATRIX_ARGS TRACES_ARGS SMOKE_PLAN
    [ -z "$plan$max" ] || export MATRIX_ARGS="$(matrix_args "$plan" "$max")"
    [ -z "$plan" ] || export SMOKE_PLAN="$plan"
    [ "$(setting allow_iam_role)" != 1 ] || export TRACES_ARGS=--allow-iam-role
}

hours_over() { awk -v low="$1" -v max="$2" 'BEGIN { exit !(low + 0 > max + 0) }'; }

# Decided here and not by the side that watches, which may be a closed laptop by now. Runs again on
# every resume, so a stopped session stays stopped until someone answers it. Only a plan that holds
# the matrix can run over; a lone stage asked for by name goes ahead.
budget_allows() {
    local max low=""
    max=$(setting max_hours)
    [ -n "$max" ] || return 0
    case " $1 " in *" reproduce "*) ;; *) return 0 ;; esac
    [ ! -f results/smoke/refined_hours.txt ] || read -r low _ < results/smoke/refined_hours.txt
    case "$low" in
        ''|*[!0-9.]*)
            echo "=== [budget] smoke left no estimate to hold against --max-hours $max; the matrix still stops at it"
            return 0
            ;;
    esac
    hours_over "$low" "$max" || return 0
    if [ "$(setting continue_over_budget)" = 1 ]; then
        echo "=== [budget] smoke's low estimate of $low hours is over --max-hours $max; going on, as --continue-over-budget asks"
        return 0
    fi
    echo "$low $max" > "$STATE/budget_stop"
    echo "=== [budget] STOPPED: smoke's low estimate of $low hours is over --max-hours $max. Nothing past smoke runs."
    return 1
}

# One line per stage: status start end exit commit. Renamed into place, so a fetch or a poll
# never sees half of it.
write_stage() {
    local commit
    commit=$(cat "$STATE/synced_commit" 2>/dev/null || echo -)
    printf '%s %s %s %s %s\n' "$2" "$3" "$4" "$5" "$commit" > "$STATE/stages/.$1.tmp"
    mv -f "$STATE/stages/.$1.tmp" "$STATE/stages/$1"
}

# Setup can add this user to the docker group, which only new logins pick up, and this process
# outlives setup. sg gives the later stages the group without a reconnect.
run_stage_shell() {
    if id -nG "$(id -un)" | grep -qw docker && ! id -nG | grep -qw docker; then
        sg docker -c "$1"
    else
        bash -c "$1"
    fi
}

# RunPod hands the pod's id and its own API key to the container's first process, and its images
# write them to /etc/rp_environment for login shells. A command over ssh gets neither by itself.
runpod_env() {
    local line
    [ -z "${RUNPOD_POD_ID:-}" ] || return 0
    # shellcheck disable=SC1091
    [ ! -r /etc/rp_environment ] || . /etc/rp_environment
    [ -z "${RUNPOD_POD_ID:-}" ] && [ -r /proc/1/environ ] || return 0
    while IFS= read -r -d '' line; do
        case "$line" in RUNPOD_POD_ID=*|RUNPOD_API_KEY=*) export "${line?}" ;; esac
    done < /proc/1/environ
}

# runpodctl 2 says "pod stop ID", and earlier versions "stop pod ID".
runpod_ctl() {
    runpodctl pod "$1" "$2" > /dev/null 2>&1 || runpodctl "$1" pod "$2" > /dev/null 2>&1
}

cancel_pod_stop() {
    local pid
    pid=$(cat "$STATE/stop_timer.pid" 2>/dev/null) || return 0
    rm -f "$STATE/stop_timer.pid"
    [ -n "$pid" ] || return 0
    # Its own process group where setsid exists, so the sleep goes with it.
    kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
}

# Runs detached, and a newer timer replaces it by killing it.
stop_pod_after() {
    echo $$ > "$STATE/stop_timer.pid"
    printf '%s %s\n' "$1" "$2" > "$STATE/stop_timer"
    sleep $(($1 * 60))
    runpod_env
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') stopping pod ${RUNPOD_POD_ID:-unknown}: $2"
    rm -f "$STATE/stop_timer.pid"
    runpod_ctl stop "${RUNPOD_POD_ID:-}" || echo "could not stop the pod; it keeps billing until someone does"
}

schedule_pod_stop() {
    local detach=
    cancel_pod_stop
    command -v setsid >/dev/null 2>&1 && detach=setsid
    nohup $detach bash "$ROOT/scripts/remote_gpu.sh" __stop_pod_after "$1" "$2" >> "$STATE/logs/stop.log" 2>&1 < /dev/null &
    for _ in $(seq 100); do
        [ -s "$STATE/stop_timer.pid" ] && return 0
        sleep 0.05
    done
    return 1
}

# The machine's own shutdown, so that it never bills idle. Asking for one replaces any pending.
schedule_shutdown() {
    [ "$(setting auto_shutdown)" = 1 ] || return 0
    if [ "$(setting provider)" = runpod ]; then
        schedule_pod_stop "$1" "$2" && return 0
    else
        sudo -n shutdown -h "+$1" "$2" > /dev/null 2>&1 && return 0
    fi
    echo "=== WARNING: could not schedule a shutdown; the machine will keep billing until stopped"
}

# After a fetch, for a machine that was started only to be fetched from.
stop_machine_soon() {
    if [ "$(setting provider)" = runpod ]; then
        schedule_pod_stop 1 "Layover results fetched"
    else
        sudo -n shutdown -h +1 "Layover results fetched" > /dev/null
    fi
}

runner_ends() {
    echo "=== runner finished $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    if [ "$(setting auto_shutdown)" = 1 ]; then
        echo "=== the machine shuts down in a minute (auto-shutdown)"
        schedule_shutdown 1 "Layover run ended"
    fi
}

# On RunPod the volume keeps the run directory through a stop, and the container disk does not:
# the caches move to the volume, and a new container runs setup again for the toolkit it lost.
runpod_stage_setup() {
    local stage
    export XDG_CACHE_HOME="$ROOT/.cache" HF_HOME="$ROOT/.cache/huggingface" UV_PYTHON_INSTALL_DIR="$ROOT/.cache/uv-python"
    [ "$(cut -d' ' -f1 "$STATE/stages/setup" 2>/dev/null)" = "done" ] || return 0
    [ ! -f "$HOME/.layover-setup-here" ] || return 0
    echo "=== [setup] this container is new, and its disk lost what setup installed: setup runs again"
    rm -f "$STATE/stages/setup"
    for stage in "$@"; do
        [ "$stage" != setup ] || return 0
    done
    return 1
}

remote_runner() {
    local keep_going=$1 stage start end rc status budget_checked=0 minutes
    shift
    echo $$ > "$STATE/runner.pid"
    trap runner_ends EXIT
    minutes=$(awk -v h="$(setting deadman_hours)" 'BEGIN { m = int(h * 60); print (m > 0 ? m : 2880) }')
    [ "$(setting auto_shutdown)" != 1 ] || echo "=== dead-man: the machine shuts down in $minutes minutes whatever happens"
    schedule_shutdown "$minutes" "Layover dead-man limit reached"
    cd "$ROOT" || exit 1
    unset TRITON_INTERPRET
    export PYTHONUNBUFFERED=1
    export_stage_settings
    if [ "$(setting provider)" = runpod ]; then
        runpod_stage_setup "$@" || set -- setup "$@"
    fi
    # Only the traces stage runs containers, so setup leaves Docker alone without it.
    case " $* " in *" traces "*) ;; *) export LAYOVER_NO_DOCKER=1 ;; esac
    for stage in "$@"; do
        status=$(cut -d' ' -f1 "$STATE/stages/$stage" 2>/dev/null)
        if [ "$status" = "done" ]; then
            echo "=== [$stage] already done, skipping"
            continue
        fi
        case "$stage" in
            setup|smoke) ;;
            *)
                [ "$budget_checked" = 1 ] || budget_allows "$*" || break
                budget_checked=1
                ;;
        esac
        start=$(date +%s)
        write_stage "$stage" running "$start" - -
        echo "=== [$stage] started $(date -u '+%Y-%m-%dT%H:%M:%SZ'): $(stage_command "$stage")"
        (
            # Non-login ssh shells have neither the toolkit nor the environment on PATH.
            # shellcheck disable=SC1091
            [ -f .venv/cuda_env.sh ] && . .venv/cuda_env.sh
            export PATH="$PWD/.venv/bin:$PATH"
            run_stage_shell "$(stage_command "$stage")"
        ) 2>&1 < /dev/null | tee -a "$STATE/logs/$stage.log"
        rc=${PIPESTATUS[0]}
        end=$(date +%s)
        if [ "$rc" -eq 0 ]; then
            write_stage "$stage" "done" "$start" "$end" 0
            [ "$stage" != setup ] || [ "$(setting provider)" != runpod ] || : > "$HOME/.layover-setup-here"
            echo "=== [$stage] done in $((end - start))s"
        else
            write_stage "$stage" failed "$start" "$end" "$rc"
            echo "=== [$stage] FAILED with exit $rc after $((end - start))s"
            [ "$keep_going" = 1 ] || break
        fi
    done
}

remote_start() {
    local keep_going=$1 detach=
    shift
    rm -f "$STATE/runner.pid"
    echo "$*" > "$STATE/plan"
    # Emptied, not removed: a fetch never deletes, so the copy on the other side must be overwritten.
    : > "$STATE/budget_stop"
    # Everything redirected, or ssh would wait on the open descriptors. setsid takes the runner
    # out of the session, so the connection's hangup cannot reach it.
    command -v setsid >/dev/null 2>&1 && detach=setsid
    nohup $detach bash "$ROOT/scripts/remote_gpu.sh" __runner "$keep_going" "$@" \
        >> "$STATE/logs/runner.log" 2>&1 < /dev/null &
    for _ in $(seq 100); do
        [ -f "$STATE/runner.pid" ] && return 0
        sleep 0.05
    done
    echo "the runner did not start; the end of its log:" >&2
    tail -n 20 "$STATE/logs/runner.log" >&2
    return 1
}

# Whether AWS has given this spot instance notice. Off AWS nothing answers, within a second.
spot_notice() {
    local imds=http://169.254.169.254/latest token answer
    command -v curl >/dev/null 2>&1 || return 1
    token=$(curl -fsS --noproxy '*' --connect-timeout 1 -m 2 -X PUT -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' "$imds/api/token" 2>/dev/null) || return 1
    answer=$(curl -fsS --noproxy '*' --connect-timeout 1 -m 2 -H "X-aws-ec2-metadata-token: $token" "$imds/meta-data/spot/instance-action" 2>/dev/null) || return 1
    [ -n "$answer" ] || return 1
    echo "$answer" > "$STATE/logs/spot_notice.json"
}

# First line: alive, log size, smoke as status:end, the stage running, and any spot notice. Then
# the log from the offset to that size. Liveness is read first, because a dead runner writes
# nothing more, so the size read after it is final.
remote_poll() {
    local offset=$1 spot=${2:-0} log="$STATE/logs/runner.log" size=0 alive=0 smoke=none current=- notice=- file status end
    runner_alive && alive=1
    [ -f "$log" ] && size=$(($(wc -c < "$log")))
    [ "$offset" -ge 0 ] || offset=$((size > 4096 ? size - 4096 : 0))
    [ "$offset" -le "$size" ] || offset=$size
    if [ -f "$STATE/stages/smoke" ]; then
        read -r status _ end _ < "$STATE/stages/smoke"
        smoke="$status:$end"
    fi
    for file in "$STATE"/stages/*; do
        [ -f "$file" ] || continue
        read -r status _ < "$file"
        [ "$status" != running ] || current=${file##*/}
    done
    [ "$spot" != 1 ] || ! spot_notice || notice=notice
    echo "$alive $size $smoke $current $notice"
    [ "$size" -le "$offset" ] || tail -c +$((offset + 1)) "$log" | head -c $((size - offset))
    # The log may have grown past the size, so head can cut tail off. Only ssh failing means a cut reply.
    return 0
}

remote_status() {
    local stage status start end rc commit alive=0
    runner_alive && alive=1
    echo "session:  $(cat "$STATE/session" 2>/dev/null || echo none)"
    echo "synced:   $(cat "$STATE/synced_commit" 2>/dev/null || echo unknown)"
    echo "plan:     $(cat "$STATE/plan" 2>/dev/null || echo none)"
    if [ "$alive" = 1 ]; then
        echo "runner:   running, pid $(cat "$STATE/runner.pid") (to stop it and its work: kill -- -$(cat "$STATE/runner.pid"))"
    else
        echo "runner:   not running"
    fi
    printf '\n%-18s %-12s %-11s %-11s %-5s %s\n' stage status start end exit commit
    for stage in preflight sync $STAGES; do
        status=pending start=- end=- rc=- commit=-
        [ -f "$STATE/stages/$stage" ] && read -r status start end rc commit < "$STATE/stages/$stage"
        [ "$status" = running ] && [ "$alive" = 0 ] && status=interrupted
        printf '%-18s %-12s %-11s %-11s %-5s %s\n' "$stage" "$status" "$start" "$end" "$rc" "$commit"
    done
    [ -f "$STATE/logs/runner.log" ] && printf '\n--- last lines of the log ---\n' && tail -n 20 "$STATE/logs/runner.log"
    return 0
}

case "${1:-}" in
    __*)
        command=$1
        shift
        mkdir -p "$STATE/logs" "$STATE/stages"
        case "$command" in
            __runner) remote_runner "$@" ;;
            __start) remote_start "$@" ;;
            __poll) remote_poll "$@" ;;
            __status) remote_status ;;
            __alive) runner_alive && echo 1 || echo 0 ;;
            __stop_machine) stop_machine_soon ;;
            __stop_pod_after) stop_pod_after "$@" ;;
            __session) cat "$STATE/session" ;;
            __mark) name=$1; shift; write_stage "$name" "$@" ;;
            # Kept as it was, with the commit it was done at. A marker already here is newer.
            __restore_stage)
                name=$1; shift
                [ -f "$STATE/stages/$name" ] || echo "$*" > "$STATE/stages/$name"
                ;;
            __clear)
                for stage in "$@"; do
                    rm -f "$STATE/stages/$stage"
                    # The price of one plan must not be held against another's budget.
                    [ "$stage" != smoke ] || rm -f "$ROOT/results/smoke/refined_hours.txt"
                done
                ;;
            # A restore onto a machine that still has its markers: only this tree's may stand.
            __forget_other_trees)
                for file in "$STATE"/stages/*; do
                    [ -f "$file" ] || continue
                    read -r _ _ _ _ commit < "$file"
                    [ "$commit" = "$1" ] || rm -f "$file"
                done
                ;;
            __settings) printf '%s\n' "$1" > "$STATE/settings" ;;
            __get_settings) cat "$STATE/settings" 2>/dev/null || true ;;
            __init)
                mkdir -p "$ROOT/results" "$ROOT/traces"
                echo "$3" > "$STATE/synced_commit"
                # The first sync names the session; resumes keep that name and its directory here.
                [ -f "$STATE/session" ] || { echo "$1" > "$STATE/session"; echo "$2" > "$STATE/session_start"; }
                cat "$STATE/session"
                ;;
            *) echo "unknown subcommand: $command" >&2; exit 2 ;;
        esac
        exit $?
        ;;
esac

# ---------------------------------------------------------------------------------------
# The local side. Keep this to bash 3.2 and BSD tools: macOS ships them.
# ---------------------------------------------------------------------------------------

say() { printf '\n==> %s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
usage_error() { printf 'ERROR: %s\nSee %s --help\n' "$*" "$0" >&2; exit 2; }
fail() { local code=$1; shift; printf '\nERROR: %s\n' "$*" >&2; exit "$code"; }
hms() { printf '%dh %02dm %02ds' $(($1 / 3600)) $(($1 % 3600 / 60)) $(($1 % 60)); }
q() { printf "'%s'" "$1"; }
in_list() { case " $2 " in *" $1 "*) return 0 ;; esac; return 1; }

stage_exit_code() {
    case "$1" in
        preflight) echo 10 ;; sync) echo 11 ;; setup) echo 12 ;; gpu-tests) echo 13 ;;
        sanitize) echo 14 ;; baselines) echo 15 ;; traces) echo 16 ;; reproduce) echo 17 ;;
        fetch) echo 18 ;; smoke) echo 19 ;; budget) echo 20 ;; *) echo 1 ;;
    esac
}

HOST="" PORT="" KEY="" ONLY_STAGE="" FROM_STAGE="" STAGE_LIST="" HOURLY_USD="" EXTRA_SSH=""
DRY_RUN=0 STATUS=0 FETCH_ONLY=0 KEEP_GOING=0 SECRET_SCAN=1 PLAN_NAME="" MAX_HOURS="" SAVED_PLAN="" CONTINUE_OVER_BUDGET=0
REMOTE_DIR=layover-run FETCH_EVERY=600 FETCH_EVERY_RUNS=60 POLL=5 RECONNECT_TIMEOUT=1800 MIN_DISK_GB=200 MIN_GPU_MIB=22000 MIN_RAM_GB=64 ALLOW_IAM_ROLE=0
STOP_AFTER_SMOKE=0 RESTORE_FROM="" SPOT_NOTICE=0 RESUME_FLAGS=""
DETACH=0 AUTO_SHUTDOWN=1 MAX_SESSION_HOURS="" LEAVE_RUNNING=0 RUNNER_STARTED=0 PROVIDER="" REMOTE_DIR_GIVEN=0

need_value() { [ "$2" -ge 2 ] || usage_error "$1 needs a value"; }
need_int() { case "$2" in ''|*[!0-9]*) usage_error "$1 needs a whole number, got '$2'" ;; esac; }
number_ok() { case "$1" in ''|.|*[!0-9.]*|*.*.*) return 1 ;; esac; }
need_number() { number_ok "$2" || usage_error "$1 needs a number, got '$2'"; }
# It goes through ssh, a shell and make, which expands $ on its own, and every plan is a plain word.
plan_ok() { case "$1" in ''|-*|*[!A-Za-z0-9._-]*) return 1 ;; esac; }

while [ $# -gt 0 ]; do
    arg=$1 value=${2:-} before=$#
    case "$1" in
        -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
        -p) need_value "$1" $#; need_int "$1" "$2"; PORT=$2; shift 2 ;;
        -i) need_value "$1" $#; KEY=$2; shift 2 ;;
        --stage) need_value "$1" $#; ONLY_STAGE=$2; shift 2 ;;
        --stages) need_value "$1" $#; STAGE_LIST=$2; shift 2 ;;
        --from) need_value "$1" $#; FROM_STAGE=$2; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --status) STATUS=1; shift ;;
        --fetch-only) FETCH_ONLY=1; shift ;;
        --continue-on-failure) KEEP_GOING=1; shift ;;
        --plan) need_value "$1" $#; plan_ok "$2" || usage_error "--plan needs a plain name, got '$2'"; PLAN_NAME=$2; shift 2 ;;
        --max-hours) need_value "$1" $#; need_number "$1" "$2"; MAX_HOURS=$2; shift 2 ;;
        --continue-over-budget) CONTINUE_OVER_BUDGET=1; shift ;;
        --stop-after-smoke) STOP_AFTER_SMOKE=1; shift ;;
        --detach) DETACH=1; shift ;;
        --no-auto-shutdown) AUTO_SHUTDOWN=0; shift ;;
        --max-session-hours) need_value "$1" $#; need_number "$1" "$2"; MAX_SESSION_HOURS=$2; shift 2 ;;
        --leave-running) LEAVE_RUNNING=1; shift ;;
        --runpod) PROVIDER=runpod; shift ;;
        --spot-notice) SPOT_NOTICE=1; shift ;;
        --restore-from) need_value "$1" $#; RESTORE_FROM=$2; shift 2 ;;
        --no-secret-scan) SECRET_SCAN=0; shift ;;
        --allow-iam-role) ALLOW_IAM_ROLE=1; shift ;;
        --hourly-usd) need_value "$1" $#; need_number "$1" "$2"; HOURLY_USD=$2; shift 2 ;;
        --remote-dir) need_value "$1" $#; REMOTE_DIR=$2; REMOTE_DIR_GIVEN=1; shift 2 ;;
        --ssh-option) need_value "$1" $#; EXTRA_SSH="$EXTRA_SSH
$2"; shift 2 ;;
        --fetch-every) need_value "$1" $#; need_int "$1" "$2"; FETCH_EVERY=$2; shift 2 ;;
        --fetch-every-runs) need_value "$1" $#; need_int "$1" "$2"; FETCH_EVERY_RUNS=$2; shift 2 ;;
        --poll) need_value "$1" $#; need_number "$1" "$2"; POLL=$2; shift 2 ;;
        --reconnect-timeout) need_value "$1" $#; need_int "$1" "$2"; RECONNECT_TIMEOUT=$2; shift 2 ;;
        --min-disk-gb) need_value "$1" $#; need_int "$1" "$2"; MIN_DISK_GB=$2; shift 2 ;;
        --min-gpu-mib) need_value "$1" $#; need_int "$1" "$2"; MIN_GPU_MIB=$2; shift 2 ;;
        --min-ram-gb) need_value "$1" $#; need_int "$1" "$2"; MIN_RAM_GB=$2; shift 2 ;;
        -*) usage_error "unknown option: $1" ;;
        *) [ -z "$HOST" ] || usage_error "more than one host: $HOST and $1"; HOST=$1; shift ;;
    esac
    # What a resume on another machine needs of this command line.
    case "$arg" in
        --restore-from|--stage|--stages|--from|--stop-after-smoke|--status|--fetch-only|--dry-run|--leave-running) ;;
        -*)
            RESUME_FLAGS="$RESUME_FLAGS $(printf '%q' "$arg")"
            [ $((before - $#)) -lt 2 ] || RESUME_FLAGS="$RESUME_FLAGS $(printf '%q' "$value")"
            ;;
    esac
done

[ -n "$HOST" ] || usage_error "no host given"
case "$HOST" in
    -*|*[!A-Za-z0-9._@%-]*) usage_error "unusable host: $HOST" ;;
esac
if [ -n "$KEY" ]; then
    [ -r "$KEY" ] && [ -f "$KEY" ] || usage_error "cannot read key file: $KEY"
    case "$KEY" in
        *\'*) usage_error "key path contains a single quote: $KEY" ;;
        /*) ;;
        *) KEY="$PWD/$KEY" ;;
    esac
fi
[ -z "$ONLY_STAGE" ] || in_list "$ONLY_STAGE" "preflight sync $STAGES" ||
    usage_error "no stage called '$ONLY_STAGE'; stages: preflight sync $STAGES"
[ -z "$FROM_STAGE" ] || in_list "$FROM_STAGE" "$STAGES" ||
    usage_error "no stage called '$FROM_STAGE'; stages: $STAGES"
[ -z "$ONLY_STAGE" ] || [ -z "$FROM_STAGE" ] || usage_error "--stage and --from exclude each other"
[ -z "$STAGE_LIST" ] || [ -z "$ONLY_STAGE$FROM_STAGE" ] || usage_error "--stages excludes --stage and --from"
for stage in $STAGE_LIST; do
    in_list "$stage" "$STAGES" || usage_error "no stage called '$stage'; stages: $STAGES"
done
[ $((STATUS + FETCH_ONLY)) -le 1 ] || usage_error "--status and --fetch-only exclude each other"
if [ "$STATUS" = 1 ] || [ "$FETCH_ONLY" = 1 ]; then
    [ -z "$ONLY_STAGE$FROM_STAGE" ] || usage_error "--status and --fetch-only cannot take --stage or --from"
    [ "$STOP_AFTER_SMOKE" = 0 ] || usage_error "--status and --fetch-only cannot take --stop-after-smoke"
    [ -z "$RESTORE_FROM" ] || usage_error "--status and --fetch-only cannot take --restore-from"
fi
[ -z "$ONLY_STAGE" ] || [ "$STOP_AFTER_SMOKE" = 0 ] || usage_error "--stage and --stop-after-smoke exclude each other"
[ -z "$RESTORE_FROM" ] || [ -z "$ONLY_STAGE$FROM_STAGE" ] ||
    usage_error "--restore-from decides which stages run, so it cannot take --stage or --from"

# The session goes on in its own directory, so that a later restore finds everything in one place.
if [ -n "$RESTORE_FROM" ]; then
    RESTORE_FROM=${RESTORE_FROM%/}
    # The command this script prints is relative to the repository, wherever it is run from.
    [ -d "$RESTORE_FROM" ] || RESTORE_FROM="$ROOT/$RESTORE_FROM"
    [ -d "$RESTORE_FROM" ] || usage_error "no session directory at $RESTORE_FROM"
    RESTORE_FROM=$(cd "$RESTORE_FROM" && pwd -P)
    case "$RESTORE_FROM" in
        "$(cd "$ROOT" && pwd -P)"/results/gpu-*) ;;
        *) usage_error "--restore-from needs a session directory under results/, like results/gpu-<timestamp>, not $RESTORE_FROM" ;;
    esac
    [ -d "$RESTORE_FROM/stages" ] || usage_error "$RESTORE_FROM has no stages/, so it is not a session directory"
fi

# A pod's home is on the container disk, which a stop wipes.
[ "$PROVIDER" != runpod ] || [ "$REMOTE_DIR_GIVEN" = 1 ] || REMOTE_DIR=/workspace/layover-run
# rsync sends the path through the remote shell, which would split a space, and the mirror
# below must never be pointed at home or at something it does not own.
REMOTE_DIR=${REMOTE_DIR#\~/}
case "$REMOTE_DIR" in
    ''|.|/|./*|*/.|*/./*|..|../*|*/..|*/../*|*//*|*[!A-Za-z0-9._/-]*) usage_error "unusable remote directory: $REMOTE_DIR" ;;
esac

# accept-new takes a new machine's key, which is every rental, and still refuses a changed one.
# BatchMode turns a password prompt into an error rather than a hang. The alive probes make a
# dead connection fail within 30 seconds. SendEnv and the forwardings keep this machine's
# environment and credentials here.
SSH_CMD=(-T -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=10 -o ServerAliveCountMax=3
    -o StrictHostKeyChecking=accept-new -o ForwardAgent=no -o ClearAllForwardings=yes -o "SendEnv=-*" -o LogLevel=ERROR)
[ -z "$PORT" ] || SSH_CMD+=(-p "$PORT")
[ -z "$KEY" ] || SSH_CMD+=(-i "$KEY" -o IdentitiesOnly=yes)
while IFS= read -r option; do
    [ -n "$option" ] || continue
    case "$option" in
        *\'*) usage_error "--ssh-option contains a single quote: $option" ;;
    esac
    SSH_CMD+=(-o "$option")
done <<EOF
$EXTRA_SSH
EOF

# rsync re-parses this string, honouring quotes, so each piece goes in quoted.
RSYNC_SHELL=ssh
for piece in "${SSH_CMD[@]}"; do
    RSYNC_SHELL="$RSYNC_SHELL $(q "$piece")"
done
RSYNC=(rsync -rlptz --timeout=120 -e "$RSYNC_SHELL")

# Excludes also shield these remote paths from --delete, so .venv and results survive a re-sync.
# The first one covers .git, .venv, .notes and every other top-level dot directory;
# .git/info/exclude holds this clone's other private files.
SYNC_EXCLUDES=(
    --exclude='/.*/' --exclude=/results/ --exclude=/native/build/ --exclude=/build/
    --exclude=__pycache__/ --exclude=.hypothesis/ --exclude=.pytest_cache/ --exclude='*.egg-info/'
    --exclude='*.pyc' --exclude='*.o' --exclude='*.so' --exclude='*.dylib' --exclude=.DS_Store
    --exclude=.env --exclude='.env.*' --exclude=.netrc --exclude='*.pem' --exclude='*.key'
    --exclude='id_rsa*' --exclude='id_ecdsa*' --exclude='id_ed25519*'
    --exclude-from="$ROOT/.git/info/exclude"
)

rssh() { ssh "${SSH_CMD[@]}" "$HOST" "$@" < /dev/null; }
rrun() { rssh "cd $(q "$REMOTE_DIR") && exec bash scripts/remote_gpu.sh $*"; }

# Which machine stages this run asks for. Done ones are skipped unless named.
PLAN=$STAGES
if [ -n "$ONLY_STAGE" ]; then
    in_list "$ONLY_STAGE" "$STAGES" && PLAN=$ONLY_STAGE || PLAN=""
elif [ -n "$FROM_STAGE" ]; then
    PLAN=$(printf '%s\n' $STAGES | sed -n "/^$FROM_STAGE\$/,\$p" | tr '\n' ' ')
    PLAN=${PLAN% }
elif [ -n "$STAGE_LIST" ]; then
    PLAN=""
    for stage in $STAGES; do
        in_list "$stage" "$STAGE_LIST" && PLAN="$PLAN $stage"
    done
    PLAN=${PLAN# }
fi
if [ "$STOP_AFTER_SMOKE" = 1 ]; then
    in_list smoke "$PLAN" || usage_error "--stop-after-smoke: this plan has no smoke stage: $PLAN"
    PLAN=$(printf '%s\n' $PLAN | sed -n '1,/^smoke$/p' | tr '\n' ' ')
    PLAN=${PLAN% }
fi

# What the session already holds fills in what this command line leaves out. Per-invocation answers,
# the IAM role and going over budget, are asked for again: a new machine or a new estimate is new.
merge_settings() {
    local key value
    while read -r key value; do
        case "$key" in
            plan)
                plan_ok "$value" || fail 1 "the session's saved plan is not a plain name: $value"
                SAVED_PLAN=$value
                [ -n "$PLAN_NAME" ] || PLAN_NAME=$value
                ;;
            max_hours)
                number_ok "$value" || fail 1 "the session's saved --max-hours is not a number: $value"
                [ -n "$MAX_HOURS" ] || MAX_HOURS=$value
                ;;
        esac
    done <<EOF
$1
EOF
}

# Long enough for the plan, short enough that a hang does not run for days.
deadman_hours() {
    if [ -n "$MAX_SESSION_HOURS" ]; then echo "$MAX_SESSION_HOURS"
    elif [ "$STOP_AFTER_SMOKE" = 1 ]; then echo 2
    elif [ -n "$MAX_HOURS" ]; then awk -v h="$MAX_HOURS" 'BEGIN { print h + 6 }'
    else echo 48
    fi
}

settings_text() {
    [ -z "$PLAN_NAME" ] || echo "plan $PLAN_NAME"
    [ -z "$MAX_HOURS" ] || echo "max_hours $MAX_HOURS"
    [ "$ALLOW_IAM_ROLE" = 0 ] || echo "allow_iam_role 1"
    [ "$CONTINUE_OVER_BUDGET" = 0 ] || echo "continue_over_budget 1"
    echo "auto_shutdown $AUTO_SHUTDOWN"
    echo "deadman_hours $(deadman_hours)"
    [ -z "$PROVIDER" ] || echo "provider $PROVIDER"
    return 0
}

[ -z "$RESTORE_FROM" ] || merge_settings "$(cat "$RESTORE_FROM/settings" 2>/dev/null)"

if [ "$DRY_RUN" = 1 ]; then
    echo "Dry run: nothing is contacted or changed."
    echo "host:          $HOST"
    echo "run directory: $REMOTE_DIR"
    if [ "$STATUS" = 1 ]; then
        echo "would only print the session's status"
    elif [ "$FETCH_ONLY" = 1 ]; then
        echo "would only copy results back"
    else
        printf 'preflight   this machine: tools, make targets, secret scan. remote: ssh, rsync, x86_64, GPU, %s, %s GB free, %s GiB RAM, no IAM role\n' \
            "$([ -n "$PLAN" ] && ! in_list traces "$PLAN" && echo 'no Docker needed' || echo 'Docker runs a container')" "$MIN_DISK_GB" "$MIN_RAM_GB"
        [ "$PROVIDER" != runpod ] || echo "runpod      the pod stops itself with runpodctl; run directory and caches on the volume disk"
        if [ "$ONLY_STAGE" != preflight ]; then
            printf 'sync       '; printf ' %q' "${RSYNC[@]}" --delete --exclude=/traces/ "${SYNC_EXCLUDES[@]}" "$ROOT/" "$HOST:$REMOTE_DIR/"; echo
            printf 'sync-traces'; printf ' %q' "${RSYNC[@]}" --update "${SYNC_EXCLUDES[@]}" "$ROOT/traces/" "$HOST:$REMOTE_DIR/traces/"; echo
            printf 'sync-calib '; printf ' %q' "${RSYNC[@]}" --update "${SYNC_EXCLUDES[@]}" "$ROOT/results/calibration" "$HOST:$REMOTE_DIR/results/"; echo
            if [ -n "$RESTORE_FROM" ]; then
                for part in results traces; do
                    printf 'restore    '; printf ' %q' "${RSYNC[@]}" --update --exclude='*.tmp' "$RESTORE_FROM/$part/" "$HOST:$REMOTE_DIR/$part/"; echo
                done
                echo "restore     setup and baselines rerun; stages done at this commit are skipped; results come back to $RESTORE_FROM"
            fi
        fi
        for stage in $PLAN; do
            printf '%-11s detached on the machine: %s\n' "$stage" "$(stage_command "$stage")"
        done
        echo "stage env   MATRIX_ARGS=[$(matrix_args "$PLAN_NAME" "$MAX_HOURS")] SMOKE_PLAN=[$PLAN_NAME] TRACES_ARGS=[$([ "$ALLOW_IAM_ROLE" = 0 ] || echo --allow-iam-role)]"
        if [ -n "$MAX_HOURS" ] && [ "$CONTINUE_OVER_BUDGET" = 0 ]; then
            echo "budget      stops after smoke if its low estimate is over $MAX_HOURS hours"
        fi
        echo "fetch       $HOST:$REMOTE_DIR/results/ and traces/ to results/gpu-<timestamp>/"
    fi
    exit 0
fi

command -v ssh >/dev/null 2>&1 || fail 1 "ssh is not installed on this machine"
command -v rsync >/dev/null 2>&1 || fail 1 "rsync is not installed on this machine"

SESSION="" SESSION_DIR="" COMMIT="" FETCH_FAILED=0 RUNNER_RUNNING=0 INSTANCE_GONE=0 NOTICE_SEEN=0
SESSION_START=$(date +%s)
RESTORE_SESSION=""
if [ -n "$RESTORE_FROM" ]; then
    RESTORE_SESSION=${RESTORE_FROM##*/gpu-}
fi

on_exit() {
    if [ "$INSTANCE_GONE" = 1 ]; then
        cat <<EOF

================================================================================
  The instance looks gone. If it only lost its network it is still billing, so
  check your provider's console.
================================================================================
EOF
        return
    fi
    # Only a run on the machine shuts it down: a failed preflight or sync leaves it idle.
    if [ "$AUTO_SHUTDOWN" = 1 ] && [ "$RUNNER_STARTED" = 1 ]; then
        cat <<EOF

================================================================================
  The machine shuts itself down when its run ends, and at the latest $(deadman_hours) hours
  after it started. Re-run to attach, --status to check, --fetch-only to copy results.
  $([ "$PROVIDER" = runpod ] && echo 'Delete the pod once the results are here: stopped, it still bills for its disk.' || echo "Terminate the instance in your provider's console once the results are here.")
================================================================================
EOF
        return
    fi
    cat <<EOF

================================================================================
  The instance is still running, and still billing$([ "$AUTO_SHUTDOWN" = 1 ] && echo ': no run started on it, so nothing' || echo ' (--no-auto-shutdown)').
  $([ "$AUTO_SHUTDOWN" = 1 ] && echo 'will stop it. Fix the problem and re-run, or stop it in your provider'"'"'s console.' || echo 'Re-run to attach or resume. Once results are back, terminate it in your provider'"'"'s console.')
================================================================================
EOF
}
trap on_exit EXIT
trap 'printf "\nStopped watching. The remote run, if any, carries on.\n" >&2; exit 130' INT TERM

# ---- preflight ------------------------------------------------------------------------

list_sync_files() {
    rsync --list-only -rl "${SYNC_EXCLUDES[@]}" "$1" |
        sed -n -E 's#^-.* [0-9]{4}/[0-9]{2}/[0-9]{2} [0-9:]{8} ##p'
}

# What this tree sends. A restore's results also go, but are not part of the tree: they grow while
# the code stands still.
tree_manifest() {
    list_sync_files "$ROOT/"
    [ ! -d "$ROOT/results/calibration" ] || list_sync_files "$ROOT/results/calibration/" | sed 's#^#results/calibration/#'
}

sync_manifest() {
    local part
    tree_manifest
    [ -n "$RESTORE_FROM" ] || return 0
    for part in results traces; do
        [ ! -d "$RESTORE_FROM/$part" ] || list_sync_files "$RESTORE_FROM/$part/" | sed "s#^#results/gpu-$RESTORE_SESSION/$part/#"
    done
}

# Look inside files, not at names. The patterns need a run of key characters, which is why this
# file does not match itself.
scan_for_secrets() {
    local hits
    hits=$(cd "$ROOT" && printf '%s\n' "$1" | tr '\n' '\0' | LC_ALL=C xargs -0 grep -IlE \
        -e '(^|[^A-Za-z0-9])sk-[A-Za-z0-9_-]{32,}' \
        -e 'AKIA[0-9A-Z]{16}' -e 'hf_[A-Za-z0-9]{30,}' -e 'gh[pousr]_[A-Za-z0-9]{36}' \
        -e 'BEGIN [A-Z ]*PRIVATE KEY' 2>/dev/null)
    if [ -n "$hits" ]; then
        printf 'These files look like they hold credentials and would be synced:\n%s\n' "$hits" >&2
        fail 10 "refusing to sync. Clean them, or pass --no-secret-scan if these are false alarms."
    fi
}

check_local_tree() {
    local manifest stage target
    [ -f "$ROOT/.git/info/exclude" ] ||
        fail 10 "no .git/info/exclude in $ROOT: it lists the private files that must not be synced"
    [ -f "$ROOT/scripts/setup_gpu.sh" ] || fail 10 "scripts/setup_gpu.sh is missing"
    [ -f "$ROOT/Makefile" ] || fail 10 "there is no Makefile in $ROOT"
    if in_list baselines "$PLAN"; then
        [ -f "$ROOT/scripts/setup_baselines.sh" ] || fail 10 "scripts/setup_baselines.sh is missing"
    fi
    if in_list traces "$PLAN" && [ ! -f "$ROOT/traces/prepared/manifest.json" ]; then
        fail 10 "traces/prepared/manifest.json is missing: run make traces-prepare here first, it is free"
    fi
    for stage in $PLAN; do
        case "$(stage_command "$stage")" in
            make\ *)
                target=$(stage_command "$stage"); target=${target#make }
                grep -Eq "^([^:#]* )?$target( [^:#]*)?:" "$ROOT/Makefile" ||
                    fail 10 "the Makefile has no $target target, so the $stage stage cannot run"
                # traces/ is a directory, so without this make says "up to date" and the stage "passes".
                if [ -e "$ROOT/$target" ] && ! grep -Eq "^\.PHONY:(.* )?$target( |\$)" "$ROOT/Makefile"; then
                    fail 10 "the Makefile's $target target is not .PHONY and $target exists here, so make would do nothing"
                fi
                ;;
        esac
    done
    manifest=$(sync_manifest) || fail 10 "could not list the files to sync"
    [ -n "$manifest" ] || fail 10 "the list of files to sync is empty"
    if [ "$SECRET_SCAN" = 1 ]; then
        scan_for_secrets "$manifest"
    else
        warn "--no-secret-scan: files were not checked for credentials"
    fi
    echo "files to sync: $(printf '%s\n' "$manifest" | wc -l | tr -d ' ')"
}

# Names and contents of what the sync sends, so that it changes with an untracked file too and not
# with the index or a file that stays here. Symbolic links travel as links, so they count by target.
tree_hash() {
    local names
    names=$(tree_manifest | LC_ALL=C sort) || return 1
    {
        printf '%s\n' "$names"
        printf '%s\n' "$names" | (cd "$ROOT" && git hash-object --no-filters --stdin-paths)
        rsync --list-only -rl "${SYNC_EXCLUDES[@]}" "$ROOT/" | sed -n -E 's#^l.* [0-9]{4}/[0-9]{2}/[0-9]{2} [0-9:]{8} ##p' | LC_ALL=C sort
    } | git hash-object --stdin | cut -c1-16
}

# What gets synced is the working tree, and .git is not synced, so remember what that was. A restore
# skips what was done at this same id, so two different trees must never share one.
describe_commit() {
    local commit hash
    if ! commit=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null); then
        warn "$ROOT is not a git checkout with commits; results cannot name a commit"
        hash=$(tree_hash) || return 1
        echo "no-commit-$hash"
    elif [ -n "$(git -C "$ROOT" --no-optional-locks status --porcelain 2>/dev/null)" ]; then
        warn "uncommitted changes: the working tree is synced, not commit ${commit:0:12}"
        hash=$(tree_hash) || return 1
        echo "$commit-dirty-$hash"
    else
        echo "$commit"
    fi
}

# Runs on the machine, fed through stdin, since nothing is synced yet.
preflight_script() {
    cat <<'REMOTE'
set -u
dir=$1 min_gb=$2 min_gpu_mib=$3 min_ram_gb=$4 allow_iam_role=$5 auto_shutdown=$6 need_docker=${7:-1} provider=${8:-}
status=0
problem() { echo "PROBLEM: $*"; status=1; }
gb_free() { df -Pk "$1" | awk 'NR == 2 { print int($4 / 1048576) }'; }

echo "machine: $(hostname), $(uname -sm), user $(id -un)"
[ "$(uname -s)" = Linux ] || problem "this machine runs $(uname -s), and CUDA needs Linux"
[ "$(uname -m)" = x86_64 ] || problem "this machine is $(uname -m); the task containers for trace re-execution are x86_64 images"
# Rented containers often lack it, and log in as root, so it can go in here before the sync needs it.
if ! command -v rsync >/dev/null 2>&1 && [ "$(id -u)" -eq 0 ] && command -v apt-get >/dev/null 2>&1; then
    echo "rsync: missing; installing it"
    { apt-get -o DPkg::Lock::Timeout=300 update && DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=300 install -y --no-install-recommends rsync; } > /dev/null 2>&1
fi
command -v rsync >/dev/null 2>&1 || problem "rsync is missing on the machine; install it (apt-get install rsync)"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    problem "nvidia-smi not found: no NVIDIA driver is installed. Pick an image with one, or see
https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/ubuntu.html"
elif ! gpu=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits 2>&1); then
    problem "nvidia-smi failed, so the driver is not working: $gpu"
elif [ -z "$gpu" ]; then
    problem "nvidia-smi lists no GPUs"
else
    echo "gpu: $gpu"
    mib=$(echo "$gpu" | head -n1 | cut -d, -f2 | tr -d ' ')
    driver=$(echo "$gpu" | head -n1 | cut -d, -f3 | tr -d ' ')
    [ "$mib" -ge "$min_gpu_mib" ] || problem "the GPU has ${mib} MiB and the benchmarks need ${min_gpu_mib} MiB"
    [ "${driver%%.*}" -ge 580 ] || problem "driver $driver is too old: any CUDA 13 runtime needs at least 580"
fi

# Trace re-execution runs the task containers, so this is as required as the GPU. Only a real
# container proves it: an installed client says nothing about a daemon.
container_hint="Trace re-execution needs a Docker daemon that can run containers. Rental offerings
that are themselves containers commonly cannot do that; rent a VM or bare metal instead."
docker_root=/var/lib
if [ "$need_docker" != 1 ]; then
    echo "docker: not needed, since the traces stage is not in this run"
elif command -v docker >/dev/null 2>&1; then
    limit=
    command -v timeout >/dev/null 2>&1 && limit="timeout 300"
    if out=$($limit docker run --rm hello-world 2>&1); then
        echo "docker: runs containers"
        docker_root=$(docker info --format '{{.DockerRootDir}}' 2>/dev/null)
    elif [ "$(id -u)" -ne 0 ] && sudo -n $limit docker run --rm hello-world > /dev/null 2>&1; then
        echo "docker: runs containers through sudo; setup adds $(id -un) to the docker group"
        docker_root=$(sudo -n docker info --format '{{.DockerRootDir}}' 2>/dev/null)
    else
        problem "docker run --rm hello-world failed: $(echo "$out" | tail -n 3)
$container_hint"
    fi
elif [ -f /.dockerenv ] || [ -f /run/.containerenv ]; then
    problem "docker is missing and this machine is itself a container. $container_hint"
elif ! command -v apt-get >/dev/null 2>&1 || { [ "$(id -u)" -ne 0 ] && ! sudo -n true 2>/dev/null; }; then
    problem "docker is missing, and installing it takes apt-get and root or passwordless sudo. $container_hint"
else
    echo "docker: missing on this plain VM; setup installs it first and checks it runs a container"
fi

# Layover's 32 GiB tier pins real memory on top of its host cache, up to 48 GiB in all, the
# crossover adds about 10 GiB, and the baselines are given matched CPU budgets. Running short
# shows up hours in, as a lost cell or the OOM killer.
# A container sees its host's RAM through getconf, and gets only its cgroup's limit.
cgroup_mib() {
    local root=${LAYOVER_CGROUP_ROOT:-/sys/fs/cgroup} raw
    raw=$(cat "$root/memory.max" 2>/dev/null || cat "$root/memory/memory.limit_in_bytes" 2>/dev/null) || return 1
    case "$raw" in ''|*[!0-9]*) return 1 ;; esac
    echo $((raw / 1048576))
}
if [ "$min_ram_gb" -gt 0 ]; then
    pages=$(getconf _PHYS_PAGES 2>/dev/null)
    page_size=$(getconf PAGESIZE 2>/dev/null)
    if [ -z "$pages" ] || [ -z "$page_size" ]; then
        problem "cannot read this machine's RAM with getconf; check it holds ${min_ram_gb} GiB, then pass --min-ram-gb 0"
    else
        ram_mib=$((pages / 1024 * page_size / 1024))
        limit_mib=$(cgroup_mib) || limit_mib=$ram_mib
        if [ "$limit_mib" -lt "$ram_mib" ]; then
            echo "ram: $((limit_mib / 1024)) GiB, this container's limit on a $((ram_mib / 1024)) GiB host"
            ram_mib=$limit_mib
        else
            echo "ram: $((ram_mib / 1024)) GiB"
        fi
        # The kernel keeps a few percent, so a machine sold as 64 GiB reports about 62.
        [ $((ram_mib * 100)) -ge $((min_ram_gb * 1024 * 95)) ] || problem "the machine has $((ram_mib / 1024)) GiB of RAM, and Layover's pinned memory, the crossover and the baselines' matched CPU budgets need ${min_ram_gb} GiB; rent a larger one, or lower this with --min-ram-gb"
    fi
fi

# Task containers run third-party commands as root on the default network, where they can reach
# the metadata service and so the role's credentials. Asked from here, the host, so that the
# hop limit does not hide the answer. A machine off AWS has nothing there, which is fine.
if [ "$allow_iam_role" = 1 ]; then
    echo "iam role: not checked (--allow-iam-role)"
elif ! command -v curl >/dev/null 2>&1; then
    problem "curl is missing, so whether this instance has an IAM role cannot be checked; install curl, or pass --allow-iam-role"
else
    imds=http://169.254.169.254/latest
    token=$(curl -fsS --noproxy '*' --connect-timeout 2 -m 4 -X PUT -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' "$imds/api/token" 2>/dev/null)
    if [ -z "$token" ]; then
        echo "iam role: no AWS metadata service answered"
    else
        answer=$(curl -sS --noproxy '*' --connect-timeout 2 -m 4 -w '\n%{http_code}' -H "X-aws-ec2-metadata-token: $token" "$imds/meta-data/iam/security-credentials/" 2>/dev/null)
        case "${answer##*$'\n'}" in
            404) echo "iam role: none" ;;
            200) problem "this instance has the IAM role ${answer%$'\n'*}, and the task containers run third-party commands that can reach its credentials. Detach the role, or pass --allow-iam-role if it can do no harm" ;;
            *) problem "the metadata service would not say whether this instance has an IAM role (got ${answer##*$'\n'}); check that it has none, then pass --allow-iam-role" ;;
        esac
    fi
fi

# Without this the machine cannot stop itself, and bills idle after the run. On RunPod the pod's
# own key may stop the pod but not read it (runpodctl 1.14 answers a read with Unauthorized), so
# nothing here can prove the stop short of making it: this checks that its pieces are in place.
if [ "$auto_shutdown" = 1 ] && [ "$provider" = runpod ]; then
    if [ -z "${RUNPOD_POD_ID:-}" ] && [ -r /etc/rp_environment ]; then
        # shellcheck disable=SC1091
        . /etc/rp_environment
    fi
    if [ -z "${RUNPOD_POD_ID:-}" ] && [ -r /proc/1/environ ]; then
        while IFS= read -r -d '' line; do
            case "$line" in RUNPOD_POD_ID=*|RUNPOD_API_KEY=*) export "${line?}" ;; esac
        done < /proc/1/environ
    fi
    if [ -z "${RUNPOD_POD_ID:-}" ]; then
        problem "--runpod, but RUNPOD_POD_ID is not set here, so this does not look like a RunPod pod and could not stop itself"
    elif ! command -v runpodctl >/dev/null 2>&1; then
        problem "runpodctl is missing, so the pod cannot stop itself; install it, or pass --no-auto-shutdown and stop the pod yourself"
    elif [ -z "${RUNPOD_API_KEY:-}" ]; then
        problem "RUNPOD_API_KEY is not set here, so runpodctl could not stop pod $RUNPOD_POD_ID; pass --no-auto-shutdown and stop the pod yourself"
    else
        echo "auto-stop: runpodctl $(runpodctl --version 2>/dev/null | head -n1 | awk '{ print $NF }') with this pod's key, for pod $RUNPOD_POD_ID (the stop is not tried here)"
    fi
elif [ "$auto_shutdown" = 1 ]; then
    if command -v shutdown >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
        echo "auto-shutdown: possible (passwordless sudo)"
    else
        problem "auto-shutdown needs shutdown and passwordless sudo on the machine; fix that, or pass --no-auto-shutdown and stop it yourself"
    fi
fi

# On a resume, setup and the task images have already spent most of that disk.
if [ "$(cut -d' ' -f1 "$dir/.layover-run/stages/setup" 2>/dev/null)" = "done" ]; then
    min_gb=$((min_gb < 5 ? min_gb : 5))
fi
# The run directory's filesystem is that of the nearest directory on its way that exists.
here=$dir
while [ ! -d "$here" ]; do here=$(dirname "$here"); done
if [ "$need_docker" = 1 ]; then
    [ -d "$docker_root" ] || docker_root=/
    free_gb=$(gb_free "$docker_root")
    echo "free disk: ${free_gb} GB at $docker_root, $(gb_free "$here") GB at $here"
    [ "$free_gb" -ge "$min_gb" ] || problem "only ${free_gb} GB free at $docker_root, and the task images and everything else need ${min_gb} GB; attach a larger disk, or lower this with --min-disk-gb"
else
    echo "free disk: $(gb_free "$here") GB at $here"
fi
[ "$(gb_free "$here")" -ge "$((min_gb < 60 ? min_gb : 60))" ] || problem "only $(gb_free "$here") GB free at $here, and the toolkit, torch, weights and baselines need about 60 GB"
# The CUDA toolkit goes to /usr/local, which on a pod is the container disk and not the volume.
if [ "$provider" = runpod ] && [ "$(gb_free /)" -lt "$((min_gb < 15 ? min_gb : 15))" ]; then
    problem "only $(gb_free /) GB free on the container disk, and the CUDA toolkit needs about 15 GB; create the pod with a larger container disk"
fi

# sync mirrors with --delete, so it only ever touches a directory this tool made.
if [ -d "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ] && [ ! -d "$dir/.layover-run" ]; then
    problem "$dir already exists with other files in it. Refusing to mirror into it; use --remote-dir"
fi
if [ "$status" -eq 0 ]; then
    mkdir -p "$dir/.layover-run/logs" "$dir/.layover-run/stages" || problem "cannot create $dir"
    echo "$gpu" > "$dir/.layover-run/logs/gpu.txt"
    { uname -a; cat /etc/os-release 2>/dev/null; command -v lscpu >/dev/null && lscpu; nvidia-smi -q 2>&1; docker version 2>&1; } > "$dir/.layover-run/logs/machine.txt"
fi
exit "$status"
REMOTE
}

resume_command() {
    printf '%s user@NEW-HOST%s --restore-from results/gpu-%s' "$0" "$RESUME_FLAGS" "$1"
}

# The newest session here that a machine went away from, if any: setup done, reproduce not.
latest_unfinished_session() {
    local dir status found=""
    for dir in "$ROOT"/results/gpu-*; do
        [ "$(cut -d' ' -f1 "$dir/stages/setup" 2>/dev/null)" = "done" ] || continue
        status=$(cut -d' ' -f1 "$dir/stages/reproduce" 2>/dev/null)
        [ "$status" != "done" ] || continue
        [ "${dir##*/gpu-}" = "$SESSION" ] || found=${dir##*/gpu-}
    done
    echo "$found"
}

hint_resume_from_latest() {
    local name
    name=$(latest_unfinished_session)
    [ -n "$name" ] || return 0
    printf '\nIf this machine was interrupted, resume on a new one with:\n    %s\n' "$(resume_command "$name")" >&2
}

# Fetching on failure matters only on a resume: a first run has nothing over there yet.
stage_failed() {
    printf '\nERROR: %s\n' "$2" >&2
    if load_session 2>/dev/null; then
        final_fetch || warn "could not copy results back; the machine still holds them. Try --fetch-only."
        write_summary
    fi
    exit "$(stage_exit_code "$1")"
}

run_preflight() {
    local output rc
    say "preflight: this machine"
    check_local_tree
    COMMIT=$(describe_commit) || fail 10 "could not hash the files to sync"
    echo "commit: $COMMIT"
    say "preflight: $HOST"
    # Preflight alone checks for everything a session may run.
    need_docker=1
    [ -z "$PLAN" ] || in_list traces "$PLAN" || need_docker=0
    output=$(preflight_script | ssh "${SSH_CMD[@]}" "$HOST" "bash -s -- $(q "$REMOTE_DIR") $MIN_DISK_GB $MIN_GPU_MIB $MIN_RAM_GB $ALLOW_IAM_ROLE $AUTO_SHUTDOWN $need_docker ${PROVIDER:-none}" 2>&1)
    rc=$?
    printf '%s\n' "$output"
    if [ "$rc" -eq 255 ]; then
        hint_resume_from_latest
        fail 10 "cannot reach $HOST over ssh. Login must work with a key and no prompt; the ssh error is above."
    fi
    [ "$rc" -eq 0 ] || stage_failed preflight "the machine is not ready; see the PROBLEM lines above"
}

# ---- sync -----------------------------------------------------------------------------

# traces/ goes on its own, without --delete and keeping what is newer there, because the machine
# adds to it: a re-sync on resume must not take back what the traces stage wrote. The calibration
# goes the same way, the one part of results/ that travels: it takes long on one core and needs no
# GPU, so it is computed here, and a machine that computed its own keeps it through a re-sync.
run_sync() {
    local rc=0
    say "sync: $ROOT to $HOST:$REMOTE_DIR"
    "${RSYNC[@]}" --delete --exclude=/traces/ "${SYNC_EXCLUDES[@]}" "$ROOT/" "$HOST:$REMOTE_DIR/" || rc=$?
    if [ "$rc" -eq 0 ] && [ -d "$ROOT/traces" ]; then
        "${RSYNC[@]}" --update "${SYNC_EXCLUDES[@]}" "$ROOT/traces/" "$HOST:$REMOTE_DIR/traces/" || rc=$?
    fi
    if [ "$rc" -eq 0 ] && [ -d "$ROOT/results/calibration" ]; then
        "${RSYNC[@]}" --update "${SYNC_EXCLUDES[@]}" "$ROOT/results/calibration" "$HOST:$REMOTE_DIR/results/" || rc=$?
    fi
    [ "$rc" -eq 0 ] || stage_failed sync "rsync failed with exit $rc; re-run to continue"
}

use_session() {
    SESSION=$1
    SESSION_DIR="$ROOT/results/gpu-$SESSION"
    mkdir -p "$SESSION_DIR/stages"
}

load_session() {
    local name
    name=$(rrun __session) && [ -n "$name" ] && use_session "$name"
}

# The machine's copy for a resume, and for a new machine the copy that came back from the old one.
load_settings() {
    local saved
    saved=$(rrun __get_settings 2>/dev/null)
    [ -n "$saved" ] || saved=$(cat "$SESSION_DIR/settings" 2>/dev/null)
    merge_settings "$saved"
}

# ---- going on from a machine that went away ----------------------------------------------

# Another GPU or driver changes every number, and run_matrix.py would refuse it, but only after
# setup, the baselines and everything else before the matrix. Run after preflight, which records
# this machine's.
check_restore_gpu() {
    local before now
    before=$(head -n1 "$RESTORE_FROM/logs/gpu.txt" 2>/dev/null) || return 0
    [ -n "$before" ] || return 0
    now=$(rssh "head -n1 $(q "$REMOTE_DIR")/.layover-run/logs/gpu.txt" 2>/dev/null) || stage_failed preflight "could not read this machine's GPU"
    [ "$before" = "$now" ] ||
        fail 10 "$RESTORE_FROM was measured on '$before' and this machine has '$now'. Their results must not share a table; rent the same GPU and image, or start a new session."
}

# Push what the lost machine made, older than nothing here, and mark what need not run again.
restore_session() {
    local part stage status start end rc commit
    # The fetches below replace these with the new machine's, so the old ones step aside.
    [ ! -d "$SESSION_DIR/logs" ] || mv "$SESSION_DIR/logs" "$SESSION_DIR/logs-before-$(date +%Y%m%d-%H%M%S)"
    say "restore: $SESSION_DIR to $HOST:$REMOTE_DIR"
    rrun __forget_other_trees "$(q "$COMMIT")" > /dev/null || stage_failed sync "could not check the stage markers on the machine; re-run to continue"
    for part in results traces; do
        [ -d "$SESSION_DIR/$part" ] || continue
        "${RSYNC[@]}" --update --exclude='*.tmp' "$SESSION_DIR/$part/" "$HOST:$REMOTE_DIR/$part/" ||
            stage_failed sync "could not push $part/ to the new machine; re-run to continue"
    done
    # Setup and baselines live on the machine, so they run again. A stage done at another commit
    # runs again too: the Makefile's guards keep its finished outputs.
    for stage in $STAGES; do
        case "$stage" in setup|baselines) continue ;; esac
        [ -f "$SESSION_DIR/stages/$stage" ] || continue
        read -r status start end rc commit < "$SESSION_DIR/stages/$stage"
        if [ "$status" = "done" ] && [ "$commit" = "$COMMIT" ]; then
            rrun __restore_stage "$stage" "$status" "$start" "$end" "$rc" "$commit" > /dev/null ||
                stage_failed sync "could not mark $stage done on the new machine; re-run to continue"
            echo "$stage: done at this commit, skipped"
        fi
    done
}

# ---- watching the detached run ----------------------------------------------------------

probe_alive() {
    [ "$(rssh "if [ -f $(q "$REMOTE_DIR")/scripts/remote_gpu.sh ]; then cd $(q "$REMOTE_DIR") && bash scripts/remote_gpu.sh __alive; else echo 0; fi" 2>/dev/null)" = 1 ]
}

# Results, what the traces stage wrote, and the logs and stage markers. Nothing here is removed.
do_fetch() {
    local rc from to
    for pair in "results/ results/" "traces/ traces/" ".layover-run/ ./"; do
        from=${pair%% *} to=${pair##* }
        rc=0
        mkdir -p "$SESSION_DIR/$to"
        "${RSYNC[@]}" --exclude=runner.pid --exclude='.*.tmp' "$HOST:$REMOTE_DIR/$from" "$SESSION_DIR/$to" || rc=$?
        # 24: a file vanished mid-transfer, which a running benchmark may do.
        [ "$rc" -eq 0 ] || [ "$rc" -eq 24 ] || return "$rc"
    done
}

# Everything fetched so far is already in the session directory, and nothing here removes it.
instance_gone() {
    INSTANCE_GONE=1
    write_summary
    printf '\nThe results fetched so far are safe in %s.\n' "$SESSION_DIR"
    printf 'On a replacement instance, go on with:\n    %s\n' "$(resume_command "$SESSION")"
    fail 1 "no connection to $HOST for $1s. If the machine is up after all, the run is still going: re-run this command to attach."
}

# Runs stages for hours, so an interruption costs more the longer since the last copy.
fetch_interval() {
    if [ "$NOTICE_SEEN" = 1 ]; then
        echo 10
    else
        case "$1" in
            traces|reproduce) echo $((FETCH_EVERY < FETCH_EVERY_RUNS ? FETCH_EVERY : FETCH_EVERY_RUNS)) ;;
            *) echo "$FETCH_EVERY" ;;
        esac
    fi
}

# The refined plan, the moment smoke ends, even if a failure stops the session right after.
report_smoke() {
    local file="$SESSION_DIR/results/smoke/refined_hours.txt" low high
    do_fetch || warn "could not copy smoke's results back; will try again"
    if [ ! -f "$file" ]; then
        say "smoke ended with no refined hours: see $SESSION_DIR/results/smoke/smoke.json"
        return 0
    fi
    read -r low high < "$file"
    say "smoke's refined plan: $low to $high hours (an estimate from rates measured on this GPU)"
    tail -n +2 "$file"
    if [ -n "$HOURLY_USD" ]; then
        awk -v low="$low" -v high="$high" -v r="$HOURLY_USD" 'BEGIN { printf "At $%s/hour: $%.2f to $%.2f for the plan.\n", r, low * r, high * r }'
    fi
    if [ -n "$MAX_HOURS" ] && hours_over "$low" "$MAX_HOURS"; then
        echo "That low estimate is over --max-hours $MAX_HOURS."
    fi
}

watch_runner() {
    local offset=-1 alive size smoke current notice last_fetch now down_since=0 poll_file="$SESSION_DIR/.poll" limit
    local reported_file="$SESSION_DIR/.smoke_reported"
    last_fetch=$(date +%s)
    while :; do
        now=$(date +%s)
        # A cut reply is thrown away and read again, since the offset only moves on success.
        if ! rrun __poll "$offset" "$SPOT_NOTICE" > "$poll_file" 2>/dev/null; then
            limit=$RECONNECT_TIMEOUT
            # After a notice this is no network blip, so waiting long only delays the resume.
            [ "$NOTICE_SEEN" = 0 ] || [ "$limit" -le 300 ] || limit=300
            if [ "$down_since" -eq 0 ]; then
                down_since=$now
                printf '\n[lost the connection. The remote run carries on if the machine is up. Retrying for up to %ss.]\n' "$limit"
                printf '[if the instance was interrupted, this resumes on a new one: %s]\n' "$(resume_command "$SESSION")"
            elif [ $((now - down_since)) -ge "$limit" ]; then
                instance_gone $((now - down_since))
            fi
            sleep "$POLL"
            continue
        fi
        if [ "$down_since" -ne 0 ]; then
            printf '[reconnected after %ss]\n' $((now - down_since))
            down_since=0
        fi
        read -r alive size smoke current notice < "$poll_file"
        tail -n +2 "$poll_file"
        offset=$size
        if [ "$notice" = notice ] && [ "$NOTICE_SEEN" = 0 ]; then
            NOTICE_SEEN=1
            printf '\n[SPOT INTERRUPTION NOTICE: this instance is going away. Copying results now, then every 10 s.]\n'
            printf '[afterwards, this resumes on a new one: %s]\n' "$(resume_command "$SESSION")"
            do_fetch || printf '[could not copy partial results; will try again]\n'
            last_fetch=$now
        fi
        case "$smoke" in
            done:*|failed:*)
                if [ "$smoke" != "$(cat "$reported_file" 2>/dev/null)" ]; then
                    report_smoke
                    printf '%s\n' "$smoke" > "$reported_file"
                    last_fetch=$now
                fi
                ;;
        esac
        [ "$alive" = 1 ] || break
        if [ $((now - last_fetch)) -ge "$(fetch_interval "$current")" ]; then
            do_fetch && printf '[partial results copied to %s]\n' "$SESSION_DIR" ||
                printf '[could not copy partial results; will try again]\n'
            last_fetch=$now
        fi
        sleep "$POLL"
    done
}

# ---- end of the session -----------------------------------------------------------------

final_fetch() {
    local start attempt
    say "fetch: results, traces and logs to $SESSION_DIR"
    start=$(date +%s)
    # This copy is the one that matters, and a connection that just dropped may be back shortly.
    for attempt in 1 2 3; do
        if do_fetch; then
            printf 'done %s %s 0 -\n' "$start" "$(date +%s)" > "$SESSION_DIR/stages/fetch"
            probe_alive && RUNNER_RUNNING=1
            FETCH_FAILED=0
            return 0
        fi
        warn "fetch attempt $attempt failed"
        [ "$attempt" = 3 ] || sleep "$POLL"
    done
    printf 'failed %s %s 1 -\n' "$start" "$(date +%s)" > "$SESSION_DIR/stages/fetch"
    FETCH_FAILED=1
    return 1
}

json_string() { printf '"%s"' "$(printf '%s' "$1" | tr -d '\000-\037' | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g')"; }

# Prints the time table and writes it as JSON, for refining the run plan. Reads the copies in
# the session directory, so it also works after the machine is gone.
write_summary() {
    local dir=$SESSION_DIR stage status start end rc commit seconds now wall sum=0 latest=0 first=1
    local session_start rate=null cost=null rows=""
    now=$(date +%s)
    session_start=$(cat "$dir/session_start" 2>/dev/null || echo "$SESSION_START")
    printf '\n%-18s %-12s %14s  %s\n' stage status time commit
    for stage in preflight sync $STAGES fetch; do
        status=pending start=- end=- rc=- commit=-
        [ -f "$dir/stages/$stage" ] && read -r status start end rc commit < "$dir/stages/$stage"
        [ "$status" != running ] || [ "$RUNNER_RUNNING" = 1 ] || status=interrupted
        seconds=null
        if [ "$start" != - ]; then
            [ "$end" != - ] && seconds=$((end - start)) sum=$((sum + end - start)) || seconds=$((now - start))
            latest=$(( ${end/-/$now} > latest ? ${end/-/$now} : latest ))
        fi
        printf '%-18s %-12s %14s  %s\n' "$stage" "$status" "$([ "$seconds" = null ] && echo - || hms "$seconds")" "${commit:0:12}"
        [ "$first" = 1 ] || rows="$rows,
"
        first=0
        rows="$rows    {\"name\": \"$stage\", \"status\": \"$status\", \"start_epoch\": ${start/#-/null}, \"end_epoch\": ${end/#-/null}, \"seconds\": $seconds, \"exit_code\": ${rc/#-/null}, \"commit\": \"$commit\"}"
    done
    wall=$(( latest > session_start ? latest - session_start : 0 ))
    if [ -n "$HOURLY_USD" ]; then
        rate=$HOURLY_USD
        cost=$(awk -v s="$wall" -v r="$HOURLY_USD" 'BEGIN { printf "%.2f", s / 3600 * r }')
    fi
    cat > "$dir/stage_times.json" <<EOF
{
  "session": $(json_string "$SESSION"),
  "commit": $(json_string "$(cat "$dir/synced_commit" 2>/dev/null || echo unknown)"),
  "gpu": $(json_string "$(head -n1 "$dir/logs/gpu.txt" 2>/dev/null)"),
  "wall_clock_s": $wall,
  "stage_sum_s": $sum,
  "hourly_usd": $rate,
  "cost_usd": $cost,
  "stages": [
$rows
  ]
}
EOF
    printf '\nTotal wall-clock: %s (sum of stages: %s)\n' "$(hms "$wall")" "$(hms "$sum")"
    if [ -n "$HOURLY_USD" ]; then
        printf 'Cost: $%s at $%s/hour, from the first preflight. The meter started earlier, when the instance launched.\n' "$cost" "$HOURLY_USD"
        if [ -n "$RESTORE_SESSION" ]; then
            echo "This session spans instances, so that counts the time between them too, and each at this one rate."
        fi
    fi
    echo "Stage times: $dir/stage_times.json"
}

# Fetch, print the table and any failed logs, and exit with the first stage not done.
finish() {
    local stage status rc code=0 plan low="" max=""
    final_fetch || warn "could not copy results back; the machine still holds them. Try --fetch-only."
    write_summary
    plan=$(cat "$SESSION_DIR/plan" 2>/dev/null || true)
    [ ! -s "$SESSION_DIR/budget_stop" ] || read -r low max < "$SESSION_DIR/budget_stop"
    for stage in $plan; do
        status=pending rc=-
        [ -f "$SESSION_DIR/stages/$stage" ] && read -r status _ _ rc _ < "$SESSION_DIR/stages/$stage"
        [ "$status" != "done" ] || continue
        if [ "$code" = 0 ] && [ -n "$low" ] && [ "$status" = pending ]; then
            code=$(stage_exit_code budget)
        elif [ "$code" = 0 ]; then
            code=$(stage_exit_code "$stage")
            if [ "$status" != failed ] && [ "$RUNNER_RUNNING" = 0 ]; then
                printf '\nNot finished: %s is %s. Re-run this command to resume.\n' "$stage" "$status"
            fi
        fi
        if [ "$status" = failed ]; then
            printf '\nFAILED: %s, exit %s. Full log: %s\n' "$stage" "$rc" "$SESSION_DIR/logs/$stage.log"
            tail -n 15 "$SESSION_DIR/logs/$stage.log" 2>/dev/null | sed 's/^/    /'
        fi
    done
    [ "$FETCH_FAILED" = 0 ] || code=$(stage_exit_code fetch)
    [ "$FETCH_ONLY" = 0 ] || code=$((FETCH_FAILED ? code : 0))
    if [ -n "$low" ]; then
        printf '\nOVER BUDGET: smoke estimates at least %s hours for this plan and --max-hours is %s, so the session stopped before the long stages.\n' "$low" "$max"
        echo "To go on, run the same command with --continue-over-budget, a higher --max-hours, or a smaller --plan."
    fi
    echo "Results: $SESSION_DIR"
    if [ "$STOP_AFTER_SMOKE" = 1 ] && [ "$code" = 0 ]; then
        printf '\nStopped after smoke, as asked. To go on, run the same command without --stop-after-smoke%s.\n' "$([ "$AUTO_SHUTDOWN" = 1 ] && echo ', after starting the instance again: it shuts itself down now' || echo '; the instance is still billing')"
    fi
    exit "$code"
}

# After a --fetch-only whose copy came back whole, the machine that was started for it stops again.
final_fetch_then_stop() {
    local rc=$?
    if [ "$FETCH_FAILED" = 0 ] && rrun __stop_machine > /dev/null 2>&1; then
        echo "The machine shuts down in a minute. --leave-running keeps it up."
    else
        warn "the machine was not shut down; stop it yourself"
    fi
    on_exit
    exit "$rc"
}

# ---- what this invocation does ----------------------------------------------------------

if [ "$STATUS" = 1 ]; then
    out=$(rssh "if [ -f $(q "$REMOTE_DIR")/scripts/remote_gpu.sh ]; then cd $(q "$REMOTE_DIR") && bash scripts/remote_gpu.sh __status; else echo 'no run on this machine yet'; fi" 2>&1)
    rc=$?
    printf '%s\n' "$out"
    [ "$rc" -eq 0 ] || fail 1 "cannot reach $HOST (ssh exit $rc)"
    exit 0
fi

if [ "$FETCH_ONLY" = 1 ]; then
    load_session || fail 18 "no session in $REMOTE_DIR on $HOST: the machine is unreachable, or nothing was synced yet"
    # A machine started only to fetch from goes back down, unless its run is still going.
    if probe_alive; then
        RUNNER_STARTED=1
    elif [ "$AUTO_SHUTDOWN" = 1 ] && [ "$LEAVE_RUNNING" = 0 ]; then
        trap 'final_fetch_then_stop' EXIT
    fi
    finish
fi

if probe_alive; then
    # Never change code under a running stage.
    [ -z "$ONLY_STAGE$FROM_STAGE" ] ||
        fail 1 "a runner is going on the machine. Wait for it, or stop it (--status shows how), before starting other stages."
    echo "A runner is already going on the machine; attaching to it."
    RUNNER_STARTED=1
    [ "$STOP_AFTER_SMOKE" = 0 ] || warn "--stop-after-smoke applies only when a run starts; this one keeps its own plan"
    [ -z "$PLAN_NAME$MAX_HOURS" ] || warn "--plan and --max-hours apply only when a run starts; this one keeps its own"
    PLAN_NAME="" MAX_HOURS=""
    load_session || fail 1 "lost $HOST while attaching; re-run"
    load_settings
    [ -z "$RESTORE_SESSION" ] || [ "$SESSION" = "$RESTORE_SESSION" ] ||
        fail 1 "this machine is running session $SESSION, not $RESTORE_SESSION. Point --restore-from at it, or use another machine."
else
    preflight_start=$(date +%s)
    run_preflight
    preflight_end=$(date +%s)
    [ "$ONLY_STAGE" != preflight ] || exit 0

    [ -z "$RESTORE_FROM" ] || check_restore_gpu
    run_sync
    new_name=$(date +%Y%m%d-%H%M%S)
    if [ -n "$RESTORE_SESSION" ]; then
        new_name=$RESTORE_SESSION
        SESSION_START=$(cat "$RESTORE_FROM/session_start" 2>/dev/null || echo "$SESSION_START")
    fi
    name=$(rrun __init "$new_name" "$SESSION_START" "$COMMIT") && [ -n "$name" ] ||
        stage_failed sync "could not record the synced commit on the machine"
    [ -z "$RESTORE_SESSION" ] || [ "$name" = "$RESTORE_SESSION" ] ||
        fail 1 "this machine holds session $name, and restoring $RESTORE_SESSION onto it would mix them. Use another --remote-dir or another machine."
    use_session "$name"
    echo "session: $SESSION, results in $SESSION_DIR"
    load_settings
    if [ -n "$RESTORE_SESSION" ]; then
        restore_session
    else
        unfinished=$(latest_unfinished_session)
        [ -z "$unfinished" ] || warn "results/gpu-$unfinished never finished. If this machine replaces that one, stop and use --restore-from, or its results stay apart from these."
    fi
    # Here as well as there: an instance can vanish before anything has been fetched from it.
    settings_text > "$SESSION_DIR/settings"
    rrun __settings "$(q "$(settings_text)")" > /dev/null || stage_failed sync "could not record the session's settings on the machine"
    # Smoke priced the plan it was given.
    [ "$SAVED_PLAN" = "$PLAN_NAME" ] || rrun __clear smoke > /dev/null || fail 1 "could not clear the old estimate"
    rrun __mark preflight "done" "$preflight_start" "$preflight_end" 0 > /dev/null
    rrun __mark sync "done" "$preflight_end" "$(date +%s)" 0 > /dev/null
    [ "$ONLY_STAGE" != sync ] || exit 0

    # --stage and --from mean run it again, even if it is marked done.
    if [ -n "$ONLY_STAGE$FROM_STAGE" ]; then
        # shellcheck disable=SC2086
        rrun __clear $PLAN || fail 1 "could not clear the old stage markers"
    fi
    say "Starting on the machine: $PLAN"
    # shellcheck disable=SC2086
    rrun __start "$KEEP_GOING" $PLAN > /dev/null || fail 1 "could not start the remote run"
    RUNNER_STARTED=1
fi
if [ "$DETACH" = 1 ]; then
    say "Detached. The run goes on without this machine."
    echo "Check on it:     $0 $HOST$RESUME_FLAGS --status"
    echo "Copy results:    $0 $HOST$RESUME_FLAGS --fetch-only"
    exit 0
fi
watch_runner
finish
