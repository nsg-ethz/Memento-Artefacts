#! /bin/bash

resultdir="results"
mkdir -p "$resultdir"


# Repeat 22 times with different RNG to have plenty of data.
maxrun=21

# Parallelization
procs=${1:-1}  # 1 process per default


# Build modules once, then run without building.
waf build

function exprun {
    # First arg: rng run, secong arg: experiment name.
    # Run with some default settings
    cmd="NS_GLOBAL_VALUE=\"RngRun=$1\" waf --run-no-build \"trafficgen ${@:3} --prefix=$resultdir/$2_$1\""
    echo "STARTING $2($1):"
    echo "$cmd"
    output=$(eval "$cmd" 2>&1)
    output="OUTPUT $2($1):\n${output}\n"
    printf "$output"
}
export -f exprun   # make it visible for xargs
export resultdir   # same same

function run {  # First argument is experiment name, others are passed through.
    args=$@
    seq 0 $maxrun | xargs --max-procs $procs -n1 -I {} bash -c "exprun {} $args"
}


# Simulate different traffic mixes
# By default, for each workload we start 20 apps that send a combined
# 20 Mbps, and the link capacity is 30 Mbps (66% utilization on average).
run "w1" --w1=1 --w2=0 --w3=0 --congestion=0
run "w2" --w1=0 --w2=1 --w3=0 --congestion=0
run "w3" --w1=0 --w2=0 --w3=1 --congestion=0

# Also simulate congestion levels
# 100% average network utilization
run "w1_c100" --w1=1 --w2=0 --w3=0 --congestion=10Mbps
run "w2_c100" --w1=0 --w2=1 --w3=0 --congestion=10Mbps
run "w3_c100" --w1=0 --w2=0 --w3=1 --congestion=10Mbps

# 133% average network utilization
run "w1_c133" --w1=1 --w2=0 --w3=0 --congestion=20Mbps
run "w2_c133" --w1=0 --w2=1 --w3=0 --congestion=20Mbps
run "w3_c133" --w1=0 --w2=0 --w3=1 --congestion=20Mbps
