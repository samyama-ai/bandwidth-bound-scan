#!/usr/bin/env bash
# eval_aws.sh <candidate.c> — run the evaluation cascade on the AWS AVX-512 box (c6i, Ice Lake).
# Ships the candidate to the instance and runs evaluate.sh there with clang -march=native (AVX-512).
# Set BBS_AWS_IP to the instance public IP (ephemeral). Key: ~/.ssh/pem/graph.pem, user ec2-user.
set -u
CAND="${1:?usage: eval_aws.sh <candidate.c>}"
BN="$(basename "$CAND")"
IP="${BBS_AWS_IP:?set BBS_AWS_IP to the c6i public IP}"
KEY="$HOME/.ssh/pem/graph.pem"
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o LogLevel=ERROR"
rsync -az -e "$SSH" "$CAND" ec2-user@"$IP":~/projects/bandwidth-bound-scan/evolve/ 2>/dev/null
$SSH ec2-user@"$IP" "cd ~/projects/bandwidth-bound-scan/evolve && CC=clang BBS_WIDTHS='${BBS_WIDTHS:-}' BBS_REPEATS=${BBS_REPEATS:-11} BBS_N=${BBS_N:-67108864} ./evaluate.sh $BN"
