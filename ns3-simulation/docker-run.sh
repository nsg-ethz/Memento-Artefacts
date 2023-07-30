#! /bin/bash

project="generator"
name="${project}-runner"

# Get the current script directory, no matter where we are called from.
# https://stackoverflow.com/questions/59895/how-to-get-the-source-directory-of-a-bash-script-from-within-the-script-itself
dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
uid="$(id -u)"

# Attempt to start it (also succeeds if it's running already).
docker start $name > /dev/null 2>&1 || \
# Otherwise, create a new one.
docker run \
    -td \
    --name $name \
    -v "${dir}/simulation:/ns3/scratch" \
    -v "${dir}/${project}:/ns3/src/${project}" \
    -v "${dir}:/home" \
    -w "/home" \
    "notspecial/ns-3-dev" > /dev/null 2>&1 && \
# After building, set permissions to user.
docker exec $name chown -R $uid /ns3 /home

# Run the provided command in the container.
docker exec --user $uid $name "$@"
