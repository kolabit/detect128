#!/bin/bash
#
# Get VectorBlox-SDK, release-v2.0.3
#
if ssh -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
    git clone git@github.com:Microchip-Vectorblox/VectorBlox-SDK.git --branch "release-v2.0.3"
else
    git clone https://github.com/Microchip-Vectorblox/VectorBlox-SDK.git --branch "release-v2.0.3"
fi
