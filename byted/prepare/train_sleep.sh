#!/bin/bash

sh ./byted/prepare/prepare.sh

export NCCL_HOSTID=${MY_POD_NAME}

sleep infinity
