#!/usr/bin/env bash
RANK=$1
RUN_ID=$2
python3 tf_fedml_main.py --cf config/fedml_config.yaml --run_id $RUN_ID --rank $RANK --role client