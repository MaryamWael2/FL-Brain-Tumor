#!/usr/bin/env bash
RUN_ID=$1
python3 tf_fedml_main.py --cf config/fedml_config.yaml --run_id $RUN_ID --rank 0 --role server