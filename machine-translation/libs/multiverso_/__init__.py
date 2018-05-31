#! /usr/bin/python
# -*- encoding: utf-8 -*-

"""The multiverso library.

Copied from v-yixia.
"""

from api import init, shutdown, barrier, workers_num, worker_id, server_id, is_master_worker
from tables import ArrayTableHandler, MatrixTableHandler
