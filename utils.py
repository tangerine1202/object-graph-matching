import warnings
from itertools import combinations
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as scipy_R
from scipy.optimize import minimize as scipy_minimize

# ----- Graph -----


def comp_graph_overlap(g1, g2):
    all_inst_ids = list(set(g1.inst_ids) | set(g2.inst_ids))
    anchor_inst_ids = list(set(g1.inst_ids) & set(g2.inst_ids))
    e1i = [g1.inst2node[inst_id] for inst_id in anchor_inst_ids]
    e1j = [g1.inst2node[inst_id] for inst_id in g1.inst_ids if inst_id not in anchor_inst_ids]
    e2i = [g2.inst2node[inst_id] for inst_id in anchor_inst_ids]
    e2j = [g2.inst2node[inst_id] for inst_id in g2.inst_ids if inst_id not in anchor_inst_ids]
    return e1i, e1j, e2i, e2j
