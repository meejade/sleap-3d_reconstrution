# -*- coding: utf-8 -*-
import os
import cv2
import time
import math
import numpy as np
import tempfile
import multiprocessing as mp
from omegaconf import OmegaConf
from sleap_nn.inference.predictors import SingleInstancePredictor
import cv2
import socket
import json
import time
from collections import deque
from sleap_io.model.skeleton import Skeleton, Node, Edge

# ========= 用户需修改的路径 =========
MODEL_A_DIR = r"D:\Meiqi\Extra files for sleap test\2025-11-10\models\None_251111_103704.single_instance.n=50"  # 摄像头0的模型目录（含 best.ckpt / training_config.yaml）
MODEL_B_DIR = r"D:\Meiqi\Extra files for sleap test\2025-11-10\models\None_251111_123231.single_instance.n=50"  # 摄像头1的模型目录
#CALIB_NPZ   = r"D:\calib\stereo_calib_cam0_cam1.npz"  # 标定结果
USE_CUDA    = True  # 没GPU改 False
NODE_NAMES = []
# ========= 通用：从 YAML 抽取推理所需 preprocess（缺失字段补齐）=========
def extract_preprocess_config(cfg):
    # 常见结构：data.preprocessing
    if "data_config" in cfg and "preprocessing" in cfg["data_config"]:
        pre = cfg["data_config"]["preprocessing"]
    elif "preprocessing" in cfg:
        pre = cfg["preprocessing"]
    else:
        # 回退：顶层（个别项目把键平铺了）
        pre = cfg

    # 归一化 + 默认值兜底
    pre = OmegaConf.create(dict(pre))
    defaults = {
        "max_height": 512,
        "max_width": 512,
        "resize_input_to_multiple_of": 16,
        "pad_to_stride": 16,
        "stride": 16,
        "max_stride": 16,
        "ensure_rgb": False,         # 我们会手动 BGR->RGB
        "normalize_color": True,
        "clip_input_range": True,
        "scale": 1.0,
        "normalize_input": False,
        "normalize_input_range": False,
        "standardize_input": False,
        "square": False,
        "center_on_largest": True,
        "rotation": 0.0,
        "flip": False,
        "dtype": "uint8",
        "crop": None,
        "crop_pad": 0,
    }
    for k, v in defaults.items():
        if k not in pre or pre[k] is None:
            pre[k] = v
    return pre

# ========= 工具：安全拿 (x,y) 与 score =========
def xy_from_pt(pt):
    """将 sleap point（可能是结构化标量/对象/ndarray）转为 (x,y)；无效返回 None。"""
    if pt is None:
        return None
    # 结构化 numpy 标量：优先字段 'xy'
    if isinstance(pt, np.void) and getattr(pt, "dtype", None) is not None and pt.dtype.fields:
        if "xy" in pt.dtype.fields:
            x, y = pt["xy"]
            if math.isnan(x) or math.isnan(y):
                return None
            return float(x), float(y)
    # 对象属性
    if hasattr(pt, "xy"):
        x, y = pt.xy
        if math.isnan(x) or math.isnan(y):
            return None
        return float(x), float(y)
    if hasattr(pt, "x") and hasattr(pt, "y"):
        x, y = pt.x, pt.y
        if math.isnan(x) or math.isnan(y):
            return None
        return float(x), float(y)
    # 常规数组/列表
    try:
        arr = np.asarray(pt, dtype=float).ravel()
        if arr.size < 2 or np.any(np.isnan(arr[:2])):
            return None
        return float(arr[0]), float(arr[1])
    except Exception:
        return None

def score_from_pt(pt, default=1.0):
    """尝试从 point 中取置信度 'score'；取不到则返回 default。"""
    try:
        if isinstance(pt, np.void) and pt.dtype.fields and "score" in pt.dtype.fields:
            s = float(pt["score"])
            return s
    except Exception:
        pass
    if hasattr(pt, "score"):
        try:
            return float(pt.score)
        except Exception:
            pass
    return float(default)

class KF2D_CV:
    def __init__(self, x0, y0, dt, sigma_proc=30.0, sigma_meas=2.5):
        self.x = np.array([x0, y0, 0.0, 0.0], dtype=float)  # [x,y,vx,vy]
        self.P = np.eye(4) * 1e3
        self.dt = dt
        self.sigma_proc = float(sigma_proc)  # 过程噪声标度（像素/秒^2）
        self.sigma_meas = float(sigma_meas)  # 测量噪声（像素）

    def _F(self, dt):
        return np.array([[1,0,dt,0],
                         [0,1,0,dt],
                         [0,0,1, 0],
                         [0,0,0, 1]], dtype=float)

    def _Q(self, dt):
        q = self.sigma_proc ** 2
        dt2, dt3, dt4 = dt*dt, dt**3, dt**4
        return q * np.array([[dt4/4,    0,   dt3/2,  0],
                             [0,     dt4/4,  0,   dt3/2],
                             [dt3/2,   0,    dt2,   0],
                             [0,     dt3/2,  0,    dt2]], dtype=float)

    def predict(self, dt=None, inflate=1.0):
        if dt is None: dt = self.dt
        F = self._F(dt); Q = self._Q(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        if inflate != 1.0:
            self.P *= float(inflate)
        return self.x.copy()

    def update(self, z_xy):
        H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
        R = np.eye(2) * (self.sigma_meas ** 2)
        z = np.array(z_xy, dtype=float)
        y = z - (H @ self.x)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        I = np.eye(4)
        self.P = (I - K @ H) @ self.P
        return self.x.copy()

def maha_gate(z_xy, kf, thr):
    """马氏距离门控：返回 (是否接受, 距离)。"""
    H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
    R = np.eye(2) * (kf.sigma_meas ** 2)
    innov = np.array(z_xy, float) - (H @ kf.x)
    S = H @ kf.P @ H.T + R
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        # 协方差不可逆，说明当前状态太不确定了，直接拒绝这次观测
        return False, float("inf")
    m2 = float(innov.T @ S_inv @ innov)

    # ✨ 数值保护：有时候会是 -1e-12 这种小负数，夹到 0
    if m2 < 0:
        m2 = 0.0

    # 安全开根
    md = math.sqrt(m2)
    return (m2 <= thr*thr), math.sqrt(m2)

def kinematic_gate(i, z_xy, now_t, frame_wh, hist, JUMP_PX, V_MAX, ACC_MAX):
    """运动学门控：跳变/速度/加速度/屏外。"""
    if z_xy is None:
        return False
    W, H = frame_wh
    x, y = z_xy
    if not (0 <= x < W and 0 <= y < H):
        return False
    dq = hist[i]
    ok = True
    if len(dq) >= 1:
        t0, x0, y0 = dq[-1]
        dt = max(1e-3, now_t - t0)
        jump = math.hypot(x - x0, y - y0)
        v = jump / dt
        if jump > JUMP_PX or v > V_MAX:
            ok = False
        if len(dq) >= 2:
            t1, x1, y1 = dq[-2]
            dt1 = max(1e-3, t0 - t1)
            v0 = math.hypot(x0 - x1, y0 - y1) / dt1
            a = abs(v - v0) / dt
            if a > ACC_MAX:
                ok = False
    return ok

def build_edge_inds(NODE_NAMES, edges_cfg):
    name2idx = {n: i for i, n in enumerate(NODE_NAMES)}
    out = []
    for e in edges_cfg:
        si = name2idx.get(e["source"]["name"], None)
        di = name2idx.get(e["destination"]["name"], None)
        if si is not None and di is not None:
            out.append((si, di))
    return out

def init_ref_lengths(xy_now, EDGE_INDS):
    """初始化参考骨长（可以改成多帧中位数更稳）。"""
    ref = {}
    for (i, j) in EDGE_INDS:
        vi = xy_now[i]; vj = xy_now[j]
        if not (np.any(np.isnan(vi)) or np.any(np.isnan(vj))):
            ref[(i,j)] = float(np.linalg.norm(vj - vi))
    return ref

def bone_length_project(i, z_xy, est_xy, EDGE_INDS, REF_LEN, L_TOL):
    """把 z_xy 投影回以父点为圆心的[Lmin,Lmax]环带范围内。"""
    # 只处理作为子端点的边 (p -> i)
    candidates = [(p, q) for (p, q) in EDGE_INDS if q == i]
    if not candidates:
        return z_xy, False
    zx, zy = z_xy
    z = np.array([zx, zy], float)
    changed = False
    for (p, q) in candidates:
        if (p, q) not in REF_LEN:
            continue
        L0 = REF_LEN[(p, q)]
        Lmin, Lmax = L0 * (1 - L_TOL), L0 * (1 + L_TOL)
        pi = est_xy[p]
        if np.any(np.isnan(pi)):
            continue
        v = z - pi
        d = np.linalg.norm(v)
        if d < 1e-6:
            continue
        if d < Lmin or d > Lmax:
            L = np.clip(d, Lmin, Lmax)
            z = pi + v * (L / d)
            changed = True
    return (float(z[0]), float(z[1])), changed

# ========= 进程：摄像头 + 模型 推理 → 2D结果输出 =========
def camera_worker(cam_index:int, MODEL_DIR:str, out_queue:mp.Queue, stop_event:mp.Event):
    # 参数
    global NODE_NAMES
    TAU_SCORE = 0.35
    MAHA_THR = 5.0
    JUMP_PX = 80.0
    V_MAX = 30.0  # 像素/秒
    ACC_MAX = 15.0  # 像素/秒^2
    L_TOL = 0.2  # 骨长相对容差 ±35%
    BLEND = 0.4  # 纠偏后与原观测的融合比例

    hist = None  # 每点一个 deque 存 (t,x,y)
    REF_LEN = None  # 参考骨长 dict[(i,j)]=length

    # 1) 加载模型
    YAML_PATH = os.path.join(MODEL_DIR, "training_config.yaml")
    if not os.path.exists(YAML_PATH):
        YAML_PATH = os.path.join(MODEL_DIR, "training_config.yaml")

    cfg = OmegaConf.load(YAML_PATH)
    pre_cfg = extract_preprocess_config(cfg)

    # ---- 从 training_config.yaml 中提取 skeleton（节点名与边）----
    # 你的文件里此路径存在：data_config.skeletons[0]
    skeleton_cfg = cfg.get("data_config", {}).get("skeletons", [])[0]
    nodes_cfg = skeleton_cfg["nodes"]  # [{'name':'head'}, ...]
    edges_cfg = skeleton_cfg["edges"]  # [{'source':{'name':..}, 'destination':{'name':..}}, ...]
    # print(edges_cfg)

    NODE_NAMES = [getattr(n, "name", None) or (n["name"] if isinstance(n, dict) else str(n)) for n in nodes_cfg]
    # Skeleton/Node/Edge 类的导入（兼容不同版本）

    skel = None
    # print("created")
    # 创建唯一节点对象列表，边必须引用同一批 Node 实例
    node_objs = [Node(name=n) for n in NODE_NAMES]
    name2node = {n.name: n for n in node_objs}
    # print(name2node.keys())
    edge_objs = []
    if Edge is not None:
        for e in edges_cfg:
            src = e["source"]["name"]
            dst = e["destination"]["name"]
            # print(src, "-",dst)
            if src in name2node and dst in name2node:
                # print("found")
                edge_objs.append(Edge(source=name2node[src], destination=name2node[dst]))
        skel = Skeleton(nodes=node_objs, edges=edge_objs)
        # print("have edges")
    else:
        # 没有 Edge 类就只建节点；edge_inds 渲染时另算
        skel = Skeleton(nodes=node_objs)
        # print("only nodes")
    # print(skel)
    EDGE_INDS = build_edge_inds(NODE_NAMES, edges_cfg)
    # ========= 2) 预测器 =========
    predictor = SingleInstancePredictor.from_trained_models(
        confmap_ckpt_path=MODEL_DIR,  # 传目录，内部会找 best/latest.ckpt
        preprocess_config=pre_cfg,  # 刚提取并补齐的 OmegaConf
        device="cuda"  # 无GPU用 "cpu"
    )

    # 2) 打开摄像头
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)  # Windows常用
    if not cap.isOpened():
        print(f"[Cam{cam_index}] ❌ Failed to open camera.")
        return
    print(f"[Cam{cam_index}] ✅ Camera opened.")

    # 3) 临时文件（适配 sleap 1.5.1）
    tmp_img = os.path.join(tempfile.gettempdir(), f"sleap_cam{cam_index}.png")

    tau = 0.35  # 置信度门限
    sigma_proc = 30.0  # 过程噪声尺
    # 度（越大越“跟得动”，太大易抖）
    sigma_meas = 2.5  # 测量噪声（像素）
    max_miss_inflate = 10  # 缺测膨胀上限（控制发散）

    kfs = None  # 每点一个 KF
    miss_cnt = None  # 连续缺测计数
    REF_LEN = None
    prev_t = time.time()
    if hist is None:
        # 注意：这里要知道关键点数量，先用 NODE_NAMES 的长度
        hist = [deque(maxlen=5) for _ in range(len(NODE_NAMES))]
    print("Press ESC to quit.")

    frame_count = 0
    unrecognized_frames = 0
    start = time.time()

    # 4) 主循环：读帧→预测→发队列
    try:
        while not stop_event.is_set():
            ok, frame_bgr = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            # 写临时图 → 预测

            # 1) 预测（Dead-Reckoning）：先基于 dt 外推一遍
            t = time.time()
            dt = max(1e-3, t - prev_t)
            prev_t = t

            if kfs is not None:
                for i, kf in enumerate(kfs):
                    # 连续缺测计数越大，协方差膨胀越多
                    inflate = 1.0 + min(miss_cnt[i], max_miss_inflate) * 0.25
                    kf.predict(dt=dt, inflate=inflate)

            # 2) 调用 SLEAP 推理（这里用“临时图片 + make_pipeline(path)”的办法）
            #    你也可以换成你当前已经跑通的那套管线；核心是得到 labels 这个对象
            cv2.imwrite(tmp_img, frame_bgr)
            predictor.make_pipeline(tmp_img)
            labels = predictor.predict()

            #frame_bgr = cv2.resize(frame_bgr, (1080,720))
            # 3) 抽取预测坐标与置信度（单实例：取第0个 instance）
            H, W = frame_bgr.shape[:2]
            pts_xy = []
            pts_sc = []
            print(cam_index,":",H,",",W)
            if labels and labels.labeled_frames:
                lf = labels.labeled_frames[0]
                insts = lf.instances
                if len(insts) > 0:
                    inst = insts[0]
                    for pt in inst.points:
                        xy = xy_from_pt(pt)
                        sc = score_from_pt(pt, default=1.0)
                        pts_xy.append(xy)  # None or (x,y)
                        pts_sc.append(sc)
                else:
                    pts_xy = [None] * len(NODE_NAMES)
                    pts_sc = [0.0] * len(NODE_NAMES)
            else:
                pts_xy = [None] * len(NODE_NAMES)
                pts_sc = [0.0] * len(NODE_NAMES)

            # 初始化 KF（在第一帧有观测时）
            if kfs is None:
                kfs, miss_cnt = [], []
                for i in range(len(NODE_NAMES)):
                    if pts_xy[i] is not None and pts_sc[i] >= tau:
                        x0, y0 = pts_xy[i]
                    else:
                        x0, y0 = W / 2, H / 2  # 没观测就先放中心，后续会被更新拉回
                    kfs.append(KF2D_CV(x0, y0, dt=1 / 30.0, sigma_proc=sigma_proc, sigma_meas=sigma_meas))
                    miss_cnt.append(0)
                miss_cnt = np.array(miss_cnt, dtype=int)

            # 融合：判错 + 纠偏 + KF update / predict
            est_xy = []
            for i, kf in enumerate(kfs):
                z_ok = (pts_xy[i] is not None) and (pts_sc[i] >= TAU_SCORE)
                z = pts_xy[i] if z_ok else None

                accept = False
                if z_ok:
                    # 统计门控
                    is_ok_m, md = maha_gate(z, kf, thr=MAHA_THR)
                    # 运动学门控
                    is_ok_k = kinematic_gate(i, z_xy=z, now_t=t, frame_wh=(W, H), hist = hist,
                                              JUMP_PX = JUMP_PX, V_MAX = V_MAX, ACC_MAX = ACC_MAX)
                    #accept = True # only kf
                    accept = is_ok_m # only maha gate
                    #accept = is_ok_k # only kinematic gate
                    #accept = (is_ok_m and is_ok_k) # maha gate and kinematic gate

                    # 骨长纠偏后再验一次
                    if not accept and REF_LEN is not None:
                        # 先用当前 KF 状态组成 est 作为父点参考
                        est_now = np.array([(kk.x[0], kk.x[1]) for kk in kfs], dtype=float)
                        z_corr, changed = bone_length_project(i, z, est_now,EDGE_INDS, REF_LEN, L_TOL)
                        if changed:
                            is_ok_m2, md2 = maha_gate(z_corr, kf, thr=MAHA_THR)
                            is_ok_k2 = kinematic_gate(i, z_xy=z_corr, now_t=t, frame_wh=(W, H), hist = hist,
                                             JUMP_PX = JUMP_PX, V_MAX = V_MAX, ACC_MAX = ACC_MAX)
                            if is_ok_m2 and is_ok_k2:
                                z = ((1.0 - BLEND) * np.array(z) + BLEND * np.array(z_corr)).tolist()
                                accept = True

                if accept:
                    kf.update(z)
                    miss_cnt[i] = 0
                else:
                    miss_cnt[i] += 1

                est_xy.append((kf.x[0], kf.x[1]))

            est_xy = np.array(est_xy, dtype=float)

            # 初始化或慢更新参考骨长（更稳：用多帧中位数）
            if REF_LEN is None:
                REF_LEN = init_ref_lengths(est_xy, EDGE_INDS = EDGE_INDS)

            # 更新历史（用于跳变/速度判定）
            for i, (x, y) in enumerate(est_xy):
                if not (np.isnan(x) or np.isnan(y)):
                    hist[i].append((t, float(x), float(y)))

            # 6) 叠加可视化（骨架 + 点 + 名字）
            draw = frame_bgr.copy()

            # 6.1 画骨架（用 skel.edge_inds；若类不提供 edge_inds，则从 edges_cfg 自己生成索引对）
            def get_edge_inds():
                if skel is not None and hasattr(skel, "edge_inds") and skel.edge_inds is not None:
                    return list(skel.edge_inds)
                # 退路：用 NODE_NAMES 映射 edges_cfg → 索引对
                name2idx = {n: i for i, n in enumerate(NODE_NAMES)}
                idx_pairs = []
                for e in edges_cfg:
                    si = name2idx.get(e["source"]["name"], None)
                    di = name2idx.get(e["destination"]["name"], None)
                    if si is not None and di is not None:
                        idx_pairs.append((si, di))
                return idx_pairs

            for (i, j) in get_edge_inds():
                xi, yi = est_xy[i]
                xj, yj = est_xy[j]
                if not (np.isnan(xi) or np.isnan(yi) or np.isnan(xj) or np.isnan(yj)):
                    cv2.line(draw, (int(round(xi)), int(round(yi))),
                             (int(round(xj)), int(round(yj))),
                             (255, 255, 0), 2, cv2.LINE_AA)

            # 6.2 画点与名字（名字来自 NODE_NAMES，与模型点索引一致）
            for i, (x, y) in enumerate(est_xy):
                if np.isnan(x) or np.isnan(y):
                    continue
                p = (int(round(x)), int(round(y)))
                cv2.circle(draw, p, 4, (0, 255, 0), -1)
                # 名字稍微偏移避免遮盖
                name = NODE_NAMES[i] if i < len(NODE_NAMES) else f"node_{i}"
                text_pt = (min(p[0] + 8, W - 100), max(p[1] - 8, 15))
                cv2.putText(draw, name, text_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                # 发送（只发轻量2D数据）
                payload = {
                    "cam": cam_index,
                    "t": time.time(),
                    "shape": frame_bgr.shape[:2],  # (H,W)
                    "xy": est_xy,  # [(x,y) or None, ...] 按节点顺序                 # [float, ...]
                    "score": pts_sc
                }
                out_queue.put(payload)

            # 7) 显示
            cv2.imshow("SLEAP + Dead-Reckoning (ESC to quit)", draw)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
    except KeyboardInterrupt:
        pass
    finally:
        end = time.time()
        print(cam_index,":",frame_count/(end-start)," fps")
        cap.release()
        cv2.destroyAllWindows()
        print(f"[Cam{cam_index}] ⏹ Stopped.")

def make_projection_matrix(K, R, t):
    """返回 3x4 投影矩阵 P = K [R | t]"""
    Rt = np.hstack((R, t.reshape(3,1)))
    P = K.dot(Rt)
    return P

# 线性多视图三角化（DLT）
def triangulate_multiview(points_uv, proj_mats):
    # 至少两视角
    if len(points_uv) < 2:
        return None

    A_rows = []
    for (u, v), P in zip(points_uv, proj_mats):
        # 如果这一行的点是 nan，就跳过
        if np.isnan(u) or np.isnan(v):
            continue
        A_rows.append(u * P[2, :] - P[0, :])
        A_rows.append(v * P[2, :] - P[1, :])

    if len(A_rows) < 4:  # 2个视角 × 2 行 = 4 行，少于4行就不能三角化
        return None

    A = np.vstack(A_rows).astype(np.float64)

    # 再检查一次
    if np.isnan(A).any() or np.isinf(A).any():
        return None

    try:
        _, _, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None

    X = Vt[-1]
    if abs(X[3]) < 1e-8:
        return None

    X = X / X[3]
    return X[:3]

class MovingAverage:
    def __init__(self, window=5):
        self.window = window
        self.buff = deque(maxlen=window)
    def update(self, pt):
        if pt is None:
            return None
        self.buff.append(pt)
        arr = np.array(self.buff)
        return arr.mean(axis=0)

# ========= 进程：三角化融合 =========
def fusion_worker(in_queue_a:mp.Queue, in_queue_b:mp.Queue, stop_event:mp.Event, min_score:float=0.3):
    UDP_IP = "127.0.0.1"
    UDP_PORT = 5005
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # 1) 读取标定
    camera_params = [
        # 示例占位（务必用实际标定结果替换）
        {

        #     "K": np.array([[1.18393561e+03, 0.00000000e+00, 9.59500000e+02],
        #                 [0.00000000e+00, 1.18636266e+03, 5.39500000e+02],
        #                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
        #     "dist": np.array([[ 0.15464462, -0.27189405,  0, 0, 0]]),
        #     # 示例外参：第二台相机绕y轴平移 200mm
        #     "R": np.array([[ 0.99951816, 0.00846809, 0.02986208],
        #                  [-0.01769308, 0.9459, 0.32397554],
        #                  [-0.02550309,  -0.32434779, 0.94559405  ]] ),
        #     #"t": np.array([80.146, 430.1, 839.5])  # 单位：米（或与K相同尺度）
        #     "t": np.array([0.14335747, -0.09043696, 0.94931117])  # 单位：米（或与K相同尺度）
        # },
        # {
        #     "K": np.array([[1.13204588e+03, 0.00000000e+00, 9.59500000e+02],
        #                      [0.00000000e+00, 1.12927387e+03, 5.39500000e+02],
        #                      [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
        #     "dist": np.array([[  0.15777187, -0.29436901, 0,  0, 0]]),
        #     "R": np.array([[ 0.9213385,  -0.33172853, 0.20271049],
        #                      [ 0.21218428, 0.86600362, 0.45278644],
        #                      [-0.3257502,  -0.3741576, 0.86827006]]),
        #     #"t": np.array([9.5, -304.788, 937.249])
        #     "t": np.array([0.05804301, -0.0985619, 0.9754403])

            "K": np.array([[1.18393561e+03, 0.00000000e+00, 9.59500000e+02],
                           [0.00000000e+00, 1.18636266e+03, 5.39500000e+02],
                           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
            "dist": np.array([[0.15464462, -0.27189405, 0, 0, 0]]),
            # 示例外参：第二台相机绕y轴平移 200mm
            "R": np.array([[0.99979966, -0.01579985, -0.01228871],
                           [-0.01916193, -0.93295716, -0.35947706],
                           [-0.00578516, 0.35964052, -0.933073]]),
            # "t": np.array([80.146, 430.1, 839.5])  # 单位：米（或与K相同尺度）
            "t": np.array([0.11008484, 0.06270965, 0.95567514])  # 单位：米（或与K相同尺度）
        },
        {
            "K": np.array([[1.13204588e+03, 0.00000000e+00, 9.59500000e+02],
                           [0.00000000e+00, 1.12927387e+03, 5.39500000e+02],
                           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
            "dist": np.array([[0.15777187, -0.29436901, 0, 0, 0]]),
            "R": np.array([[0.92332481, 0.32820213, -0.19938571],
                           [0.21084256, -0.86720876, -0.45110352],
                           [-0.32096217, 0.37447607, -0.86991434]]),
            # "t": np.array([9.5, -304.788, 937.249])
            "t": np.array([0.0016432, 0.03631994, 0.93248549])

        }
    ]
    NODE_NAMES = ["head","chest","chest_left","chest_right","tail"]
    # 投影矩阵（使用去畸变后的“归一化像素坐标”）
    # 在归一化平面上，P1=[I|0], P2=[R|t]
    proj_mats = [make_projection_matrix(p["K"], p["R"], p["t"]) for p in camera_params]

    # 最近一帧缓存
    last_a = None
    last_b = None

    print("[Fusion] ✅ Started. Waiting for data... (Press Ctrl+C in main to stop)")
    try:
        while not stop_event.is_set():
            # 尝试拉取两队列的最新数据（非阻塞）
            try:
                while True:
                    last_a = in_queue_a.get_nowait()
            except mp.queues.Empty:
                pass
            try:
                while True:
                    last_b = in_queue_b.get_nowait()
            except mp.queues.Empty:
                pass

            if last_a is None or last_b is None:
                time.sleep(0.01)
                continue

            # 时间匹配（简单起见：直接用各自最新的；严格可按时间戳对齐）
            xy_a, sc_a, shape_a = last_a["xy"], last_a["score"], last_a["shape"]
            xy_b, sc_b, shape_b = last_b["xy"], last_b["score"], last_b["shape"]

            if xy_a is None or xy_b is None:
                time.sleep(0.005)
                continue

            H1, W1 = shape_a
            H2, W2 = shape_b

            # 逐节点三角化
            pts3d = {}
            for i in range(max(len(xy_a), len(xy_b))):
                pa = xy_a[i] if i < len(xy_a) else None
                pb = xy_b[i] if i < len(xy_b) else None
                sa = sc_a[i] if i < len(sc_a) else 0.0
                sb = sc_b[i] if i < len(sc_b) else 0.0

                # 置信度门限
                if pa is None or pb is None or sa < min_score or sb < min_score:
                    pts3d[NODE_NAMES[i]] = None
                    #pts3d.append(None)
                    continue

                xa, ya = pa
                xb, yb = pb

                # 去畸变 → 归一化像素坐标（在相机坐标系的z=1平面）
                # 注意：输入必须是 float32，形状 (N,1,2)
                und_a = cv2.undistortPoints(
                    np.array([[[xa, ya]]], dtype=np.float32), camera_params[0]["K"], camera_params[0]["dist"],P=camera_params[0]["K"]
                )  # (1,1,2)
                und_b = cv2.undistortPoints(
                    np.array([[[xb, yb]]], dtype=np.float32), camera_params[1]["K"], camera_params[1]["dist"],P=camera_params[1]["K"]
                )

                ua, va = float(und_a[0,0,0]), float(und_a[0,0,1])
                ub, vb = float(und_b[0,0,0]), float(und_b[0,0,1])



                # 构造齐次像素（归一化平面）
                # pts1 = np.array([[ua], [va]], dtype=np.float32)
                # pts2 = np.array([[ub], [vb]], dtype=np.float32)

                # 三角化（归一化平面用 P1,P2）
                X = triangulate_multiview([(ua,va),(ub,vb)],[proj_mats[0],proj_mats[1]])  # (4,N)


                pts3d[NODE_NAMES[i]] = (float(X[0]), float(X[1]), float(X[2]))
                print(NODE_NAMES[i],": ",float(X[0]), ",", float(X[1]), ",", float(X[2]))
            print("------------------------------------------------------")
            print(pts3d)
            packet = {
                "timestamp": time.time(),
                "points": pts3d
            }
            sock.sendto(json.dumps(packet).encode('utf-8'), (UDP_IP, UDP_PORT))
            # 这里你可以：
            # - 打印
            # - 存入共享内存
            # - 通过socket/zmq发往Unity
            # 示例：打印前3个点
            # nice = [p for p in pts3d if p is not None]
            # if nice:
            #     print("[Fusion] 3D sample:", nice[:3])

            time.sleep(0.005)

    except KeyboardInterrupt:
        pass
    finally:
        print("[Fusion] ⏹ Stopped.")

# ========= 主进程：启动/管理 三个子进程 =========
def main():
    mp.set_start_method("spawn", force=True)  # Windows 推荐

    # 两条数据队列（无界队列也可，建议有限大小避免堆积）
    q_a = mp.Queue(maxsize=8)
    q_b = mp.Queue(maxsize=8)

    # 停止事件
    stop_event = mp.Event()

    # 子进程
    proc_cam0 = mp.Process(target=camera_worker, args=(0, MODEL_A_DIR, q_a, stop_event))
    proc_cam1 = mp.Process(target=camera_worker, args=(1, MODEL_B_DIR, q_b, stop_event))
    proc_fuse = mp.Process(target=fusion_worker, args=(q_a, q_b, stop_event, 0.35))

    # 启动
    proc_cam0.start()
    proc_cam1.start()
    proc_fuse.start()

    print("[Main] ✅ Started. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("[Main] ⏹ Stopping...")
        stop_event.set()
    finally:
        for p in (proc_cam0, proc_cam1, proc_fuse):
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        print("[Main] Bye.")

if __name__ == "__main__":
    main()
