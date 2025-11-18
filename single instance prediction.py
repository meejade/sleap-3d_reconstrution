import os
import time
import math
import tempfile
import numpy as np
import cv2
from sleap_io.model.skeleton import Skeleton, Node, Edge
from omegaconf import OmegaConf
from sleap_nn.inference.predictors import SingleInstancePredictor
from collections import deque

# ========= 1) 路径与加载 =========
MODEL_DIR = r"D:\251016_124032.single_instance.n=26"  # ← 你的模型目录（有 best.ckpt / training_config.yaml）
YAML_PATH = os.path.join(MODEL_DIR, "training_config.yaml")
if not os.path.exists(YAML_PATH):
    YAML_PATH = os.path.join(MODEL_DIR, "training_config.yaml")

cfg = OmegaConf.load(YAML_PATH)

# ---- 提取预处理配置，并补齐 sleap-nn 预测阶段会访问到的键 ----
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

pre_cfg = extract_preprocess_config(cfg)

# ---- 从 training_config.yaml 中提取 skeleton（节点名与边）----
# 你的文件里此路径存在：data_config.skeletons[0]
skeleton_cfg = cfg.get("data_config", {}).get("skeletons", [])[0]
nodes_cfg = skeleton_cfg["nodes"]  # [{'name':'head'}, ...]
edges_cfg = skeleton_cfg["edges"]  # [{'source':{'name':..}, 'destination':{'name':..}}, ...]
#print(edges_cfg)
NODE_NAMES = [getattr(n, "name", None) or (n["name"] if isinstance(n, dict) else str(n)) for n in nodes_cfg]
#print(NODE_NAMES)
# Skeleton/Node/Edge 类的导入（兼容不同版本）

skel = None
#print("created")
# 创建唯一节点对象列表，边必须引用同一批 Node 实例
node_objs = [Node(name=n) for n in NODE_NAMES]
name2node = {n.name: n for n in node_objs}
#print(name2node.keys())
edge_objs = []
if Edge is not None:
    for e in edges_cfg:
        src = e["source"]["name"]
        dst = e["destination"]["name"]
        #print(src, "-",dst)
        if src in name2node and dst in name2node:
            #print("found")
            edge_objs.append(Edge(source=name2node[src], destination=name2node[dst]))
    skel = Skeleton(nodes=node_objs, edges=edge_objs)
    #print("have edges")
else:
    # 没有 Edge 类就只建节点；edge_inds 渲染时另算
    skel = Skeleton(nodes=node_objs)
    #print("only nodes")
#print(skel)
# ========= 2) 预测器 =========
predictor = SingleInstancePredictor.from_trained_models(
    confmap_ckpt_path=MODEL_DIR,     # 传目录，内部会找 best/latest.ckpt
    preprocess_config=pre_cfg,       # 刚提取并补齐的 OmegaConf
    device="cuda"                    # 无GPU用 "cpu"
)

# ========= 3) 工具函数：点坐标与分数提取 =========
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

# ========= 4) KF：每点一个常速度模型 =========
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

# ============ 5) 判错 + 纠偏（Mahalanobis / 运动学 / 骨长） ============
# 参数
TAU_SCORE = 0.35
MAHA_THR = 5.0
JUMP_PX = 80.0
V_MAX = 1200.0        # 像素/秒
ACC_MAX = 8000.0      # 像素/秒^2
L_TOL = 0.35          # 骨长相对容差 ±35%
BLEND = 0.4           # 纠偏后与原观测的融合比例

hist = None           # 每点一个 deque 存 (t,x,y)
REF_LEN = None        # 参考骨长 dict[(i,j)]=length

def maha_gate(z_xy, kf, thr=MAHA_THR):
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

def kinematic_gate(i, z_xy, now_t, frame_wh):
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

EDGE_INDS = build_edge_inds(NODE_NAMES, edges_cfg)

def init_ref_lengths(xy_now):
    """初始化参考骨长（可以改成多帧中位数更稳）。"""
    ref = {}
    for (i, j) in EDGE_INDS:
        vi = xy_now[i]; vj = xy_now[j]
        if not (np.any(np.isnan(vi)) or np.any(np.isnan(vj))):
            ref[(i,j)] = float(np.linalg.norm(vj - vi))
    return ref

def bone_length_project(i, z_xy, est_xy):
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

# ========= 5) 实时循环 =========
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# 可选：降低分辨率提速（确保为16的倍数更友好）
# cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

tmp_dir = tempfile.gettempdir()
tmp_img = os.path.join(tmp_dir, "sleap_frame.png")

tau = 0.35            # 置信度门限
sigma_proc = 30.0     # 过程噪声尺
# 度（越大越“跟得动”，太大易抖）
sigma_meas = 2.5      # 测量噪声（像素）
max_miss_inflate = 10 # 缺测膨胀上限（控制发散）

kfs = None            # 每点一个 KF
miss_cnt = None       # 连续缺测计数
REF_LEN = None
prev_t = time.time()
if hist is None:
    # 注意：这里要知道关键点数量，先用 NODE_NAMES 的长度
    hist = [deque(maxlen=5) for _ in range(len(NODE_NAMES))]
print("Press ESC to quit.")

frame_count = 0
unrecognized_frames = 0
start = time.time()

while True:
    ok, frame_bgr = cap.read()
    if not ok:
        break

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
    cv2.imwrite(tmp_img, frame_bgr)             # BGR 写文件，管线会读
    predictor.make_pipeline(tmp_img)             # 指定数据源
    labels = predictor.predict()                 # 运行推理（SingleInstance 模型：一只/一套点）

    # 3) 抽取预测坐标与置信度（单实例：取第0个 instance）
    H, W = frame_bgr.shape[:2]
    pts_xy = []
    pts_sc = []

    if labels and labels.labeled_frames:
        lf = labels.labeled_frames[0]
        insts = lf.instances
        if len(insts) > 0:
            inst = insts[0]
            for pt in inst.points:
                xy = xy_from_pt(pt)
                sc = score_from_pt(pt, default=1.0)
                pts_xy.append(xy)    # None or (x,y)
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
                x0, y0 = W/2, H/2  # 没观测就先放中心，后续会被更新拉回
            kfs.append(KF2D_CV(x0, y0, dt=1/30.0, sigma_proc=sigma_proc, sigma_meas=sigma_meas))
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
            is_ok_k = kinematic_gate(i, z_xy=z, now_t=t, frame_wh=(W, H))
            accept = (is_ok_m and is_ok_k)

            # 骨长纠偏后再验一次
            if not accept and REF_LEN is not None:
                # 先用当前 KF 状态组成 est 作为父点参考
                est_now = np.array([(kk.x[0], kk.x[1]) for kk in kfs], dtype=float)
                z_corr, changed = bone_length_project(i, z, est_now)
                if changed:
                    is_ok_m2, md2 = maha_gate(z_corr, kf, thr=MAHA_THR)
                    is_ok_k2 = kinematic_gate(i, z_xy=z_corr, now_t=t, frame_wh=(W, H))
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
        REF_LEN = init_ref_lengths(est_xy)

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
        cv2.putText(draw, name, text_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

    # 7) 显示
    cv2.imshow("SLEAP + Dead-Reckoning (ESC to quit)", draw)
    frame_count += 1
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

end = time.time()
print(frame_count/(end-start)," fps")
cap.release()
cv2.destroyAllWindows()
# print((frame_count-unrecognized_frames)," frames are fully recognized from ",frame_count, " frames")
# print("The recognition rate is ",(frame_count-unrecognized_frames)/frame_count*100,"%")
