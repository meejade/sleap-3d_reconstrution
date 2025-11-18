import time

import cv2
from omegaconf import OmegaConf
from sleap_nn.inference.predictors import SingleInstancePredictor
import os
import numpy as np
import sleap_io as sio
import tempfile
import math
from sleap_io.model.skeleton import Skeleton, Node, Edge

# 你的模型目录（里面有 best.ckpt 和 training_config.yaml）
MODEL_DIR = r"D:\251016_124032.single_instance.n=26"

yaml_path = None
for name in ("training_config.yaml", "initial_config.yaml"):
    p = os.path.join(MODEL_DIR, name)
    if os.path.isfile(p):
        yaml_path = p
        break
if yaml_path is None:
    raise FileNotFoundError("未在模型目录中找到 training_config.yaml 或 initial_config.yaml")
print(yaml_path)
pre_cfg = OmegaConf.load(yaml_path)
nodes = pre_cfg["data_config"]["skeletons"][0]["nodes"]
print(nodes)
edges = pre_cfg["data_config"]["skeletons"][0]["edges"]
print(edges)
NODE_NAMES = [getattr(n, "name", None) or (n["name"] if isinstance(n, dict) else str(n)) for n in nodes]
print(NODE_NAMES)
list_of_edges = []
list_of_nodes = [Node(name=n) for n in NODE_NAMES]
name_to_node = {n.name: n for n in list_of_nodes}
for edge in edges:
    source_node = name_to_node[edge["source"]["name"]]
    target_node = name_to_node[edge["destination"]["name"]]
    list_of_edges.append(Edge(source = source_node, destination = target_node))
print(list_of_edges)

skel = Skeleton(nodes=list_of_nodes,edges=list_of_edges)
print(skel)
defaults = {
    "max_height": 1080,                    # 你也可改成摄像头高度或训练时尺寸
    "max_width": 1920,
    "resize_input_to_multiple_of": 16,    # 一般与 max_stride 一致
    "pad_to_stride": 16,

    # 颜色与格式控制
    "ensure_rgb": False,           # 你已自己转换为 RGB
    "normalize_color": True,
    "convert_range": False,        # 若训练时没标准化到 0–1，则保持 False
    "clip_input_range": True,
    "ensure_grayscale": False,

    # 缩放与标准化
    "scale": 1.0,                  # 缩放系数
    "normalize_input": False,
    "normalize_input_range": False,
    "standardize_input": False,

    # 几何变换相关
    "square": False,
    "center_on_largest": True,
    "rotation": 0.0,
    "flip": False,

    # 其他潜在字段（部分模型训练脚本会访问）
    "crop": None,
    "crop_pad": 0,
    "stride": 16,
    "max_stride": 16,
    "dtype": "uint8",
}
for k, v in defaults.items():
    if k not in pre_cfg["data_config"]["preprocessing"] is None:
        pre_cfg["data_config"]["preprocessing"][k] = v
        print(k)

predictor = SingleInstancePredictor.from_trained_models(
    confmap_ckpt_path = MODEL_DIR,
    preprocess_config = pre_cfg["data_config"]["preprocessing"],
    device="cuda"              # 没有GPU就改为 "cpu"
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# 可选：设置分辨率
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
tmp_dir = tempfile.gettempdir()
tmp_path = os.path.join(tmp_dir, "tmp_frame.png")
def xy_from_pt(pt):
    """将 SLEAP 的 point（可能是结构化标量/ndarray/对象）安全转成 (x, y)。"""
    if pt is None:
        return None

    # 1) 结构化 numpy 标量或数组：优先取 ['xy']
    if isinstance(pt, np.void) and getattr(pt, "dtype", None) is not None and pt.dtype.fields:
        if "xy" in pt.dtype.fields:
            x, y = pt["xy"]
            if math.isnan(x) or math.isnan(y):
                return None
            return int(round(float(x))), int(round(float(y)))

    if isinstance(pt, np.ndarray) and pt.dtype.fields:
        # 结构化 ndarray（很少见，防御性处理）
        if "xy" in pt.dtype.fields:
            xy = pt["xy"]
            # 可能是标量或(1,2)之类
            xy = np.asarray(xy).astype(float).ravel()
            if xy.size < 2 or np.any(np.isnan(xy[:2])):
                return None
            return int(round(xy[0])), int(round(xy[1]))

    # 2) 对象有属性 .xy 或 .x/.y
    if hasattr(pt, "xy"):
        x, y = pt.xy
        if math.isnan(x) or math.isnan(y):
            return None
        return int(round(float(x))), int(round(float(y)))
    if hasattr(pt, "x") and hasattr(pt, "y"):
        x, y = pt.x, pt.y
        if math.isnan(x) or math.isnan(y):
            return None
        return int(round(float(x))), int(round(float(y)))

    # 3) 常规 list/tuple/ndarray
    try:
        arr = np.asarray(pt, dtype=float).ravel()
        if arr.size < 2 or np.any(np.isnan(arr[:2])):
            return None
        return int(round(arr[0])), int(round(arr[1]))
    except Exception:
        return None
frame_count = 0
unrecognized_frames = 0
start = time.time()
while True:
    recognized = True
    ok, frame = cap.read()
    if not ok:
        break
    #cv2.imshow("frame", frame)
    rgb = frame[..., ::-1]
    #cv2.imshow("rgb", rgb)
    # vid = sio.Video.from_numpy(np.expand_dims(rgb, axis=0))  # (1, H, W, 3)
    cv2.imwrite(tmp_path, rgb)  # 写入 BGR 就可以

    predictor.make_pipeline(tmp_path)  # 指定数据源（视频对象/文件路径/Labels）
    labels = predictor.predict()  # 这里才真正跑推理（返回 sio.Labels）

    # 画关键点（容错 None）
    if labels and labels.labeled_frames:
        lf = labels.labeled_frames[0]
        for inst_id, inst in enumerate(lf.instances):
            #print(f"\n🟢 Instance {inst_id}:")
            inst.skeleton = skel
            recognized_labels = 0
            for i, pt in enumerate(inst.points):
                xy = xy_from_pt(pt)  # 用我们刚才定义的安全取点函数
                name = skel.nodes[i].name  # 点的标签名
                if xy is not None:
                    #print(f"  {i:02d} | {name or 'unnamed'} : {xy}")
                    recognized_labels += 1
                else:
                    #print(f"  {i:02d} | {name or 'unnamed'} : None")
                    recognized = False
                    continue
                x, y = xy
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                cv2.putText(frame,name,(x + 8, y - 8),         # 文字左下角坐标（稍微偏右上防止重叠）
                cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                0.5,                    # 字体大小
                (255, 0, 0),            # 颜色（绿色）
                1,                      # 线宽
                cv2.LINE_AA)

            for i, j in skel.edge_inds:
                found_1st = False
                found_2nd = False
                if xy_from_pt(inst.points[i]) is not None:
                    x1, y1 = xy_from_pt(inst.points[i])
                    found_1st = True
                if xy_from_pt(inst.points[j]) is not None:
                    x2, y2 = xy_from_pt(inst.points[j])
                    found_2nd = True
                if found_1st and found_2nd:
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    if not recognized:
        unrecognized_frames += 1
        print(recognized_labels, "out of 9 labels are recognized.")
    cv2.imshow("SLEAP-NN Realtime", frame)
    frame_count += 1
    if cv2.waitKey(1) & 0xFF == 27:  # ESC 退出
        break
end = time.time()
print(frame_count/(end-start)," fps")
cap.release()
cv2.destroyAllWindows()
print((frame_count-unrecognized_frames)," frames are fully recognized from ",frame_count, " frames")
print("The recognition rate is ",(frame_count-unrecognized_frames)/frame_count*100,"%")

