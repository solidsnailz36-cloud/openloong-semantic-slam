"""逐 topic 取一条消息，dump 结构与时间戳，确认新数据集 r2 的传感器格式。
用法: python probe_r2_msgs.py <bag_dir>
"""
import sys

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

bag = sys.argv[1] if len(sys.argv) > 1 else \
    r"D:\E-Downd\openloong_r2_ex\bag\rosbag2_2026_09_23-14_09_45"

ts = get_typestore(Stores.ROS2_HUMBLE)


def stamp_of(msg):
    h = getattr(msg, "header", None)
    if h is None:
        return None
    st = h.stamp
    return float(st.sec) + float(st.nanosec) * 1e-9


with Reader(bag) as reader:
    conns = list(reader.connections)
    print(f"{len(conns)} connections")
    for c in conns:
        print("-" * 70)
        print(f"TOPIC {c.topic}  {c.msgtype}  msgs={c.msgcount}")
        for con, t, raw in reader.messages(connections=[c]):
            m = ts.deserialize_cdr(raw, con.msgtype)
            print(f"  bag_write_t={t * 1e-9:.6f}  header.stamp={stamp_of(m)}")
            name = con.msgtype.split("/")[-1]
            if name == "PointCloud2":
                fm = {f.name: (f.offset, f.datatype, f.count) for f in m.fields}
                print(f"  {m.width} x {m.height}  point_step={m.point_step} "
                      f"row_step={m.row_step} is_dense={m.is_dense} "
                      f"bigendian={m.is_bigendian} frame_id={m.header.frame_id!r}")
                print(f"  fields: {fm}")
                print(f"  data bytes={len(bytes(m.data))}  "
                      f"expect={m.width * m.height * m.point_step}")
            elif name == "Image":
                print(f"  {m.width}x{m.height} encoding={m.encoding!r} "
                      f"step={m.step} is_bigendian={m.is_bigendian} "
                      f"frame_id={m.header.frame_id!r} bytes={len(bytes(m.data))}")
            elif name == "Imu":
                av = m.angular_velocity
                la = m.linear_acceleration
                o = m.orientation
                print(f"  frame_id={m.header.frame_id!r} "
                      f"ang_vel=({av.x:.5f},{av.y:.5f},{av.z:.5f}) "
                      f"lin_acc=({la.x:.4f},{la.y:.4f},{la.z:.4f}) "
                      f"quat=({o.x:.4f},{o.y:.4f},{o.z:.4f},{o.w:.4f}) "
                      f"cov_av[0]={m.angular_velocity_covariance[0]}")
            elif name == "Odometry":
                p = m.pose.pose.position
                q = m.pose.pose.orientation
                tw = m.twist.twist.linear
                print(f"  frame_id={m.header.frame_id!r} child={m.child_frame_id!r}")
                print(f"  pos=({p.x:.4f},{p.y:.4f},{p.z:.4f}) "
                      f"quat=({q.x:.4f},{q.y:.4f},{q.z:.4f},{q.w:.4f})")
                print(f"  twist=({tw.x:.4f},{tw.y:.4f},{tw.z:.4f})")
            break
