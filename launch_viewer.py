#!/usr/bin/env python3
"""
JAKA Zu5 机械臂 MuJoCo 可视化启动脚本

功能:
  1. 加载 JAKA Zu5 MJCF 模型 (默认使用修正后的模型)
  2. 打印模型基本信息 (关节、连杆、驱动)
  3. 启动 MuJoCo 交互式可视化窗口

交互控制:
  - 鼠标左键拖动: 旋转视角
  - 鼠标右键拖动: 平移视角
  - 滚轮: 缩放
  - 双击物体: 施加力
  - Ctrl+双击: 施加力矩
  - 按 Space: 暂停/继续物理仿真
  - 按 R: 重置仿真到初始状态
  - 按 C: 切换坐标系显示
  - 按 J: 切换关节轴显示
  - 按 G: 切换接触点显示
  - 按 T: 切换透明度
  - 按 L: 切换标签显示
  - 按 0-9: 切换不同可视化层

用法:
  python launch_viewer.py                          # 加载修正模型
  python launch_viewer.py --model jaka_zu5_model.xml  # 加载原始模型
  python launch_viewer.py --info-only               # 仅打印信息不启动viewer
"""

import os
import sys
import argparse
import numpy as np

# ============================================================
# MuJoCo 加载
# ============================================================
try:
    import mujoco
    print(f"[OK] MuJoCo 版本: {mujoco.__version__}")
except ImportError:
    print("[ERROR] MuJoCo 未安装，正在尝试安装...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mujoco", "--break-system-packages"])
    import mujoco
    print(f"[OK] MuJoCo 安装成功，版本: {mujoco.__version__}")

# 尝试导入 viewer 模块 (可能因缺少 GLFW/显示环境而失败)
try:
    import mujoco.viewer
    VIEWER_AVAILABLE = True
except ImportError:
    VIEWER_AVAILABLE = False


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(SCRIPT_DIR, "jaka_zu5_model_corrected.xml")
ORIGINAL_MODEL = os.path.join(SCRIPT_DIR, "jaka_zu5_model.xml")

# 关节类型映射
JOINT_TYPES = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}


def print_model_info(model):
    """打印模型基本信息"""
    print("\n" + "=" * 60)
    print("  JAKA Zu5 模型信息")
    print("=" * 60)

    print(f"\n  Body 数量:      {model.nbody}")
    print(f"  Joint 数量:     {model.njnt}")
    print(f"  Geom 数量:      {model.ngeom}")
    print(f"  Mesh 数量:      {model.nmesh}")
    print(f"  Actuator 数量:  {model.nu}")
    print(f"  qpos 维度:      {model.nq}")
    print(f"  qvel 维度:      {model.nv}")
    print(f"  时间步:         {model.opt.timestep:.4f} s")

    # Body 列表
    print(f"\n  --- Body 列表 (含质量) ---")
    total_mass = 0.0
    for i in range(model.nbody):
        name = model.body(i).name
        mass = float(model.body(i).mass)
        total_mass += mass
        print(f"    [{i}] {name:10s}  mass={mass:.4f} kg")
    print(f"    总质量: {total_mass:.4f} kg")

    # Joint 列表
    print(f"\n  --- Joint 列表 ---")
    for i in range(model.njnt):
        name = model.jnt(i).name
        jtype = int(model.jnt(i).type)
        rng = model.jnt(i).range
        axis = model.jnt(i).axis
        print(f"    [{i}] {name:10s}  type={JOINT_TYPES.get(jtype, jtype):6s}  "
              f"range=({float(rng[0]):.3f}, {float(rng[1]):.3f})  "
              f"axis=({float(axis[0]):.0f}, {float(axis[1]):.0f}, {float(axis[2]):.0f})")

    # Actuator 列表
    print(f"\n  --- Actuator 列表 ---")
    for i in range(model.nu):
        name = model.actuator(i).name
        ctrlrange = model.actuator(i).ctrlrange
        print(f"    [{i}] {name:20s}  ctrlrange=({float(ctrlrange[0]):.0f}, {float(ctrlrange[1]):.0f})")

    # Keyframe 列表
    print(f"\n  --- Keyframe 列表 ---")
    for i in range(model.nkey):
        name = model.key(i).name
        qpos = model.key(i).qpos.copy()
        print(f"    [{i}] {name:10s}  qpos={qpos}")


def launch_viewer(model_path):
    """启动 MuJoCo 交互式可视化窗口

    参数:
        model_path: MJCF 模型文件路径
    """
    print("\n" + "=" * 60)
    print("  启动 MuJoCo 交互式可视化")
    print("=" * 60)

    # 加载模型
    print(f"  加载模型: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    print(f"  [OK] 模型加载成功")

    # 打印模型信息
    print_model_info(model)

    # 重置到第一个 keyframe
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        print(f"\n  已重置到 keyframe '{model.key(0).name}'")

    # 启动 viewer
    if not VIEWER_AVAILABLE:
        print("  [WARNING] mujoco.viewer 不可用（可能缺少 GLFW 显示环境）")
        print("  在 WSL2 中需要配置 X11 转发或使用 WSLg")
        print("  解决方案:")
        print("    1. 确保已安装 libglfw3: sudo apt install libglfw3")
        print("    2. WSL2 用户确保 WSLg 已启用 (Windows 11 默认启用)")
        print("    3. 或使用 X11 转发: export DISPLAY=:0")
        return False

    try:
        print("\n  启动 MuJoCo viewer...")
        print("  操作说明:")
        print("    - 鼠标左键拖动: 旋转视角")
        print("    - 鼠标右键拖动: 平移视角")
        print("    - 滚轮: 缩放")
        print("    - 双击物体: 施加力")
        print("    - Ctrl+双击: 施加力矩")
        print("    - 按 Space: 暂停/继续物理仿真")
        print("    - 按 R: 重置仿真")
        print("    - 按 C: 切换坐标系显示")
        print("    - 按 J: 切换关节轴显示")
        print("    - 按 G: 切换接触点显示")
        print("    - 按 T: 切换透明度")
        print("    - 按 L: 切换标签显示")
        print("    - 关闭窗口退出\n")

        mujoco.viewer.launch(model, data)
    except Exception as e:
        print(f"  [WARNING] 可视化启动失败: {e}")
        print("  可能原因: 缺少显示环境 (DISPLAY 环境变量)")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="JAKA Zu5 MuJoCo 可视化启动脚本")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"MJCF 模型文件路径 (默认: 修正后的模型)")
    parser.add_argument("--original", action="store_true",
                        help="使用原始模型 (未修正质量参数)")
    parser.add_argument("--info-only", action="store_true",
                        help="仅打印模型信息, 不启动 viewer")
    args = parser.parse_args()

    # 选择模型
    if args.original:
        model_path = ORIGINAL_MODEL
    else:
        model_path = args.model
        if not os.path.exists(model_path):
            print(f"[WARN] 修正模型不存在, 回退到原始模型")
            model_path = ORIGINAL_MODEL

    if not os.path.exists(model_path):
        print(f"[ERROR] 模型文件不存在: {model_path}")
        sys.exit(1)

    print("=" * 60)
    print("  JAKA Zu5 机械臂 MuJoCo 可视化")
    print("=" * 60)
    print(f"  模型文件: {model_path}")

    if args.info_only:
        model = mujoco.MjModel.from_xml_path(model_path)
        print_model_info(model)
        return

    # 启动 viewer
    success = launch_viewer(model_path)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
