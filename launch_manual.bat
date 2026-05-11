@echo off
REM Manual Franka control — for debugging grasp issues
REM GUI mode only (no --headless)

set PYTHONPATH=C:\Users\Owner\miniconda3\envs\env_isaaclab\Lib\site-packages
C:\isaacsim\python.bat \\wsl$\Ubuntu\home\lwin\isaac_pick_place\manual_control.py
