@echo off
REM Launch Franka pick-and-place in Isaac Sim 5.1
REM Usage:  launch.bat [--headless] [--no-vision] [--episodes N]

set PYTHONPATH=C:\Users\Owner\miniconda3\envs\env_isaaclab\Lib\site-packages
C:\isaacsim\python.bat \\wsl$\Ubuntu\home\lwin\isaac_pick_place\run.py %*
