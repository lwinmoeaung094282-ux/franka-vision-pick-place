@echo off
REM Launch Franka pick-and-place in Isaac Sim 5.1
REM
REM Usage examples:
REM   launch.bat                                        -- prompts for instruction interactively
REM   launch.bat --no-vision                            -- same, no camera detection
REM   launch.bat --command "pick the blue pillar first" -- pass instruction directly
REM   launch.bat --command "skip the yellow block" --no-vision
REM   launch.bat --command "tallest first"
REM   launch.bat --headless --no-vision                 -- fastest headless run
REM   launch.bat --episodes 3
REM
REM Pass Anthropic API key directly (most reliable — avoids env var propagation issues):
REM   launch.bat --command "pick blue first" --api-key sk-ant-YOUR_KEY_HERE --no-vision
REM
REM Or set env var before running (may not propagate through python.bat):
REM   set ANTHROPIC_API_KEY=sk-ant-...

set PYTHONPATH=C:\Users\Owner\miniconda3\envs\env_isaaclab\Lib\site-packages
C:\isaacsim\python.bat \\wsl$\Ubuntu\home\lwin\isaac_pick_place\run.py %*
