# GO2 网页源码

日常使用和部署见[项目入口](../docs/go2/README.md)与[板载使用](../docs/go2/BOARD_WEB_GUIDE.md)，本文件只说明开发。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| static/index.html、style.css、app.js | 浅色单屏工作台、实验回放、参数页 |
| server.py | HTTP、模式、参数校验、导出与接口 |
| camera.py | 本地USB/备用SSH采集、配对帧及预览 |
| wired.py | 状态桥、标定、真实影子/受限执行、心跳和采样 |
| access.py | 备用登录和Host/同源保护；当前板载免登录 |
| ../deploy/ | 部署器和开机服务配置，与网页实现分开 |
| test_*.py | 离线接口/状态/相机/权限验证 |

板载`--board --autoconnect --camera-local ... --interface eth0`：启用真实接口，启动只接输入。电脑备用`--wired`：在电脑感知规划。无实机模式参数：历史数据和流程模拟。默认模拟不执行FMM/MPPI，不代表实机结果；data/console.sqlite3仅保存本地参数及模拟记录。不要同时启动两条USB采集路径。

## 主要接口

- GET /api/state、/api/camera/preview：状态、同帧图像。
- POST /api/real/connect、disconnect：连接输入。
- POST /api/real/calibration/save、fit、approve：外参与本次地面复核。
- POST /api/real/start、stop、heartbeat：真实影子/受限执行与心跳；实际执行需显式现场确认。
- POST /api/camera/capture：同帧截图；GET /api/camera/captures/{name}.zip：板载截图下载。
- GET /api/experiments、/{id}、/{id}/export：历史、回放与导出。
- /api/tasks/*：离线流程模拟，不能当真实机器人任务。

## 本机开发与测试

```bash
cd /home/lenovo/HIMLoco
bash go2_runtime/manage.sh web
/home/lenovo/miniconda3/envs/isclab/bin/python -m unittest discover -s go2_web -p 'test_*.py'
```

前端无Node构建依赖。保留data目录和历史原始日志，勿删除标定或把历史截图伪装成当前实时数据。旧只读部署器、旧服务单元与重复接入教程已删除，完整部署统一走 ../deploy/deploy.sh。
