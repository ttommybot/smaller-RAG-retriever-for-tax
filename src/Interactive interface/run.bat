@echo off
chcp 65001 >nul
cls
echo ====================================
echo          RAG对话系统启动器
echo ====================================
echo 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误：未检测到Python，请先安装Python
    pause
    exit
)

echo 检查Flask依赖...
python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo 错误：缺少Flask依赖，请执行：pip install flask
    pause
    exit
)

echo 启动对话系统服务器...
start http://127.0.0.1:5000
python app.py

echo 服务器已关闭
pause