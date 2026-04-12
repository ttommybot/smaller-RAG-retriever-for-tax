@echo off

rem RAG对话系统启动脚本
echo ====================================
echo        RAG对话系统启动器
 echo ====================================

rem 检查Python是否安装
echo 检查Python环境...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误：Python未安装，请先安装Python 3.7或更高版本
    pause
    exit /b 1
)

rem 检查依赖是否安装
echo 检查Flask依赖...
python -m pip list | findstr "Flask" >nul 2>&1
if %errorlevel% neq 0 (
    echo 安装Flask依赖...
    python -m pip install flask flask-cors
    if %errorlevel% neq 0 (
        echo 错误：依赖安装失败
        pause
        exit /b 1
    )
)

rem 启动服务器
echo 启动RAG对话系统服务器...
rem 不使用/B参数，以便查看服务器启动日志
start "RAG对话系统服务器" python app.py

rem 等待服务器启动
echo 等待服务器启动...
ping localhost -n 3 >nul

rem 打开浏览器
echo 打开浏览器...
start http://127.0.0.1:5000

echo ====================================
echo 启动成功！
echo 系统已在 http://127.0.0.1:5000 运行
echo 请在浏览器中查看和使用
 echo ====================================

rem 保持窗口打开，以便查看日志
echo 按任意键关闭此窗口...
pause >nul