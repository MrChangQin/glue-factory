#!/bin/bash
# 脚本名称：download_revisitop1m.sh
# 功能：多线程+防阻塞 下载 revisitop1m.1-100.tar.gz
# 优化点：每个文件独立会话+强制超时+断开后重试+防限流延迟

# 核心配置
REMOTE_BASE_PATH="/revisitop1m/jpg"
LOCAL_BASE_PATH="/hy-tmp/glue-factory/data/revisitop1m/jpg"
START_NUM=1
END_NUM=100
LOG_FILE="./download_revisitop1m.log"
MAX_THREADS=1  # 降低并行数，减少限流概率（建议3-5）
TIMEOUT=300    # 单个文件下载超时（秒），超时自动中断
RETRY=2        # 单个文件失败重试次数
DELAY=2        # 每个任务启动前的短延迟，避免请求扎堆

# 初始化
mkdir -p "${LOCAL_BASE_PATH}" || { echo "错误：无法创建本地目录"; exit 1; }
> "${LOG_FILE}"

# 带重试+超时的下载函数
download_file() {
    local num=$1
    local FILE_NAME="revisitop1m.${num}.tar.gz"
    local REMOTE_FILE="${REMOTE_BASE_PATH}/${FILE_NAME}"
    local LOCAL_FILE="${LOCAL_BASE_PATH}/${FILE_NAME}"
    local retry_count=0

    # 重试逻辑
    while (( retry_count < RETRY )); do
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] 开始下载(${retry_count+1}/${RETRY})：${FILE_NAME}"
        
        # 核心：用timeout强制超时，避免永久阻塞；--kill-after确保彻底终止
        timeout --kill-after=10 ${TIMEOUT} gpushare-cli baidu down "${REMOTE_FILE}" "${LOCAL_FILE}"
        
        local exit_code=$?
        if (( exit_code == 0 )); then
            # 下载成功
            echo "[$(date +'%Y-%m-%d %H:%M:%S')] 成功：${FILE_NAME}" >> "${LOG_FILE}"
            return 0
        elif (( exit_code == 124 || exit_code == 137 )); then
            # 超时/被kill，重试
            echo "[$(date +'%Y-%m-%d %H:%M:%S')] 超时/阻塞：${FILE_NAME}，重试..."
            retry_count=$((retry_count + 1))
            sleep ${DELAY}  # 重试前延迟，重置连接
        else
            # 其他错误，记录并退出重试
            echo "[$(date +'%Y-%m-%d %H:%M:%S')] 失败：${FILE_NAME} (错误码：${exit_code})" >> "${LOG_FILE}"
            return 1
        fi
    done

    # 重试次数用尽
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] 失败：${FILE_NAME} (重试${RETRY}次仍失败)" >> "${LOG_FILE}"
    return 1
}

# 多线程调度（降低并行数，避免限流）
for (( num=START_NUM; num<=END_NUM; num++ )); do
    sleep ${DELAY}  # 每个任务启动前延迟，避免请求扎堆触发限流
    download_file "${num}" &
    
    # 严格控制并行数
    while (( $(jobs -p | wc -l) >= MAX_THREADS )); do
        wait -n
    done
done

# 等待所有任务完成
wait

# 最终统计
echo "========================================"
echo "所有下载任务执行完毕！"
echo "日志文件：${LOG_FILE}"
echo "成功数：$(grep -c "成功" ${LOG_FILE})"
echo "失败数：$(grep -c "失败" ${LOG_FILE})"
echo "========================================"