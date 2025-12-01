#!/bin/bash
TARGET_DIR="./data/res/jpg"

cd "$TARGET_DIR" || exit 1

echo "开始解压 $TARGET_DIR 目录下的所有.tar.gz文件..."

find . -name "*.tar.gz" -type f | while read -r tar_file; do
    filename=$(basename "$tar_file")
    echo "正在解压: $filename"
    if tar -xzf "$tar_file"; then
        echo "解压成功: $filename"

        rm "$tar_file"
        echo "已删除: $filename"
    else
        echo "解压失败: $filename"
    fi
    echo "------------------------"
done

echo "所有文件解压完成！"