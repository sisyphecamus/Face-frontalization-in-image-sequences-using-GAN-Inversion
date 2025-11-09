import os
import re
from pathlib import Path

print("=" * 70)
print("搜索所有预处理相关代码")
print("=" * 70)

# 搜索 transforms 和 Normalize 的使用
patterns = [
    (r'transforms\.Compose', '使用 Compose'),
    (r'transforms\.Resize', '调整大小'),
    (r'transforms\.Normalize', '归一化'),
    (r'transforms\.ToTensor', '转换为张量'),
    (r'\.transform\(', '应用 transform'),
    (r'def __getitem__', '__getitem__ 方法'),
]

# 遍历项目文件
for root, dirs, files in os.walk('.'):
    # 跳过不需要的目录
    dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.venv', 'venv']]
    
    for file in files:
        if not file.endswith('.py'):
            continue
        
        filepath = os.path.join(root, file)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
                for pattern, description in patterns:
                    if re.search(pattern, content):
                        # 计算行号
                        lines = content.split('\n')
                        for i, line in enumerate(lines, 1):
                            if re.search(pattern, line):
                                print(f"\n【{description}】")
                                print(f"  文件: {filepath}")
                                print(f"  行号: {i}")
                                print(f"  内容: {line.strip()[:80]}")
                                break
        except:
            pass

print("\n" + "=" * 70)