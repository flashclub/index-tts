"""
在 Colab 中修复 index_tts_api.py 的脚本
添加 flush=True 参数到所有 print 语句
"""

import re
from pathlib import Path

def add_flush_to_prints(file_path):
    """为所有 print 语句添加 flush=True 参数"""
    
    print(f"正在修复文件: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 备份原文件
    backup_path = file_path.with_suffix('.py.backup')
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ 已创建备份: {backup_path}")
    
    # 使用正则表达式查找所有 print(...) 并添加 flush=True
    # 匹配 print("...") 或 print(f"...") 等，但不匹配已经有 flush=True 的
    
    def replace_print(match):
        original = match.group(0)
        # 如果已经包含 flush，则不修改
        if 'flush=' in original:
            return original
        
        # 在右括号前添加 , flush=True
        # 处理多行和单行的情况
        if original.endswith(')'):
            return original[:-1] + ', flush=True)'
        return original
    
    # 匹配 print(...)，包括多行的情况
    # 这个正则表达式会匹配 print 语句，处理嵌套括号
    pattern = r'print\([^)]*\)'
    
    # 简单的替换策略：逐行处理
    lines = content.split('\n')
    modified_lines = []
    
    for line in lines:
        # 只处理包含 print 但不包含 flush= 的行
        if 'print(' in line and 'flush=' not in line and not line.strip().startswith('#'):
            # 在行末的 ) 之前添加 , flush=True
            # 处理 print(f"...") 和 print("...") 等情况
            modified_line = re.sub(r'\)(\s*)$', r', flush=True)\1', line)
            modified_lines.append(modified_line)
        else:
            modified_lines.append(line)
    
    modified_content = '\n'.join(modified_lines)
    
    # 写入修改后的内容
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    # 统计修改数量
    original_print_count = content.count('print(')
    modified_print_count = modified_content.count('flush=True')
    
    print(f"✅ 修复完成!")
    print(f"   - 原文件包含 {original_print_count} 个 print 语句")
    print(f"   - 已为 {modified_print_count} 个 print 语句添加 flush=True")
    print(f"   - 修改后的文件已保存")

if __name__ == "__main__":
    # 在 Colab 中使用
    api_file = Path("/content/index-tts/index_tts_api.py")
    
    if not api_file.exists():
        print(f"❌ 文件不存在: {api_file}")
        print("   请确保已经克隆了仓库并且路径正确")
    else:
        add_flush_to_prints(api_file)
        print("\n✅ 全部完成! 现在可以启动 API 服务了")
