from pathlib import Path
import pandas as pd
import re
import sys


def sanitize_sheet_name(name: str) -> str:
    # Excel sheet name max 31 chars and cannot contain: : \/ ? * [ ]
    name = re.sub(r'[:\\/?*\[\]]', '_', name)
    return name[:31]


def find_excel_files(folder: Path):
    exts = ('.xlsx', '.xls', '.xlsm', '.xlsb')
    files = [p for p in folder.rglob('*') if p.suffix.lower() in exts]
    return sorted(files)


def read_first_sheet(path: Path) -> pd.DataFrame:
    try:
        # Let pandas choose engine; prefer openpyxl for xlsx
        return pd.read_excel(path)
    except Exception as e:
        # Last resort: try with openpyxl explicitly
        try:
            return pd.read_excel(path, engine='openpyxl')
        except Exception:
            raise RuntimeError(f'无法读取 Excel: {path}，错误: {e}')


def merge_city_excels(source_root: Path, output_file: Path):
    if not source_root.exists() or not source_root.is_dir():
        raise FileNotFoundError(f'找不到目录: {source_root}')

    any_written = False

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for city_dir in sorted([p for p in source_root.iterdir() if p.is_dir()], key=lambda x: x.name):
            files = find_excel_files(city_dir)
            if not files:
                print(f'警告：{city_dir.name} 下未找到 Excel 文件，跳过。')
                continue

            dfs = []
            for f in files:
                try:
                    df = read_first_sheet(f)
                    # optionally add source column
                    df['_source_file'] = f.name
                    dfs.append(df)
                except Exception as e:
                    print(f'读取文件失败: {f}，跳过。错误: {e}')

            if not dfs:
                print(f'警告：{city_dir.name} 下所有 Excel 均无法读取，跳过。')
                continue

            merged = pd.concat(dfs, ignore_index=True)
            sheet_name = sanitize_sheet_name(city_dir.name)
            merged.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f'已写入城市: {city_dir.name} ({len(dfs)} 个文件，共 {len(merged)} 行)')
            any_written = True

    if any_written:
        print(f'合并完成，输出文件: {output_file}')
    else:
        # 如果没有写入，尝试删除可能创建的空文件
        try:
            if output_file.exists() and output_file.stat().st_size == 0:
                output_file.unlink()
        except Exception:
            pass
        print('未写入任何数据，未生成输出文件。')


def main():
    root = Path(__file__).parent
    source = root / '各城市迭代结果'
    output = root / '各城市迭代结果.xlsx'

    # 支持命令行覆盖
    if len(sys.argv) >= 2:
        source = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        output = Path(sys.argv[2])

    merge_city_excels(source, output)


if __name__ == '__main__':
    main()
