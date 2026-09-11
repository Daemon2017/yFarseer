import os
import numpy as np
import pandas as pd
import config
import classes
import utils


def rebuild_multicopies_to_base(row, base_col, expected_len):
    suffixes = ['a', 'b', 'c', 'd'][:expected_len]
    alleles = []
    for suf in suffixes:
        val = row.get(f"{base_col}{suf}")
        if pd.notna(val) and val >= 0:
            alleles.append(int(round(val)))
    if not alleles:
        return None
    alleles.sort()
    return "-".join(map(str, alleles))


def main():
    print("1. Загрузка полных образцов и приведение к каноническим именам...")
    df_extended = utils.load_and_transform_dataset(only_complete=True)
    print("\n2. Инициализация топологии дерева для воссоздания иерархии...")
    topo_manager = classes.HierarchyTopologyManager()
    topo_manager.load_topology(df_extended['Haplogroup'].unique().tolist())
    all_snps = topo_manager.all_snps
    parent_indices = topo_manager.parent_indices
    children_map = {i: [] for i in range(-1, len(all_snps))}
    for child_idx, p_idx in enumerate(parent_indices):
        children_map[p_idx].append(child_idx)
    levels = [-1] * len(all_snps)
    for i in range(len(all_snps)):
        path_len = 0
        curr = parent_indices[i]
        while curr != -1:
            path_len += 1
            curr = parent_indices[curr]
        levels[i] = path_len
    max_level = max(levels) if levels else 0
    print(f"Максимальная глубина дерева: {max_level}")
    print("\n3. Идентификация терминальных (листовых) и внутренних SNP...")
    terminal_snps = [all_snps[idx] for idx, siblings in children_map.items() if idx != -1 and len(siblings) == 0]
    internal_snps = [all_snps[idx] for idx, siblings in children_map.items() if idx != -1 and len(siblings) > 0]
    print(f"Всего SNP в дереве: {len(all_snps)}")
    print(f"Из них терминальных (листьев): {len(terminal_snps)}")
    print(f"Из них внутренних (предковых ветвей): {len(internal_snps)}")
    modal_registry = pd.DataFrame(index=all_snps, columns=config.EXTENDED_STR_COLS, dtype=float)
    print("\n4. Векторизованный расчет модальных гаплотипов для ТЕРМИНАЛЬНЫХ SNP...")
    df_extended['Canonical_Haplogroup'] = df_extended['Haplogroup'].astype(str).str.strip().map(topo_manager.synonym_to_snp).fillna(df_extended['Haplogroup'].astype(str).str.strip())
    df_terminal_only = df_extended[df_extended['Canonical_Haplogroup'].isin(terminal_snps)]
    if not df_terminal_only.empty:
        raw_modes = df_terminal_only.groupby('Canonical_Haplogroup')[config.EXTENDED_STR_COLS].agg(lambda x: x.mode().iloc[0] if not x.dropna().empty else np.nan)
        modal_registry.loc[raw_modes.index, config.EXTENDED_STR_COLS] = raw_modes
    print("\n5. Быстрая филогенетическая реконструкция внутренних SNP снизу вверх...")
    for current_lvl in range(max_level, -1, -1):
        lvl_indices = [idx for idx, l in enumerate(levels) if l == current_lvl]
        for idx in lvl_indices:
            snp_name = all_snps[idx]
            if snp_name in internal_snps:
                children_idxs = children_map[idx]
                children_names = [all_snps[c_idx] for c_idx in children_idxs]
                children_modal_data = modal_registry.loc[children_names]
                if not children_modal_data.dropna(how='all').empty:
                    computed_modes = children_modal_data.mode(dropna=True)
                    if not computed_modes.empty:
                        modal_registry.loc[snp_name, config.EXTENDED_STR_COLS] = computed_modes.iloc[0]
    print("\n6. Конвертация EXTENDED_STR_COLS обратно в BASE_STR_COLS...")
    base_modal_data = []
    for snp_name, row in modal_registry.iterrows():
        base_row = {'Haplogroup': snp_name}
        for base_col in config.BASE_STR_COLS:
            if base_col in config.MULTICOPIES:
                base_row[base_col] = rebuild_multicopies_to_base(row, base_col, config.MULTICOPIES[base_col])
            else:
                val = row.get(base_col)
                base_row[base_col] = int(round(val)) if pd.notna(val) and val >= 0 else None
        base_modal_data.append(base_row)
    df_modal_base = pd.DataFrame(base_modal_data)
    df_modal_base = df_modal_base.dropna(subset=config.BASE_STR_COLS, how='any')
    print("\n7. Сохранение результата в файл...")
    df_modal_base = df_modal_base[config.BASE_STR_COLS + ['Haplogroup']]
    df_modal_base.to_csv("modal_haplotypes.csv", index=False, encoding='utf-8')
    print("Генерация успешно завершена!")


if __name__ == '__main__':
    main()
