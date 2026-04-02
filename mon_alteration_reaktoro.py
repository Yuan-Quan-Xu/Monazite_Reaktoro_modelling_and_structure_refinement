%matplotlib inline
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
from reaktoro import *
from tqdm import tqdm

def check_species_existence(db, species_names, species_type="species"):
    """
    检查物种名称是否在数据库中存在，返回存在的名称列表，并打印缺失警告。
    species_type: "species" 或 "mineral"

    注意：对于Phreeqc数据库，矿物相也包含在db.species()中。
    """
    # 获取数据库中所有物种名称
    # 对于Phreeqc数据库，无论是水溶液物种还是矿物相，都在db.species()中
    db_names = {s.name() for s in db.species()}

    existing = []
    missing = []
    for name in species_names:
        if name in db_names:
            existing.append(name)
        else:
            missing.append(name)
    if missing:
        print(f"警告: 以下 {species_type} 在数据库中不存在，将被忽略: {missing}")
    return existing


def plot_phase_diagram(combo_grid, PH_VALUES, LOG_F_VALUES, T, P, existing_minerals):
    """
    绘制单个 T-P 条件下的相图，并保存为文件。
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = ListedColormap(['white', 'blue', 'green', 'cyan', 'orange', 'purple', 'yellow', 'black'])
    plot_grid = np.where(combo_grid == -1, np.nan, combo_grid)
    X, Y = np.meshgrid(LOG_F_VALUES, PH_VALUES)
    c = ax.pcolormesh(X, Y, plot_grid, cmap=cmap, shading='auto', vmin=0, vmax=7)

    # 标注推断流体范围（示例）
    rect = patches.Rectangle(
        (-4.0, 7.0), 2.0, 2.5,
        linewidth=1.5, edgecolor='red', facecolor='red', alpha=0.2, linestyle='--'
    )
    ax.add_patch(rect)

    ax.set_xlabel(r'$\log a_{F^-}$')
    ax.set_ylabel('pH')
    ax.set_title(f'T={T}°C, P={P} bar')

    # 添加图例
    legend_text = []
    if "Monazite_Ce" in existing_minerals:
        legend_text.append("Monazite_Ce (blue)")
    if "Fluorapatite" in existing_minerals:
        legend_text.append("Fluorapatite (green)")
    if "Fluorite" in existing_minerals:
        legend_text.append("Fluorite (cyan)")
    if legend_text:
        ax.legend(legend_text, loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(f'stability_T{T}_P{P}.jpg', dpi=300, format='jpg', bbox_inches='tight')
    plt.savefig(f'stability_T{T}_P{P}.pdf', dpi=300, format='pdf', bbox_inches='tight')
    plt.show()
    plt.close()


def run_thermodynamic_simulation():
    # 1. 数据库与物相定义自适应加载
    db_file = r"D:\Star-Sky\Documents\PythonProject\Jupyter Notebook Files\Monazite_BayanObo\custom_llnl_monazite.dat"
    if not os.path.exists(db_file):
        raise FileNotFoundError("数据库文件未找到！")

    # 加载自定义 PHREEQC 数据库
    db = PhreeqcDatabase.fromFile(db_file)

    # 定义需要的水溶液物种（根据数据库中的名称）
    desired_aqueous = ["H2O", "Ce+3", "PO4-3", "Ca+2", "Na+", "K+", "Cl-", "F-", "OH-", "H+"]
    existing_aqueous = check_species_existence(db, desired_aqueous, species_type="species")
    if not existing_aqueous:
        raise RuntimeError("数据库中没有任何所需的水溶液物种，请检查数据库文件。")

    aqueous_species = speciate(existing_aqueous)
    aqueous = AqueousPhase(aqueous_species)
    aqueous.setActivityModel(ActivityModelHKF())

    # 定义需要的矿物相
    desired_minerals = ["Monazite_Ce", "Fluorapatite", "Fluorite", "Bastnaesite_Ce", "Cerianite"]
    existing_minerals = check_species_existence(db, desired_minerals, species_type="mineral")
    if not existing_minerals:
        raise RuntimeError("数据库中没有任何所需的矿物相，请检查数据库文件。")
    minerals = MineralPhases(existing_minerals)

    # 构建化学系统
    system = ChemicalSystem(db, aqueous, minerals)

    # 2. 求解器与约束条件设定
    specs = EquilibriumSpecs(system)
    specs.temperature()
    specs.pressure()
    specs.pH()
    specs.activity("F-")
    specs.charge()
    specs.openTo("Cl-")

    solver = EquilibriumSolver(specs)

    # 设定模拟的温压及化学梯度参数矩阵
    T_VALUES = [25, 100, 200, 250, 300]      # 温度 (°C)
    P_VALUES = [1, 100, 500, 1000]          # 压力 (bar)
    PH_VALUES = np.linspace(4.0, 10.0, 40)
    LOG_F_VALUES = np.linspace(-6.0, 0.0, 40)

    # 存储所有条件的组合网格用于总览图
    all_combo_grids = {}

    # 3. 核心计算循环
    for T in T_VALUES:
        for P in P_VALUES:
            print(f"正在计算 T={T}°C, P={P} bar 的相平衡边界...")

            state = ChemicalState(system)
            state.set("H2O", 1.0, "kg")
            state.set("Ce+3", 0.01, "mol")
            state.set("PO4-3", 0.01, "mol")
            state.set("Ca+2", 0.01, "mol")
            state.set("Na+", 0.1, "mol")
            state.set("K+", 0.05, "mol")
            state.set("Cl-", 0.15, "mol")

            conditions = EquilibriumConditions(specs)
            conditions.temperature(T, "celsius")
            conditions.pressure(P, "bar")
            conditions.charge(0.0)

            combo_grid = np.zeros((len(PH_VALUES), len(LOG_F_VALUES)))

            for i, pH in enumerate(tqdm(PH_VALUES)):
                for j, logf in enumerate(LOG_F_VALUES):
                    conditions.pH(pH)
                    conditions.activity("F-", 10**logf)

                    result = solver.solve(state, conditions)

                    if not result.optima.succeeded:
                        combo_grid[i, j] = -1
                        continue

                    # 固相平衡判定（只针对实际存在的矿物）
                    code = 0
                    if "Monazite_Ce" in existing_minerals and state.speciesAmount("Monazite_Ce") > 1e-6:
                        code |= 1
                    if "Fluorapatite" in existing_minerals and state.speciesAmount("Fluorapatite") > 1e-6:
                        code |= 2
                    if "Fluorite" in existing_minerals and state.speciesAmount("Fluorite") > 1e-6:
                        code |= 4
                    # 可根据需要增加更多矿物，但需更新 code 位

                    combo_grid[i, j] = code

            # 存储组合网格用于总览图
            all_combo_grids[(T, P)] = combo_grid.copy()

            # 生成单个条件的图（现在已实现，会保存为文件）
            plot_phase_diagram(combo_grid, PH_VALUES, LOG_F_VALUES, T, P, existing_minerals)

    # 4. 生成总览图
    plot_overview_diagram(all_combo_grids, PH_VALUES, LOG_F_VALUES, T_VALUES, P_VALUES, existing_minerals)


def plot_overview_diagram(all_combo_grids, ph_vals, logf_vals, T_VALUES, P_VALUES, existing_minerals):
    """
    生成所有温压条件下的总览图
    """
    # 创建子图网格：行数为温度数量，列数为压力数量
    n_rows = len(T_VALUES)
    n_cols = len(P_VALUES)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), sharex=True, sharey=True)

    # 如果只有一行或一列，确保axes是二维数组
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    cmap = ListedColormap(['white', 'blue', 'green', 'cyan', 'orange', 'purple', 'yellow', 'black'])

    # 为每个温压条件绘制子图
    for i, T in enumerate(T_VALUES):
        for j, P in enumerate(P_VALUES):
            ax = axes[i, j]
            combo_grid = all_combo_grids.get((T, P))

            if combo_grid is not None:
                plot_grid = np.where(combo_grid == -1, np.nan, combo_grid)
                X, Y = np.meshgrid(logf_vals, ph_vals)
                c = ax.pcolormesh(X, Y, plot_grid, cmap=cmap, shading='auto', vmin=0, vmax=7)

                # 标注推断流体范围
                rect = patches.Rectangle(
                    (-4.0, 7.0), 2.0, 2.5,
                    linewidth=1.5, edgecolor='red', facecolor='red', alpha=0.2, linestyle='--'
                )
                ax.add_patch(rect)

            # 设置标题和标签
            ax.set_title(f'T={T}°C, P={P} bar', fontsize=10)

            # 只在最左侧子图显示y轴标签
            if j == 0:
                ax.set_ylabel('pH', fontsize=10)

            # 只在最底部子图显示x轴标签
            if i == n_rows - 1:
                ax.set_xlabel(r'$\log a_{F^-}$', fontsize=10)

    # 添加整体标题
    plt.suptitle('Mineral Stability Phase Diagrams at Different T-P Conditions', fontsize=14, y=1.02)

    # 添加图例说明
    legend_text = []
    if "Monazite_Ce" in existing_minerals:
        legend_text.append("Monazite_Ce (blue)")
    if "Fluorapatite" in existing_minerals:
        legend_text.append("Fluorapatite (green)")
    if "Fluorite" in existing_minerals:
        legend_text.append("Fluorite (cyan)")

    if legend_text:
        # 将图例放在图外
        fig.legend(legend_text, loc='upper center', bbox_to_anchor=(0.5, 0.98),
                  ncol=len(legend_text), fontsize=9, framealpha=0.8)

    plt.tight_layout()

    # 保存总览图
    plt.savefig('stability_overview_all_conditions.jpg', dpi=300, format='jpg', bbox_inches='tight')
    plt.savefig('stability_overview_all_conditions.pdf', dpi=300, format='pdf', bbox_inches='tight')

    # 显示总览图
    plt.show()
    plt.close()

    print("总览图已保存为 'stability_overview_all_conditions.jpg' 和 'stability_overview_all_conditions.pdf'")


# 直接运行主函数
run_thermodynamic_simulation()