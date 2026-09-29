import time
from pathlib import Path

import uno
from com.sun.star.awt.FontWeight import BOLD, NORMAL
from com.sun.star.beans import PropertyValue
from com.sun.star.style.ParagraphAdjust import BLOCK, CENTER, LEFT
from com.sun.star.text.ControlCharacter import PARAGRAPH_BREAK
from com.sun.star.text.TextContentAnchorType import AS_CHARACTER


SOURCE = Path("/home/hzz/project/HIMLoco/tmp/midterm_opening_report/source/中期报告_工作副本.docx")
OUTPUT = Path("/home/hzz/桌面/中期报告_实验方案及后续修订版.docx")
FIG_ROOT = Path(
    "/home/hzz/project/HIMLoco/assets/runs/navigation_benchmark/ch3_paper_figures_20260925"
)


def prop(name, value):
    item = PropertyValue()
    item.Name = name
    item.Value = value
    return item


def connect():
    local_ctx = uno.getComponentContext()
    resolver = local_ctx.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver", local_ctx
    )
    for _ in range(30):
        try:
            ctx = resolver.resolve(
                "uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext"
            )
            desktop = ctx.ServiceManager.createInstanceWithContext(
                "com.sun.star.frame.Desktop", ctx
            )
            return ctx, desktop
        except Exception:
            time.sleep(0.2)
    raise RuntimeError("Unable to connect to LibreOffice")


def set_text_style(cursor, role):
    cursor.NumberingStyleName = ""
    cursor.CharFontName = "宋体"
    cursor.CharFontNameAsian = "宋体"
    cursor.CharPosture = 0
    cursor.ParaTopMargin = 0
    cursor.ParaBottomMargin = 120
    cursor.ParaLeftMargin = 0
    cursor.ParaRightMargin = 0
    line_spacing = uno.createUnoStruct("com.sun.star.style.LineSpacing")
    line_spacing.Mode = 0
    line_spacing.Height = 150
    cursor.ParaLineSpacing = line_spacing
    if role == "title":
        cursor.CharFontName = "黑体"
        cursor.CharFontNameAsian = "黑体"
        cursor.CharHeight = 16
        cursor.CharWeight = BOLD
        cursor.ParaAdjust = LEFT
        cursor.ParaFirstLineIndent = 0
        cursor.ParaTopMargin = 240
        cursor.ParaBottomMargin = 180
    elif role == "heading":
        cursor.CharFontName = "黑体"
        cursor.CharFontNameAsian = "黑体"
        cursor.CharHeight = 14
        cursor.CharWeight = BOLD
        cursor.ParaAdjust = LEFT
        cursor.ParaFirstLineIndent = 0
        cursor.ParaTopMargin = 180
        cursor.ParaBottomMargin = 120
    elif role == "subheading":
        cursor.CharFontName = "黑体"
        cursor.CharFontNameAsian = "黑体"
        cursor.CharHeight = 12
        cursor.CharWeight = BOLD
        cursor.ParaAdjust = LEFT
        cursor.ParaFirstLineIndent = 0
        cursor.ParaTopMargin = 120
    elif role == "caption":
        cursor.CharHeight = 10.5
        cursor.CharWeight = NORMAL
        cursor.ParaAdjust = CENTER
        cursor.ParaFirstLineIndent = 0
        cursor.ParaBottomMargin = 180
    elif role == "list":
        cursor.CharHeight = 12
        cursor.CharWeight = NORMAL
        cursor.ParaAdjust = BLOCK
        cursor.ParaFirstLineIndent = 0
    else:
        cursor.CharHeight = 12
        cursor.CharWeight = NORMAL
        cursor.ParaAdjust = BLOCK
        cursor.ParaFirstLineIndent = 740


def add_paragraph(text, role="body"):
    cursor.gotoEnd(False)
    set_text_style(cursor, role)
    doc.Text.insertString(cursor, text, False)
    doc.Text.insertControlCharacter(cursor, PARAGRAPH_BREAK, False)


def add_image(filename, caption, width_cm=15.4):
    path = FIG_ROOT / filename
    if not path.exists():
        raise FileNotFoundError(path)
    cursor.gotoEnd(False)
    set_text_style(cursor, "caption")
    graphic = doc.createInstance("com.sun.star.text.TextGraphicObject")
    graphic.GraphicURL = uno.systemPathToFileUrl(str(path))
    graphic.AnchorType = AS_CHARACTER
    graphic.Width = int(width_cm * 1000)
    # Most paper figures are landscape; this height keeps each with its caption.
    graphic.Height = int(width_cm * 0.58 * 1000)
    doc.Text.insertTextContent(cursor, graphic, False)
    doc.Text.insertControlCharacter(cursor, PARAGRAPH_BREAK, False)
    add_paragraph(caption, "caption")


def add_table(headers, rows, font_size=9.0):
    cursor.gotoEnd(False)
    table = doc.createInstance("com.sun.star.text.TextTable")
    table.initialize(len(rows) + 1, len(headers))
    doc.Text.insertTextContent(cursor, table, False)
    for row_idx, row in enumerate([headers] + rows):
        for col_idx, value in enumerate(row):
            name = chr(ord("A") + col_idx) + str(row_idx + 1)
            cell = table.getCellByName(name)
            cell.String = str(value)
            tc = cell.createTextCursor()
            tc.CharFontName = "宋体"
            tc.CharFontNameAsian = "宋体"
            tc.CharHeight = font_size
            tc.CharWeight = BOLD if row_idx == 0 else NORMAL
            tc.ParaAdjust = CENTER
            if row_idx == 0:
                cell.BackColor = 0xD9EAF7
    cursor.gotoEnd(False)
    doc.Text.insertControlCharacter(cursor, PARAGRAPH_BREAK, False)


ctx, desktop = connect()
doc = desktop.loadComponentFromURL(
    uno.systemPathToFileUrl(str(SOURCE)),
    "_blank",
    0,
    (prop("Hidden", True), prop("ReadOnly", False)),
)
if doc is None:
    raise RuntimeError("Failed to load working document")

# Remove the copied reference content from this heading to the end.
target = None
enum = doc.Text.createEnumeration()
while enum.hasMoreElements():
    item = enum.nextElement()
    if item.supportsService("com.sun.star.text.Paragraph"):
        if item.String.strip() == "制定的实验方案和技术路线":
            target = item
            break
if target is None:
    raise RuntimeError("Section start not found")
cursor = doc.Text.createTextCursorByRange(target.Start)
cursor.gotoEnd(True)
cursor.String = ""
cursor = doc.Text.createTextCursorByRange(doc.Text.End)

add_paragraph("制定的实验方案和技术路线", "title")
add_paragraph("实验方案", "heading")
add_paragraph(
    "本课题面向无先验地图的野外复杂地形，采用“底层运动控制验证、局部视觉感知与规划、位姿鲁棒性分析、视觉可通行性扩展和实物系统集成”的实验方案。硬件平台为Unitree Go2四足机器人和Intel RealSense D435i相机，仿真平台采用Isaac Gym。系统不直接训练由图像到关节动作的完全端到端策略，而是将环境感知、局部规划和强化学习运动控制分层实现，使各模块能够分别评价，并在后续实验中逐步组合。实验方案如图6所示。"
)
add_paragraph(
    "底层运动控制：使用前期完成的HIMLoco强化学习策略接收期望线速度和角速度，并输出Go2十二个关节的控制量。通过与RAM方法的速度跟踪结果比较，验证该策略在离散高度地形上的跟踪精度和运动稳定性。后续所有导航方法均使用同一HIMLoco策略，避免底层控制能力差异影响上层规划结果。",
    "list",
)
add_paragraph(
    "局部视觉感知：根据D435i深度图和相机内参恢复三维点云，经相机坐标系、机身坐标系和局部世界坐标系变换后生成鸟瞰高度图。系统依据点云高度差建立障碍占据栅格，并由距离变换得到障碍净空和连续安全代价，为局部规划提供统一输入。",
    "list",
)
add_paragraph(
    "规划方法消融：在相同场景、深度图、目标点、位姿输入和底层策略下，对固定膨胀FMM、双层代价FMM以及双层代价FMM引导的足迹MPPI进行比较。通过逐步加入连续软安全代价和机身足迹轨迹预测，分析各模块对成功率、控制平滑性、障碍净空和规划耗时的影响。",
    "list",
)
add_paragraph(
    "场景分级：在同一种离散高度地形上设置简单、中等和困难三档随机障碍。简单档包含10个大石头、2棵树和100个小碎石；中等档包含15个大石头、3棵树和200个小碎石；困难档包含20个大石头、4棵树和400个小碎石。三档均使用20261000至20261029共30个固定随机种子，每个“方法与种子”的组合构成一次由随机起点到固定方向目标的完整试验。",
    "list",
)
add_paragraph(
    "位姿鲁棒性：将仿真真值定义为理想VIO，在规划输入端加入位置噪声、航向噪声、随机游走、延迟和丢帧，构造ideal、mild、medium和severe四档位姿输入。真值只用于评价最终位置误差，不反馈给局部建图和规划器，以分析定位误差对地图一致性、目标变换和到达判定的影响。",
    "list",
)
add_paragraph(
    "外部方法比较：选择iPlanner、SAPF和PS-DWA作为近年方法对比。所有方法使用相同的三档场景、固定种子、理想VIO、HIMLoco底层控制器和成功判据，只替换上层局部规划器。iPlanner使用公开预训练模型，SAPF和PS-DWA依据论文方法在本文BEV与速度接口中实现。",
    "list",
)
add_paragraph(
    "系统应用：在第三章几何局部规划的基础上，后续将RGB语义特征与深度几何特征融合，输出树木、岩石、草地、碎石和低矮台阶等区域的连续可通行代价，并将视觉可通行性结果接入足迹MPPI。最终在Go2与D435i平台上完成相机标定、VIO、ROS2数据流、速度接口和安全急停等系统集成。",
    "list",
)
add_image("07_experiment_protocol_flowchart.png", "图6 仿真实验与统计分析方案")

add_paragraph("技术路线", "heading")
add_paragraph("本课题的技术路线分为以下几个阶段，技术路线图如图7所示：")
add_paragraph(
    "运动控制准备：完成HIMLoco强化学习控制策略的训练与部署，明确本体状态、期望速度和关节动作之间的接口，利用已有的HIMLoco与RAM速度跟踪实验确定后续导航共用的底层控制器。",
    "list",
)
add_paragraph(
    "视觉几何建图：从D435i深度图恢复局部点云，经过坐标变换和地面过滤生成局部BEV高度图与障碍图。对障碍图进行距离变换，建立绝对不可通行的硬碰撞层和随距离连续衰减的软安全代价层。",
    "list",
)
add_paragraph(
    "局部运动规划：利用FMM从当前位置向局部目标传播代价场并提取参考走廊。在参考走廊附近采样短时速度序列，采用MPPI计算目标推进、路径偏差、足迹碰撞、软安全距离和控制变化等代价，输出期望线速度和角速度。",
    "list",
)
add_paragraph(
    "鲁棒性与对比验证：在简单、中等和困难三档离散高度地形中开展30个配对种子的内部消融；在中等难度下开展四档VIO噪声实验；随后使用相同协议比较iPlanner、SAPF和PS-DWA，并对成功率进行McNemar配对检验，对共同成功样本的连续指标进行Wilcoxon检验。",
    "list",
)
add_paragraph(
    "视觉能力扩展：在几何BEV上增加RGB-D语义与几何融合网络，输出连续可通行代价而不是单一二值障碍。该代价作为MPPI轨迹评价的一部分，使机器人能够区分刚性障碍、可跨越区域和具有不同风险的地表。",
    "list",
)
add_paragraph(
    "实物系统部署：在Go2上安装D435i并完成外参和时间同步标定，利用D435i图像与IMU建立VIO位姿输入，将规划速度发送给HIMLoco执行。测试由低速直线和单障碍开始，逐步扩展到随机障碍和混合地形。",
    "list",
)
add_image("05_hierarchical_navigation_flowchart.png", "图7 四足机器人分层视觉导航技术路线")

add_paragraph("完成的实验研究（论文第三章）", "title")
add_paragraph(
    "目前已完成论文第三章的主要算法实现与正式实验。第三章首先固定底层运动控制能力，再研究有限视野下的局部几何感知与运动规划。系统输入为D435i深度图、机器人位姿和局部目标，经过BEV建图、双层代价生成、FMM走廊搜索和足迹MPPI优化后输出机身速度指令，最后由HIMLoco策略生成关节控制量。整体过程如图8所示。"
)
add_image("10_rgbd_bev_cost_planning_pipeline.png", "图8 RGB-D局部感知、代价建模与轨迹规划过程")

add_paragraph("1 底层强化学习运动控制", "heading")
add_paragraph(
    "HIMLoco策略接收机身角速度、重力方向投影、期望线速度与角速度、关节位置、关节速度和历史动作等观测，输出十二自由度关节动作。训练过程中加入离散高度地形和动力学参数随机化，使策略在模型误差和不规则接触条件下仍能保持速度跟踪与姿态稳定。第三章使用前期小论文中HIMLoco与RAM的速度跟踪结果说明底层控制能力，后续所有导航算法统一使用HIMLoco，不同时改变上层规划和底层步态。"
)

add_paragraph("2 深度图与局部BEV建图", "heading")
add_paragraph(
    "对于深度图像中的像素(u,v)及深度z，根据相机内参fx、fy、cx和cy可恢复相机坐标系三维点X=(u-cx)z/fx、Y=(v-cy)z/fy、Z=z。点云经过相机外参和机器人位姿变换后投影到局部鸟瞰栅格，通过有效深度过滤、地面高度过滤和栅格高度统计得到局部高度图与障碍占据图。当前正式实验已经关闭仿真演员类别白名单，规划器只利用相机深度构建地图，并修复了场景边界墙、离散地形误检和障碍物穿模等问题。"
)

add_paragraph("3 双层代价FMM", "heading")
add_paragraph(
    "固定膨胀FMM将障碍物按统一半径扩张，再在二值栅格上计算目标到达时间场。该方法实现简单，但膨胀半径过小时容易贴近障碍，膨胀半径过大时又会封闭原本能够通过的区域。为此，本研究将安全约束分为硬碰撞层和连续软代价层：当栅格到障碍的距离小于硬安全半径时设为不可通行；在硬半径与软安全距离之间，代价按照距离连续衰减；超出软安全距离后代价为零。当前硬安全半径为0.20 m，软代价作用距离为0.65 m，固定膨胀基线半径为0.25 m。该设计使规划器在开阔区域主动远离障碍，同时保留满足机身硬尺寸约束的通行空间。"
)

add_paragraph("4 FMM引导的足迹MPPI", "heading")
add_paragraph(
    "在双层代价场之上，FMM首先生成由当前位置指向局部目标的参考走廊，MPPI围绕上一时刻控制序列采样多组线速度和角速度，并利用平面运动学模型展开未来轨迹。每条轨迹的评价包含目标距离、FMM路径偏差、硬碰撞、连续软代价、控制幅值和控制变化。Go2在平面上采用0.72 m×0.42 m矩形动态包络，每个预测位姿均检查旋转后的机身足迹，而不只检查机器人中心点。当前每轮采用384条候选轨迹和20步预测时域，规划器以12.5 Hz更新，速度命令以50 Hz下发。完整算法流程如图9所示。"
)
add_image("06_footprint_mppi_flowchart.png", "图9 双层代价FMM引导的足迹MPPI流程")
add_paragraph(
    "为减少空旷区无意义转向，局部地图中没有障碍时直接跟踪连续目标方向，检测到障碍后再启用FMM走廊和MPPI。系统同时加入转向方向保持、停滞检测和受控脱困逻辑。前期实现的世界坐标时序障碍记忆在固定种子筛选中没有获得稳定收益，并可能在位姿存在误差时产生障碍重影，因此正式方法关闭该模块，将其保留为负消融结果。"
)

add_paragraph("5 理想VIO与位姿噪声", "heading")
add_paragraph(
    "仿真中的机器人真实位姿被定义为理想VIO，而不是默认实物能够获得绝对真值。在此基础上，通过位置白噪声、航向白噪声、随机游走、固定延迟和随机丢帧构造不同误差档位。导航目标变换、局部建图和规划均使用受扰位姿，真值仅用于计算最终到达误差和位姿均方根误差。该设置能够观察定位漂移引起的地图错位、轨迹偏差和VIO误到达，并为实物阶段利用D435i实测轨迹重新标定噪声提供依据。"
)

add_paragraph("取得的实验数据及分析", "title")
add_paragraph("实验设置与评价指标", "heading")
add_paragraph(
    "正式实验均使用同一种0至0.08 m离散高度地形，机器人从随机安全起点向约24 m外的目标区域运动。三档场景使用相同的障碍生成区域、出生规则和目标规则，仅改变石头、树木和小碎石数量。每次试验最长60 s，严格成功要求机器人在规定时间内到达目标，且终止时真值距离不超过0.15 m。失败被划分为碰撞、卡住、跌倒和超时。评价指标包括成功率、95% Wilson置信区间、到达时间、路径长度、角速度指令总变差、最小净空、规划耗时、接触次数、能耗代理指标以及VIO位置和航向误差。"
)
add_image("08_isaacgym_scene_comparison.png", "图10 简单、中等和困难三档Isaac Gym场景")

add_paragraph("规划方法内部消融", "heading")
add_paragraph(
    "固定膨胀FMM、双层代价FMM和足迹MPPI在三档场景中分别运行30个配对种子，共完成270轮正式试验。所有进程均正常结束，相机不可用记录为0，同种子不同方法的障碍布局哈希完全一致。主要结果如表1所示。"
)
add_table(
    ["难度", "方法", "成功率", "成功用时/s", "平滑度", "规划/ms"],
    [
        ["简单", "固定膨胀FMM", "17/30", "42.49", "0.0241", "6.85"],
        ["简单", "双层代价FMM", "24/30", "36.23", "0.0247", "6.61"],
        ["简单", "足迹MPPI", "25/30", "34.21", "0.0131", "8.91"],
        ["中等", "固定膨胀FMM", "10/30", "45.95", "0.0267", "4.37"],
        ["中等", "双层代价FMM", "10/30", "41.97", "0.0259", "4.16"],
        ["中等", "足迹MPPI", "20/30", "41.46", "0.0127", "6.21"],
        ["困难", "固定膨胀FMM", "3/30", "50.58", "0.0263", "3.85"],
        ["困难", "双层代价FMM", "5/30", "51.82", "0.0282", "4.03"],
        ["困难", "足迹MPPI", "14/30", "43.38", "0.0136", "5.91"],
    ],
    8.5,
)
add_paragraph("表1 三档随机障碍场景的规划方法消融结果", "caption")
add_paragraph(
    "简单档中，足迹MPPI的成功率为83.3%，相对固定膨胀FMM的56.7%有所提高，但配对检验p=0.0574，尚未达到0.05显著性水平；相对双层代价FMM的80.0%没有显著差异。中等档中，足迹MPPI成功率为66.7%，相对两种FMM的33.3%均显著提高，p=0.00635。困难档中，足迹MPPI成功率为46.7%，相对固定膨胀FMM的p=0.00342，相对双层代价FMM的p=0.0352。随着障碍密度增加，未来足迹预测和短时控制优化的作用更加明显。"
)
add_paragraph(
    "三档场景中，足迹MPPI的角速度变化分别为0.0131、0.0127和0.0136，约为两种FMM的一半，说明其速度指令更加平滑。MPPI每次规划比FMM增加约2至3 ms，但仍明显低于80 ms的规划周期，能够满足当前在线运行要求。结果表明，本文方法在中高难度障碍密度下提高了导航成功率，并在三档场景中持续改善控制平滑性。"
)
add_image("01_success_rate_with_ci.png", "图11 三档场景规划成功率及95%置信区间")

add_paragraph("近年外部方法比较", "heading")
add_paragraph(
    "外部对比采用iPlanner、PS-DWA、SAPF和本文足迹MPPI，四种方法全部在同一RTX 4060 8 GB显卡上重新运行。三档各30个固定种子，每档120轮，共完成360轮。场景、理想VIO、底层HIMLoco策略、60 s时间上限以及到达、碰撞、卡住和超时判据均保持一致，外部对比只替换上层规划器。总体结果如表2所示。"
)
add_table(
    ["难度", "本文足迹MPPI", "iPlanner", "PS-DWA", "SAPF"],
    [
        ["简单", "25/30", "22/30", "13/30", "5/30"],
        ["中等", "18/30", "19/30", "13/30", "4/30"],
        ["困难", "10/30", "14/30", "6/30", "1/30"],
        ["合计", "53/90", "55/90", "32/90", "10/90"],
    ],
    9.0,
)
add_paragraph("表2 本文方法与近年局部规划方法的导航成功率", "caption")
add_paragraph(
    "iPlanner三档合计成功55次，本文方法成功53次，90组配对McNemar检验p=0.8388，二者成功率不存在显著差异。本文方法相对PS-DWA为28个种子仅本文成功、7个种子仅对方成功，p=0.0005083；相对SAPF为45比2，p=1.60×10^-11，成功率优势显著。"
)
add_paragraph(
    "本文方法全样本平均规划耗时为5.82 ms，iPlanner为7.20 ms，PS-DWA为45.99 ms，SAPF为3.56 ms。SAPF计算开销最低，但总体成功率仅11.1%。在本文方法与iPlanner共同成功的42组种子上，路径平滑度分别为0.0363 rad和0.1517 rad，本文方法降低约76.0%，Wilcoxon检验p=1.65×10^-8；能耗代理指标分别为740.04和774.49，p=7.58×10^-5。iPlanner的共同成功样本平均到达时间更短，为32.58 s，本文方法为36.57 s，p=0.000453。由此可见，本文方法取得了与iPlanner统计相当的成功率，并在轨迹平滑性和规划开销方面表现出优势，但不能表述为所有指标均优于对比方法。"
)
add_image("19_external_success_rate.png", "图12 本文方法与外部方法的三档导航成功率")

add_paragraph("VIO噪声鲁棒性", "heading")
add_paragraph(
    "VIO实验固定使用中等难度场景、足迹MPPI和同一组30个种子，只改变规划器接收的位姿输入。ideal、mild、medium和severe四档共完成120轮，全部结果文件与日志完整，同种子四档障碍布局一致。结果如表3所示。"
)
add_table(
    ["VIO档位", "严格成功率", "误差≤0.20m", "误到达", "位置RMS/m", "航向RMS"],
    [
        ["ideal", "17/30", "17/30", "0", "0.000", "0.00°"],
        ["mild", "7/30", "15/30", "8", "0.033", "0.77°"],
        ["medium", "7/30", "15/30", "11", "0.095", "1.70°"],
        ["severe", "5/30", "8/30", "12", "0.191", "3.53°"],
    ],
    8.8,
)
add_paragraph("表3 不同VIO噪声档位下的导航结果", "caption")
add_paragraph(
    "以真值终点误差不超过0.15 m作为严格成功判据，mild、medium和severe相对ideal的成功率均显著下降，对应p值分别为0.00635、0.02127和0.00183。mild档的8次误到达均位于0.151至0.189 m，说明厘米级定位偏差已经会影响严格终点判定；随着噪声增强，误到达次数和最大真实误差继续增加。实物系统不能仅根据单帧VIO估计距离立即宣布到达，应加入连续帧确认、减速停止后的二次确认，并在必要时结合目标视觉重定位或GPS。当前噪声参数属于仿真压力测试，后续需依据D435i实测轨迹重新标定。"
)
add_image("16_vio_navigation_success.png", "图13 VIO噪声对导航成功率的影响")

add_paragraph("实验结果总结", "heading")
add_paragraph(
    "目前第三章已经形成由底层运动控制、深度视觉建图、双层代价FMM、足迹MPPI、三档随机障碍消融、近年方法对比和VIO鲁棒性分析组成的完整实验链路。内部消融说明足迹MPPI在中等和困难障碍密度下显著提高成功率，并持续降低角速度指令变化；外部比较说明本文方法与iPlanner的成功率统计相当，同时输出轨迹更平滑、规划开销更低，并相对PS-DWA和SAPF取得显著成功率优势。实验还表明，时序障碍记忆在当前静态随机场景中没有稳定收益，因此不作为论文主要贡献；VIO误差则是实物部署中必须重点处理的因素。"
)

add_paragraph("成果及下一步计划", "title")
add_paragraph("成果", "heading")
add_paragraph(
    "目前已完成HIMLoco底层强化学习运动控制与上层局部规划的统一接口；实现D435i深度图到局部BEV的在线建图、双层代价FMM和矩形足迹MPPI；完成三档内部消融270轮、VIO鲁棒性120轮以及外部方法对比360轮，共计750轮正式试验；建立固定种子、障碍布局哈希、断点续跑、日志保存、统计检验和论文图表自动生成流程。第三章所需的统计图、Isaac Gym场景图、RGB-D图、BEV图、软代价图、FMM势场图和规划轨迹图均已整理完成。"
)
add_paragraph(
    "学术论文及其他个人成果：【请根据实际情况补充已发表、在投论文、专利、软件著作权和竞赛获奖，不填写无法提供证明的内容。】"
)
add_paragraph("下一步计划", "heading")
add_paragraph(
    "完成论文第三章：整理HIMLoco与RAM速度跟踪结果，统一第三章符号、公式、参数和方法命名，将三档规划消融、外部方法对比和VIO鲁棒性结果写入论文，并核对正文、表格和原始数据的一致性。",
    "list",
)
add_paragraph(
    "开展视觉可通行性研究：建立RGB-D数据生成与标注流程，融合语义、地表纹理、高度差、坡度和粗糙度等特征，输出0至1连续可通行代价。通过IoU、AUC、危险区域漏检率和闭环导航成功率评价模型，并与纯深度FMM、纯几何足迹MPPI及ViPlanner或WVN类方法进行比较。",
    "list",
)
add_paragraph(
    "完成Go2与D435i实物链路：完成相机安装、外参标定、深度尺度和时间同步检查，部署VIO并采集重复轨迹统计位置与航向误差，打通ROS2图像、位姿、规划速度和Go2控制接口。测试按照静止感知、低速直行、单障碍、稀疏随机障碍和混合地形逐步开展，并保留遥控急停。",
    "list",
)
add_paragraph(
    "完成系统集成与论文撰写：将第三章几何规划器、第四章连续可通行性模型和HIMLoco整合。任务目标可由操作者给定相对方向和距离，条件允许时加入GPS航点或前沿探索子目标，并将人员检测作为任务触发。最终完成第五章实物实验、第六章总结以及全文图表和参考文献整理。",
    "list",
)
add_paragraph(
    "当前已完成的工作能够支撑论文第三章，后续研究重点为视觉可通行性学习和Go2实物部署。对于尚未完成的第四章和第五章内容，将在取得真实实验数据后再确定最终贡献和结论。"
)

doc.storeAsURL(
    uno.systemPathToFileUrl(str(OUTPUT)),
    (prop("FilterName", "Office Open XML Text"), prop("Overwrite", True)),
)
doc.close(True)
print(OUTPUT)
