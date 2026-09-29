import time
from pathlib import Path

import uno
from PIL import Image
from com.sun.star.awt.FontWeight import BOLD, NORMAL
from com.sun.star.beans import PropertyValue
from com.sun.star.style.ParagraphAdjust import BLOCK, CENTER, LEFT
from com.sun.star.text.ControlCharacter import PARAGRAPH_BREAK
from com.sun.star.text.TextContentAnchorType import AS_CHARACTER


SOURCE = Path("/home/hzz/project/HIMLoco/tmp/midterm_results/source/midterm_latest.docx")
OUTPUT = Path("/home/hzz/project/HIMLoco/tmp/midterm_results/output/中期报告_实验数据及分析补充版.docx")
FIG = Path("/home/hzz/project/HIMLoco/assets/runs/navigation_benchmark/ch3_paper_figures_20260925")
image_jobs = []


def prop(name, value):
    p = PropertyValue()
    p.Name = name
    p.Value = value
    return p


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
            return desktop
        except Exception:
            time.sleep(0.2)
    raise RuntimeError("Unable to connect to LibreOffice")


def set_style(cur, role="body"):
    cur.NumberingStyleName = ""
    cur.CharFontName = "宋体"
    cur.CharFontNameAsian = "宋体"
    cur.CharWeight = NORMAL
    cur.CharColor = 0x000000
    cur.ParaLeftMargin = 0
    cur.ParaRightMargin = 0
    cur.ParaTopMargin = 0
    cur.ParaBottomMargin = 100
    spacing = uno.createUnoStruct("com.sun.star.style.LineSpacing")
    spacing.Mode = 0
    spacing.Height = 150
    cur.ParaLineSpacing = spacing
    if role == "heading":
        cur.CharFontName = "黑体"
        cur.CharFontNameAsian = "黑体"
        cur.CharHeight = 14
        cur.CharWeight = BOLD
        cur.ParaAdjust = LEFT
        cur.ParaFirstLineIndent = 0
        cur.ParaTopMargin = 220
        cur.ParaBottomMargin = 140
    elif role == "caption":
        cur.CharHeight = 10.5
        cur.ParaAdjust = CENTER
        cur.ParaFirstLineIndent = 0
        cur.ParaBottomMargin = 160
    else:
        cur.CharHeight = 12
        cur.ParaAdjust = BLOCK
        cur.ParaFirstLineIndent = 740


def reset_ins():
    global ins
    enum = doc.Text.createEnumeration()
    while enum.hasMoreElements():
        item = enum.nextElement()
        if item.supportsService("com.sun.star.text.Paragraph"):
            if item.String.strip() == "成果及下一步计划":
                ins = doc.Text.createTextCursorByRange(item.Start)
                return
    raise RuntimeError("Insertion anchor was lost")


def add_paragraph(text, role="body"):
    global ins
    set_style(ins, role)
    doc.Text.insertString(ins, text, False)
    doc.Text.insertControlCharacter(ins, PARAGRAPH_BREAK, False)
    reset_ins()


def add_image(filename, caption, max_width_cm=15.3, max_height_cm=8.4):
    path = FIG / filename
    if not path.exists():
        raise FileNotFoundError(path)
    with Image.open(path) as im:
        aspect = im.width / im.height
    width = max_width_cm
    height = width / aspect
    if height > max_height_cm:
        height = max_height_cm
        width = height * aspect
    token = f"__HIMLOCO_FIGURE_{len(image_jobs):02d}__"
    image_jobs.append((token, path, int(width * 1000), int(height * 1000)))
    add_paragraph(token, "caption")
    add_paragraph(caption, "caption")


def materialize_images():
    for token, path, width, height in image_jobs:
        search = doc.createSearchDescriptor()
        search.SearchString = token
        found = doc.findFirst(search)
        if found is None:
            raise RuntimeError(f"Image placeholder not found: {token}")
        found.String = ""
        graphic = doc.createInstance("com.sun.star.text.TextGraphicObject")
        graphic.GraphicURL = uno.systemPathToFileUrl(str(path))
        graphic.AnchorType = AS_CHARACTER
        graphic.Width = width
        graphic.Height = height
        doc.Text.insertTextContent(found, graphic, False)


def add_table(headers, rows, caption, font_size=8.5):
    global ins
    table = doc.createInstance("com.sun.star.text.TextTable")
    table.initialize(len(rows) + 1, len(headers))
    table.Split = True
    table.RepeatHeadline = True
    table.HeaderRowCount = 1
    doc.Text.insertTextContent(ins, table, False)
    for r, row in enumerate([headers] + rows):
        for c, value in enumerate(row):
            cell = table.getCellByName(chr(ord("A") + c) + str(r + 1))
            cell.String = str(value)
            tc = cell.createTextCursor()
            tc.CharFontName = "宋体"
            tc.CharFontNameAsian = "宋体"
            tc.CharHeight = font_size
            tc.CharWeight = BOLD if r == 0 else NORMAL
            tc.ParaAdjust = CENTER
            if r == 0:
                cell.BackColor = 0xD9EAF7
            elif r % 2 == 0:
                cell.BackColor = 0xF7F9FB
    reset_ins()
    doc.Text.insertControlCharacter(ins, PARAGRAPH_BREAK, False)
    reset_ins()
    add_paragraph(caption, "caption")


desktop = connect()
doc = desktop.loadComponentFromURL(
    uno.systemPathToFileUrl(str(SOURCE)), "_blank", 0,
    (prop("Hidden", True), prop("ReadOnly", False)),
)
if doc is None:
    raise RuntimeError("Failed to open source document")

start = None
end = None
enum = doc.Text.createEnumeration()
while enum.hasMoreElements():
    item = enum.nextElement()
    if not item.supportsService("com.sun.star.text.Paragraph"):
        continue
    t = item.String.strip().replace(" ", "")
    if start is None and t.startswith("2.局部视觉感知与BEV代价地图构建效果"):
        start = item
    if t == "成果及下一步计划":
        end = item
        break
if start is None or end is None:
    raise RuntimeError("Could not locate replacement range")

cut = doc.Text.createTextCursorByRange(start.Start)
cut.gotoRange(end.Start, True)
cut.String = ""
ins = doc.Text.createTextCursorByRange(cut.Start)

add_paragraph("2. 局部视觉感知与BEV代价地图构建效果", "heading")
add_paragraph(
    "局部视觉感知模块负责将D435i采集的深度图转换为规划器能够使用的局部环境表示。系统首先根据相机内参将有效深度像素恢复为三维点云，再经过相机坐标系、机身坐标系和局部世界坐标系变换，将点云投影到机器人周围的鸟瞰栅格中。栅格内的高度统计结果用于生成局部BEV高度图，并根据地面高度差建立障碍占据图。当前正式实验已经关闭仿真演员类别白名单，规划器只使用相机深度和机器人位姿完成局部建图，避免直接读取障碍物类别或位置真值。"
)
add_image(
    "10_rgbd_bev_cost_planning_pipeline.png",
    "图6 RGB-D局部感知、双层代价建模与轨迹规划过程",
)
add_paragraph(
    "图6展示了从RGB-D图像到局部规划结果的完整数据链路。彩色图用于观察场景外观，深度图提供障碍物与机器人之间的几何距离，局部高度图和障碍图用于表示机器人周围的可通行空间。在障碍占据图的基础上，系统通过距离变换计算每个栅格到最近障碍物的距离，并将安全约束划分为硬碰撞层和连续软安全代价层。当栅格与障碍物之间的距离小于硬安全半径时，该位置被判定为不可通行；在硬安全半径与软安全距离之间，代价随障碍距离增加而连续衰减。"
)
add_image(
    "14_soft_cost_and_fmm_potential.png",
    "图7 连续软安全代价与FMM势场构建结果",
)
add_paragraph(
    "由图7可以看出，二值障碍图只给出可通行与不可通行两种状态，而连续软代价能够进一步反映不同位置的碰撞风险。FMM在双层代价地图上从目标位置向外传播，得到局部到达时间场和参考走廊。该表示既保留了满足机器人机身尺寸约束的窄通道，又使参考路线在空间允许时主动远离障碍，为后续足迹MPPI轨迹优化提供了稳定的几何引导。"
)

add_paragraph("3. 不同局部规划方法的对比实验", "heading")
add_paragraph(
    "为了评价本文方法相对于现有局部规划方法的综合性能，选取iPlanner、PS-DWA和SAPF作为对比方法。四种方法使用相同的离散高度地形、障碍物布局、随机种子、理想VIO输入、HIMLoco底层控制器和成功判据，仅替换上层局部规划器。实验场景分为简单、中等和困难三档，分别包含10、15和20个大石头，2、3和4棵树，以及100、200和400个小碎石。每档采用30个固定随机种子，共完成360轮外部方法对比实验。"
)
add_image(
    "08_isaacgym_scene_comparison.png",
    "图8 简单、中等和困难三档Isaac Gym随机障碍场景",
)
add_table(
    ["难度", "本文足迹MPPI", "iPlanner", "PS-DWA", "SAPF"],
    [
        ["简单", "25/30", "22/30", "13/30", "5/30"],
        ["中等", "18/30", "19/30", "13/30", "4/30"],
        ["困难", "10/30", "14/30", "6/30", "1/30"],
        ["合计", "53/90", "55/90", "32/90", "10/90"],
    ],
    "表4 本文方法与其他局部规划方法的导航成功率",
    8.8,
)
add_image(
    "19_external_success_rate.png",
    "图9 不同局部规划方法在三档场景中的导航成功率",
)
add_paragraph(
    "由表4和图9可以看出，本文足迹MPPI在90组试验中成功53次，总成功率为58.9%；iPlanner成功55次，总成功率为61.1%。两种方法的配对McNemar检验结果为p=0.8388，说明二者成功率不存在显著差异。本文方法相对PS-DWA的配对检验结果为p=0.0005083，相对SAPF为p=1.60×10^-11，成功率差异均达到显著水平。随着障碍密度增加，各方法的成功率均有所下降，但本文方法和iPlanner在困难场景中仍保持了相对较高的通过能力。"
)
add_image(
    "20_external_overall_success_and_planning.png",
    "图10 不同局部规划方法的总体成功率与规划耗时",
)
add_paragraph(
    "在计算效率方面，本文方法的平均单次规划耗时为5.82 ms，iPlanner为7.20 ms，PS-DWA为45.99 ms，SAPF为3.56 ms。SAPF的计算开销最低，但总成功率仅为11.1%；PS-DWA需要对较多速度组合进行搜索，规划耗时明显高于其他方法。在本文方法与iPlanner共同成功的42组试验中，二者的平均轨迹平滑度分别为0.0363 rad和0.1517 rad，本文方法降低约76.0%，Wilcoxon检验结果为p=1.65×10^-8；能耗代理指标分别为740.04和774.49，p=7.58×10^-5。iPlanner平均到达时间为32.58 s，短于本文方法的36.57 s。因此，本文方法与iPlanner取得了统计上相当的成功率，并在轨迹平滑性和规划开销方面表现出优势，但在到达时间方面仍存在进一步优化空间。"
)

add_paragraph("4. 局部规划方法消融实验", "heading")
add_paragraph(
    "为了分析双层安全代价和机身足迹预测对规划性能的具体作用，在三档场景中比较固定膨胀FMM、双层代价FMM和双层代价FMM引导的足迹MPPI。三种方法使用相同的深度图、目标点、位姿输入、障碍布局和HIMLoco底层控制器，每种方法在每档场景中运行30个固定随机种子，共完成270轮内部消融实验。固定膨胀FMM只在二值障碍图上按照统一半径扩张障碍；双层代价FMM在硬碰撞约束之外增加连续软安全代价；足迹MPPI进一步使用0.72 m×0.42 m矩形机身包络预测短时运动轨迹。"
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
    "表5 三档随机障碍场景的局部规划方法消融结果",
    8.0,
)
add_image(
    "01_success_rate_with_ci.png",
    "图11 三种局部规划方法的导航成功率及95%置信区间",
)
add_paragraph(
    "简单场景中，足迹MPPI的成功率为83.3%，固定膨胀FMM和双层代价FMM分别为56.7%和80.0%。足迹MPPI相对固定膨胀FMM的配对检验结果为p=0.0574，尚未达到0.05显著性水平。中等场景中，足迹MPPI成功率为66.7%，相对两种FMM的33.3%均显著提高，p=0.00635。困难场景中，足迹MPPI成功率为46.7%，相对固定膨胀FMM的p值为0.00342，相对双层代价FMM的p值为0.0352。结果表明，随着障碍物密度增加，机身足迹预测和短时轨迹优化对避障成功率的改善更加明显。"
)
add_image(
    "03_smoothness_and_clearance.png",
    "图12 三种局部规划方法的轨迹平滑度与障碍净空",
)
add_paragraph(
    "在控制平滑性方面，足迹MPPI在简单、中等和困难场景中的角速度指令变化分别为0.0131、0.0127和0.0136，约为两种FMM方法的一半。这是因为FMM主要提供当前位置的梯度方向，机器人在障碍物边缘容易随局部梯度变化频繁调整转向；足迹MPPI在预测时域内同时评价路径偏差、障碍距离和控制变化，能够抑制连续规划过程中的左右振荡。MPPI每次规划比FMM增加约2至3 ms，但仍明显低于80 ms的规划周期，满足当前在线运行要求。"
)
add_image(
    "09_scene_route_comparison.png",
    "图13 三档随机障碍场景中的典型导航轨迹",
)
add_paragraph(
    "图13给出了三档场景中的典型运动轨迹。机器人在开阔区域主要沿目标方向前进，进入障碍密集区域后根据FMM参考走廊调整方向，并由足迹MPPI选择满足机身碰撞约束的短时速度序列。该结果说明双层代价地图保留了必要的窄通道通行空间，足迹预测则降低了只检查机器人中心点时可能出现的机身侧擦风险。"
)

add_paragraph("5. VIO位姿误差下的导航鲁棒性实验", "heading")
add_paragraph(
    "仿真导航能够直接获得机器人位姿真值，但实物系统需要依靠D435i图像与IMU估计相对位姿。为评价定位误差对导航结果的影响，将仿真真值定义为理想VIO，并在规划器输入端加入位置白噪声、航向白噪声、随机游走、固定延迟和随机丢帧，构造ideal、mild、medium和severe四档位姿输入。实验固定使用中等难度场景、足迹MPPI和同一组30个随机种子，只改变规划器接收的位姿，共完成120轮试验。"
)
add_table(
    ["VIO档位", "严格成功率", "误差≤0.20 m", "误到达", "位置RMS/m", "航向RMS"],
    [
        ["ideal", "17/30", "17/30", "0", "0.000", "0.00°"],
        ["mild", "7/30", "15/30", "8", "0.033", "0.77°"],
        ["medium", "7/30", "15/30", "11", "0.095", "1.70°"],
        ["severe", "5/30", "8/30", "12", "0.191", "3.53°"],
    ],
    "表6 不同VIO噪声等级下的导航结果",
    8.5,
)
add_image(
    "16_vio_navigation_success.png",
    "图14 不同VIO噪声等级下的导航成功率",
)
add_paragraph(
    "以真值终点误差不超过0.15 m作为严格成功判据，ideal档成功17次，mild和medium档均成功7次，severe档成功5次。三档噪声相对ideal的成功率均显著下降，对应p值分别为0.00635、0.02127和0.00183。当允许终点误差不超过0.20 m时，mild和medium档均有15次试验满足要求，说明其中部分失败来自严格终点判定，而不是机器人完全未到达目标附近。"
)
add_image(
    "17_vio_pose_error_and_false_arrival.png",
    "图15 VIO位姿误差与误到达次数",
)
add_paragraph(
    "随着噪声等级提高，位置和航向均方根误差持续增大，误到达次数也由ideal档的0次增加至severe档的12次。mild档的8次误到达均位于真值终点误差0.151~0.189 m范围内，说明厘米级定位偏差已经可能影响严格到达判定。实物部署时不能只根据单帧VIO估计距离立即结束导航，应加入连续多帧确认、减速停止后的二次判断，并根据D435i实测轨迹重新标定位姿噪声参数。"
)
add_paragraph(
    "综上，目前已经完成局部视觉建图、规划方法对比、内部消融和VIO鲁棒性分析。内部消融、外部方法对比和VIO噪声实验分别包含270轮、360轮和120轮，共形成750轮正式闭环导航试验。实验结果表明，本文方法能够在保持在线规划效率的同时改善中高难度场景下的成功率和控制平滑性，但定位误差和困难场景中的局部阻塞仍是后续实物部署需要重点解决的问题。"
)

materialize_images()

doc.storeAsURL(
    uno.systemPathToFileUrl(str(OUTPUT)),
    (prop("FilterName", "Office Open XML Text"), prop("Overwrite", True)),
)
doc.close(True)
print(OUTPUT)
