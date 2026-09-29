import time
from pathlib import Path

import uno
from com.sun.star.awt import Point, Size
from com.sun.star.beans import PropertyValue
from com.sun.star.text.TextContentAnchorType import AS_CHARACTER


SOURCE = Path("/home/hzz/project/HIMLoco/tmp/paper_figure_ppt/source/experiment_route.pptx")
OUTPUT = Path("/home/hzz/project/HIMLoco/tmp/paper_figure_ppt/output/实验方案与技术路线图_增加总体框架.pptx")
ASSETS = Path("/home/hzz/project/HIMLoco/assets/runs/navigation_benchmark")

RGB = ASSETS / "ch3_visual_evidence_20260925/hard_seed20261001/camera/rgb_001000.png"
DEPTH = ASSETS / "ch3_visual_evidence_20260925/hard_seed20261001/camera/depth_001000.png"
SCENES = ASSETS / "ch3_paper_figures_20260925/08_isaacgym_scene_comparison.png"

NAVY = 0x17365D
INK = 0x1F2D3D
MUTED = 0x536878
PEACH = 0xFCE4D6
PEACH_2 = 0xF8CBAD
GREEN = 0xE2F0D9
GREEN_2 = 0xC6E0B4
BLUE = 0xD9E2F3
BLUE_2 = 0xB4C6E7
WHITE = 0xFFFFFF
ARROW = 0x8EAADB


def pv(name, value):
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


def apply_text(shape, text, size=15, bold=False, color=INK, align=3):
    shape.String = text
    shape.TextHorizontalAdjust = 1
    shape.TextVerticalAdjust = 1
    shape.TextLeftDistance = 130
    shape.TextRightDistance = 130
    shape.TextUpperDistance = 60
    shape.TextLowerDistance = 60
    cursor = shape.createTextCursor()
    cursor.CharFontName = "Times New Roman"
    cursor.CharFontNameAsian = "Microsoft YaHei"
    cursor.CharHeight = size
    cursor.CharWeight = 150.0 if bold else 100.0
    cursor.CharColor = color
    cursor.ParaAdjust = align


def text_box(slide, x, y, w, h, text, size=17, bold=False, color=INK, align=3):
    s = doc.createInstance("com.sun.star.drawing.TextShape")
    s.Position = Point(x, y)
    s.Size = Size(w, h)
    slide.add(s)
    apply_text(s, text, size, bold, color, align)
    return s


def rect(slide, x, y, w, h, text="", fill=WHITE, line=NAVY, width=32,
         radius=260, size=14, bold=False, color=INK):
    s = doc.createInstance("com.sun.star.drawing.RectangleShape")
    s.Position = Point(x, y)
    s.Size = Size(w, h)
    s.FillStyle = 1
    s.FillColor = fill
    s.LineStyle = 1
    s.LineColor = line
    s.LineWidth = width
    try:
        s.CornerRadius = radius
    except Exception:
        pass
    slide.add(s)
    if text:
        apply_text(s, text, size, bold, color)
    return s


def panel(slide, x, y, w, h, title, fill, dashed=False):
    p = rect(slide, x, y, w, h, fill=fill, line=NAVY, width=34, radius=650)
    if dashed:
        p.LineStyle = 2
        d = uno.createUnoStruct("com.sun.star.drawing.LineDash")
        d.Style = 0
        d.Dots = 0
        d.DotLen = 0
        d.Dashes = 1
        d.DashLen = 220
        d.Distance = 120
        p.LineDash = d
    text_box(slide, x + 240, y + 80, w - 480, 620, title, 15, True, INK)
    return p


def line(slide, x1, y1, x2, y2, width=34, color=INK):
    s = doc.createInstance("com.sun.star.drawing.LineShape")
    s.Position = Point(x1, y1)
    s.Size = Size(x2 - x1, y2 - y1)
    s.LineStyle = 1
    s.LineColor = color
    s.LineWidth = width
    slide.add(s)
    return s


def polygon(slide, points, fill=ARROW, line_color=NAVY):
    min_x = min(x for x, _ in points)
    min_y = min(y for _, y in points)
    max_x = max(x for x, _ in points)
    max_y = max(y for _, y in points)
    s = doc.createInstance("com.sun.star.drawing.PolyPolygonShape")
    s.Position = Point(min_x, min_y)
    s.Size = Size(max_x - min_x, max_y - min_y)
    s.PolyPolygon = (tuple(Point(x - min_x, y - min_y) for x, y in points),)
    s.FillStyle = 1
    s.FillColor = fill
    s.LineStyle = 1
    s.LineColor = line_color
    s.LineWidth = 22
    slide.add(s)
    return s


def arrow(slide, x1, y1, x2, y2, body=120, head=380, head_w=420):
    if y1 == y2 and x2 > x1:
        cy = y1
        return polygon(slide, [
            (x1, cy - body // 2), (x2 - head, cy - body // 2),
            (x2 - head, cy - head_w // 2), (x2, cy),
            (x2 - head, cy + head_w // 2), (x2 - head, cy + body // 2),
            (x1, cy + body // 2),
        ])
    if x1 == x2 and y2 > y1:
        cx = x1
        return polygon(slide, [
            (cx - body // 2, y1), (cx + body // 2, y1),
            (cx + body // 2, y2 - head), (cx + head_w // 2, y2 - head),
            (cx, y2), (cx - head_w // 2, y2 - head),
            (cx - body // 2, y2 - head),
        ])
    raise ValueError("Only rightward and downward arrows are supported")


def add_image(slide, path, x, y, w, h):
    if not path.exists():
        raise FileNotFoundError(path)
    g = doc.createInstance("com.sun.star.drawing.GraphicObjectShape")
    g.Position = Point(x, y)
    g.Size = Size(w, h)
    g.GraphicURL = uno.systemPathToFileUrl(str(path))
    g.LineStyle = 1
    g.LineColor = NAVY
    g.LineWidth = 20
    slide.add(g)
    return g


desktop = connect()
doc = desktop.loadComponentFromURL(
    uno.systemPathToFileUrl(str(SOURCE)), "_blank", 0,
    (pv("Hidden", True), pv("ReadOnly", False)),
)
pages = doc.getDrawPages()
slide = pages.insertNewByIndex(pages.getCount())

W = slide.Width
H = slide.Height

# Title
text_box(slide, 900, 250, W - 1800, 850,
         "分层视觉导航与运动控制总体框架",
         24, True, INK)
text_box(slide, 900, 1050, W - 1800, 480,
         "第三章已完成系统与实验评估流程",
         12.5, False, MUTED)

# Main regions
left_x, left_w = 300, 6900
mid_x, mid_w = 7600, 16600
right_x, right_w = 24700, 8850
top_y, top_h = 2200, 6650
bottom_y, bottom_h = 10100, 6900

panel(slide, left_x, top_y, left_w, top_h, "感知与任务输入", WHITE, True)
panel(slide, mid_x, top_y, mid_w, top_h, "局部感知与运动规划", PEACH)
panel(slide, left_x, bottom_y, left_w, bottom_h, "运动控制输入", WHITE, True)
panel(slide, mid_x, bottom_y, mid_w, bottom_h, "HIMLoco强化学习运动控制器", BLUE)
panel(slide, right_x, top_y, right_w, top_h + 8150, "已完成实验", GREEN)

# Input images and boxes
add_image(slide, RGB, left_x + 420, top_y + 900, 2860, 1610)
add_image(slide, DEPTH, left_x + 3600, top_y + 900, 2860, 1610)
text_box(slide, left_x + 420, top_y + 2510, 2860, 420, "D435i彩色图", 11.5, True)
text_box(slide, left_x + 3600, top_y + 2510, 2860, 420, "D435i深度图", 11.5, True)
rect(slide, left_x + 750, top_y + 3100, 5400, 1250,
     "VIO位姿\n（理想或含噪）", WHITE, NAVY, 28, 220, 12.5)
rect(slide, left_x + 750, top_y + 4800, 5400, 1050,
     "局部目标点", WHITE, NAVY, 28, 220, 14)

# Navigation pipeline: two rows keep labels readable.
nav_boxes = [
    (mid_x + 700, top_y + 950, 4000, 1450, "RGB-D点云投影\n生成局部BEV"),
    (mid_x + 6100, top_y + 950, 4000, 1450, "局部BEV\n高度图与障碍图"),
    (mid_x + 11500, top_y + 950, 4000, 1450, "双层代价地图\n硬碰撞层与软安全代价"),
    (mid_x + 700, top_y + 4200, 4000, 1450, "FMM参考走廊"),
    (mid_x + 6100, top_y + 4200, 4000, 1450, "足迹MPPI\n轨迹优化"),
    (mid_x + 11500, top_y + 4200, 4000, 1450, "速度指令\n(vx, wz)"),
]
for x, y, w, h, label in nav_boxes:
    rect(slide, x, y, w, h, label, PEACH_2, NAVY, 30, 260, 12.2, False)
arrow(slide, mid_x + 4700, top_y + 1675, mid_x + 6100, top_y + 1675)
arrow(slide, mid_x + 10100, top_y + 1675, mid_x + 11500, top_y + 1675)
line(slide, mid_x + 13500, top_y + 2400, mid_x + 13500, top_y + 3400, 28, NAVY)
line(slide, mid_x + 2700, top_y + 3400, mid_x + 13500, top_y + 3400, 28, NAVY)
arrow(slide, mid_x + 2700, top_y + 3400, mid_x + 2700, top_y + 4100, 90, 320, 350)
arrow(slide, mid_x + 4700, top_y + 4925, mid_x + 6100, top_y + 4925)
arrow(slide, mid_x + 10100, top_y + 4925, mid_x + 11500, top_y + 4925)

# Locomotion inputs
loc_inputs = [
    (bottom_y + 800, "期望机身速度"),
    (bottom_y + 2550, "本体感知状态"),
    (bottom_y + 4300, "历史观测"),
]
for y, label in loc_inputs:
    rect(slide, left_x + 750, y, 5400, 1150, label, WHITE, NAVY, 28, 230, 14)

# HIMLoco controller
rect(slide, mid_x + 800, bottom_y + 2200, 3600, 1700,
     "状态与历史观测\n编码器", BLUE_2, NAVY, 30, 280, 12.2)
rect(slide, mid_x + 5900, bottom_y + 1700, 4800, 2650,
     "HIMLoco\n策略网络", BLUE_2, NAVY, 34, 340, 18, True)
rect(slide, mid_x + 12200, bottom_y + 2200, 3600, 1700,
     "关节位置\n目标值", BLUE_2, NAVY, 30, 280, 12.2)
arrow(slide, mid_x + 4400, bottom_y + 3050, mid_x + 5900, bottom_y + 3050)
arrow(slide, mid_x + 10700, bottom_y + 3050, mid_x + 12200, bottom_y + 3050)
text_box(slide, mid_x + 820, bottom_y + 4600, 14900, 750,
         "所有导航规划方法共用同一底层运动控制策略",
         13, False, MUTED)

# Cross-module flow
arrow(slide, left_x + left_w + 100, top_y + 3350, mid_x - 150, top_y + 3350, 150, 460, 520)
arrow(slide, mid_x + 13500, top_y + top_h + 100,
      mid_x + 13500, bottom_y + 1600, 150, 460, 520)
arrow(slide, left_x + left_w + 100, bottom_y + 3150,
      mid_x - 150, bottom_y + 3150, 150, 460, 520)

# Completed evaluation column
add_image(slide, SCENES, right_x + 400, top_y + 900, right_w - 800, 2050)
evaluation = [
    (top_y + 3250, "简单、中等、困难三档场景\n每种方法30个固定随机种子"),
    (top_y + 5150, "内部消融\n固定膨胀FMM、双层代价FMM、\n足迹MPPI"),
    (top_y + 7050, "外部方法对比\niPlanner、PS-DWA、SAPF"),
    (top_y + 8950, "VIO鲁棒性\n理想、轻度、中度、重度噪声"),
    (top_y + 10850, "共完成750轮正式闭环实验"),
]
for idx, (y, label) in enumerate(evaluation):
    fill = GREEN_2 if idx < 4 else 0xA9D18E
    rect(slide, right_x + 700, y, right_w - 1400, 1550,
         label, fill, NAVY, 28, 250, 11.5, idx == 4)

# Output link to evaluation
arrow(slide, mid_x + mid_w + 120, bottom_y + 3050,
      right_x - 160, bottom_y + 3050, 150, 460, 520)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
doc.storeAsURL(
    uno.systemPathToFileUrl(str(OUTPUT)),
    (pv("FilterName", "Impress MS PowerPoint 2007 XML"), pv("Overwrite", True)),
)
doc.close(True)
print(OUTPUT)
