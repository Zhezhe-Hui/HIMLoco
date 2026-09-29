import time
from pathlib import Path

import uno
from com.sun.star.awt import Point, Size
from com.sun.star.beans import PropertyValue


OUT = Path("/home/hzz/project/HIMLoco/tmp/midterm_flow_ppt/output/实验方案与技术路线图.pptx")

SLIDE_W = 33867
SLIDE_H = 19050

COLORS = {
    "navy": 0x24445F,
    "blue": 0xBDD7EE,
    "blue_line": 0x7F9DB9,
    "green": 0xC6E0B4,
    "green_dark": 0x70AD47,
    "ink": 0x31404A,
    "muted": 0x657786,
    "dash": 0x8DA3B7,
    "white": 0xFFFFFF,
    "arrow": 0x9DC3E6,
}


def pv(name, value):
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


def set_text(shape, text, size=16, bold=False, color=COLORS["ink"]):
    shape.String = text
    shape.TextHorizontalAdjust = 1
    shape.TextVerticalAdjust = 2
    shape.TextLeftDistance = 170
    shape.TextRightDistance = 170
    shape.TextUpperDistance = 80
    shape.TextLowerDistance = 80
    cursor = shape.createTextCursor()
    cursor.CharFontName = "Microsoft YaHei"
    cursor.CharFontNameAsian = "Microsoft YaHei"
    cursor.CharHeight = size
    cursor.CharWeight = 150.0 if bold else 100.0
    cursor.CharColor = color
    cursor.ParaAdjust = 3


def rect(slide, x, y, w, h, text="", fill=COLORS["white"], line=COLORS["dash"],
         line_width=35, radius=220, font_size=16, bold=False, text_color=COLORS["ink"]):
    shape = doc.createInstance("com.sun.star.drawing.RectangleShape")
    shape.Position = Point(x, y)
    shape.Size = Size(w, h)
    shape.FillStyle = 1
    shape.FillColor = fill
    shape.LineStyle = 1
    shape.LineColor = line
    shape.LineWidth = line_width
    try:
        shape.CornerRadius = radius
    except Exception:
        pass
    slide.add(shape)
    if text:
        set_text(shape, text, font_size, bold, text_color)
    return shape


def text_box(slide, x, y, w, h, text, size=24, bold=False, color=COLORS["navy"], align=3):
    shape = doc.createInstance("com.sun.star.drawing.TextShape")
    shape.Position = Point(x, y)
    shape.Size = Size(w, h)
    slide.add(shape)
    shape.String = text
    shape.TextHorizontalAdjust = 1
    shape.TextVerticalAdjust = 2
    cursor = shape.createTextCursor()
    cursor.CharFontName = "Microsoft YaHei"
    cursor.CharFontNameAsian = "Microsoft YaHei"
    cursor.CharHeight = size
    cursor.CharWeight = 150.0 if bold else 100.0
    cursor.CharColor = color
    cursor.ParaAdjust = align
    return shape


def line(slide, x1, y1, x2, y2, color=COLORS["muted"], width=35, dash=False,
         arrow_end=False):
    shape = doc.createInstance("com.sun.star.drawing.LineShape")
    shape.Position = Point(x1, y1)
    shape.Size = Size(x2 - x1, y2 - y1)
    shape.LineColor = color
    shape.LineWidth = width
    shape.LineStyle = 2 if dash else 1
    if dash:
        d = uno.createUnoStruct("com.sun.star.drawing.LineDash")
        d.Style = 0
        d.Dots = 0
        d.DotLen = 0
        d.Dashes = 1
        d.DashLen = 180
        d.Distance = 110
        shape.LineDash = d
    if arrow_end:
        shape.LineEndName = "Arrow"
        shape.LineEndWidth = 260
        shape.LineEndCenter = True
    slide.add(shape)
    return shape


def arrow(slide, x1, y1, x2, y2, width=80):
    return line(slide, x1, y1, x2, y2, COLORS["blue_line"], width, False, True)


def polygon(slide, points, fill=COLORS["arrow"], line_color=COLORS["blue_line"]):
    shape = doc.createInstance("com.sun.star.drawing.PolyPolygonShape")
    min_x = min(x for x, _ in points)
    min_y = min(y for _, y in points)
    max_x = max(x for x, _ in points)
    max_y = max(y for _, y in points)
    shape.Position = Point(min_x, min_y)
    shape.Size = Size(max_x - min_x, max_y - min_y)
    shape.PolyPolygon = (tuple(Point(x - min_x, y - min_y) for x, y in points),)
    shape.FillStyle = 1
    shape.FillColor = fill
    shape.LineStyle = 1
    shape.LineColor = line_color
    shape.LineWidth = 24
    slide.add(shape)
    return shape


def block_arrow(slide, x1, y1, x2, y2, body=180, head=520, head_width=620):
    if y1 == y2 and x2 > x1:
        cy = y1
        return polygon(slide, [
            (x1, cy - body // 2), (x2 - head, cy - body // 2),
            (x2 - head, cy - head_width // 2), (x2, cy),
            (x2 - head, cy + head_width // 2), (x2 - head, cy + body // 2),
            (x1, cy + body // 2),
        ])
    if y1 == y2 and x2 < x1:
        cy = y1
        return polygon(slide, [
            (x1, cy - body // 2), (x2 + head, cy - body // 2),
            (x2 + head, cy - head_width // 2), (x2, cy),
            (x2 + head, cy + head_width // 2), (x2 + head, cy + body // 2),
            (x1, cy + body // 2),
        ])
    if x1 == x2 and y2 > y1:
        cx = x1
        return polygon(slide, [
            (cx - body // 2, y1), (cx + body // 2, y1),
            (cx + body // 2, y2 - head), (cx + head_width // 2, y2 - head),
            (cx, y2), (cx - head_width // 2, y2 - head),
            (cx - body // 2, y2 - head),
        ])
    raise ValueError("block_arrow supports horizontal or downward arrows")


def dashed_panel(slide, x, y, w, h):
    p = rect(slide, x, y, w, h, fill=COLORS["white"], line=COLORS["dash"], line_width=28, radius=0)
    p.LineStyle = 2
    d = uno.createUnoStruct("com.sun.star.drawing.LineDash")
    d.Style = 0
    d.Dots = 0
    d.DotLen = 0
    d.Dashes = 1
    d.DashLen = 210
    d.Distance = 120
    p.LineDash = d
    return p


def bracket_group(slide, label, steps, x, y, w, h, stage_num):
    # Geometry mirrors the user's reference: stage label on the left, grouped steps on the right.
    label_w = 4700
    stage = rect(
        slide, x + 420, y + h // 2 - 820, label_w, 1640,
        f"{stage_num}. {label}", COLORS["blue"], COLORS["blue"],
        0, 280, 17, True, COLORS["navy"],
    )
    step_x = x + 6900
    step_w = w - 7600
    step_h = 1350
    if len(steps) == 3:
        gaps = 520
        total = 3 * step_h + 2 * gaps
        top = y + (h - total) // 2
        ys = [top, top + step_h + gaps, top + 2 * (step_h + gaps)]
    else:
        gaps = 360
        total = 4 * 1180 + 3 * gaps
        top = y + (h - total) // 2
        step_h = 1180
        ys = [top + i * (step_h + gaps) for i in range(4)]

    boxes = []
    for sy, item in zip(ys, steps):
        boxes.append(rect(
            slide, step_x, sy, step_w, step_h, item,
            COLORS["green"], COLORS["green"], 0, 240, 16, False,
        ))

    # Bracket and branches.
    trunk_x = x + 6100
    label_right = x + 420 + label_w
    line(slide, label_right, y + h // 2, trunk_x, y + h // 2, width=28)
    line(slide, trunk_x, ys[0] + step_h // 2, trunk_x, ys[-1] + step_h // 2, width=28)
    for sy in ys:
        line(slide, trunk_x, sy + step_h // 2, step_x, sy + step_h // 2, width=28)

    # Within-stage progression arrows between step boxes.
    for i in range(len(ys) - 1):
        block_arrow(
            slide, step_x + step_w // 2, ys[i] + step_h + 35,
            step_x + step_w // 2, ys[i + 1] - 35,
            body=80, head=170, head_width=260,
        )
    return stage, boxes


def add_title(slide, title, subtitle):
    text_box(slide, 900, 360, 32000, 950, title, 28, True, COLORS["navy"])
    text_box(slide, 900, 1200, 32000, 540, subtitle, 13, False, COLORS["muted"])
    line(slide, 1500, 1800, 32367, 1800, COLORS["blue_line"], 24)


def add_footer(slide, text):
    text_box(slide, 1200, 18400, 31400, 330, text, 9.5, False, COLORS["muted"])


def make_slide(slide, title, subtitle, groups, footer):
    slide.Width = SLIDE_W
    slide.Height = SLIDE_H
    add_title(slide, title, subtitle)

    left = 900
    top = 2200
    panel_w = 15350
    panel_h = 6900
    gap_x = 1367
    gap_y = 930
    positions = [
        (left, top),
        (left + panel_w + gap_x, top),
        (left + panel_w + gap_x, top + panel_h + gap_y),
        (left, top + panel_h + gap_y),
    ]

    for idx, ((label, steps), (x, y)) in enumerate(zip(groups, positions), start=1):
        dashed_panel(slide, x, y, panel_w, panel_h)
        bracket_group(slide, label, steps, x, y, panel_w, panel_h, idx)

    # Stage sequence: top-left to top-right, then down, then bottom-right to bottom-left.
    block_arrow(slide, left + panel_w + 120, top + panel_h // 2,
                left + panel_w + gap_x - 120, top + panel_h // 2)
    block_arrow(slide, left + panel_w + gap_x + panel_w // 2, top + panel_h + 90,
                left + panel_w + gap_x + panel_w // 2, top + panel_h + gap_y - 90)
    block_arrow(slide, left + panel_w + gap_x - 120, top + panel_h + gap_y + panel_h // 2,
                left + panel_w + 120, top + panel_h + gap_y + panel_h // 2)

    add_footer(slide, footer)


ctx, desktop = connect()
doc = desktop.loadComponentFromURL("private:factory/simpress", "_blank", 0, (pv("Hidden", True),))
pages = doc.getDrawPages()
slide1 = pages.getByIndex(0)
slide2 = pages.insertNewByIndex(1)

experiment_groups = [
    ("底层运动控制", ["HIMLoco策略训练", "HIMLoco与RAM跟踪对比", "统一导航速度接口"]),
    ("局部感知与规划", ["D435i深度恢复点云", "局部BEV与双层代价", "FMM引导足迹MPPI"]),
    ("仿真实验验证", ["三档离散高度场景", "30个固定种子消融", "VIO噪声与外部对比"]),
    ("实物系统部署", ["Go2与D435i标定", "VIO与ROS2数据链路", "野外混合地形验证"]),
]

route_groups = [
    ("运动控制准备", ["训练并部署HIMLoco", "验证速度跟踪精度", "固定底层控制策略"]),
    ("视觉几何建图", ["深度图恢复局部点云", "地面过滤与BEV栅格化", "构建硬碰撞层和软代价层"]),
    ("局部运动规划", ["FMM生成参考走廊", "足迹MPPI采样与评价", "输出线速度和角速度"]),
    ("验证与能力扩展", ["三档场景和VIO噪声实验", "iPlanner等方法对比", "视觉可通行性与实物部署"]),
]

make_slide(
    slide1,
    "实验方案",
    "围绕底层控制、局部视觉规划、仿真验证和实物部署开展分层实验",
    experiment_groups,
    "平台：Unitree Go2、Intel RealSense D435i、Isaac Gym    正式实验：750轮",
)
make_slide(
    slide2,
    "技术路线",
    "深度视觉建立局部几何代价，FMM提供走廊引导，足迹MPPI生成平滑速度指令",
    route_groups,
    "第三章：几何规划    第四章：视觉可通行性    第五章：Go2实物系统",
)

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.storeAsURL(
    uno.systemPathToFileUrl(str(OUT)),
    (pv("FilterName", "Impress MS PowerPoint 2007 XML"), pv("Overwrite", True)),
)
doc.close(True)
print(OUT)
