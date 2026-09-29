import time
from pathlib import Path

import uno
from com.sun.star.beans import PropertyValue


SOURCE = Path(
    "/home/hzz/project/HIMLoco/tmp/midterm_opening_report/source/惠哲哲开题报告修改_工作副本.doc"
)
OUTPUT = Path("/home/hzz/桌面/惠哲哲开题报告修改_野外救援方向修订版_v2.docx")


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
    for _ in range(20):
        try:
            ctx = resolver.resolve(
                "uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext"
            )
            desktop = ctx.ServiceManager.createInstanceWithContext(
                "com.sun.star.frame.Desktop", ctx
            )
            return ctx, desktop
        except Exception:
            time.sleep(0.25)
    raise RuntimeError("Unable to connect to LibreOffice UNO service")


replacements = {
    3: "论文题目   面向野外复杂地形的四足机器人",
    4: "分层强化学习运动规划方法研究",
    16: "面向野外复杂地形的四足机器人分层强化学习运动规划方法研究",
    22: (
        "在全球科技智能化转型的浪潮下，机器人技术已逐渐成为衡量国家科技实力的重要指标之一，"
        "其应用场景正由传统工业制造向复杂非结构化环境快速延伸。在地震废墟搜救、野外资源勘探、"
        "森林与山地巡查、危险区域侦察等任务中，机器人能够代替人员进入坍塌、崎岖或存在次生灾害风险的区域，"
        "承担环境探测、目标搜索和信息回传等工作[1]~[2]。这类场景通常缺少可直接使用的先验地图，并伴随"
        "地形起伏、障碍物随机分布、道路损毁和视野遮挡等问题，对机器人的自主感知、导航决策和复杂地形运动"
        "能力提出了较高要求。传统轮式[3]、履带式移动机器人[4]虽在平坦连续路面上具有较高运动效率，但受到"
        "接触形式和底盘结构限制，在碎石、台阶、沟壑和不连续地面上容易出现打滑、托底或无法继续前进的情况，"
        "难以满足野外复杂环境中的连续作业需求。"
    ),
    23: "图1 履带式与轮式机器人在灾后救援和野外勘探场景中的作业",
    24: (
        "四足机器人凭借仿生腿部结构[5]和多自由度关节，能够主动选择落足位置，并根据地面高度和接触状态"
        "调整机身姿态。在遭遇离散凸起、碎石和台阶时，机器人可以通过腿部协调运动维持平衡，完成障碍跨越、"
        "楼梯攀爬[6]和不规则地面通行等动作。同时，腿式结构为任务载荷提供了较大的扩展空间，可在机身或腿部"
        "安装相机、力传感器[7]、气体检测器和生命探测设备，也可与机械臂协同完成取样、搬运等操作。凭借这些"
        "特点，四足机器人近年来成为机器人领域的研究热点，波士顿动力Spot[8]、宇树科技Go系列[9]等产品已"
        "逐步进入科研、巡检和应急作业场景。"
    ),
    25: "图2 Spot与Go2在碎石、台阶和野外场景下的运动",
    26: (
        "路径规划[10]~[12]是四足机器人自主完成任务的核心环节，是机器人依托环境感知信息，结合目标位置和"
        "自身运动特性，在起点与目标点之间持续生成可执行路线或速度指令的过程，直接影响其工作效率和运行安全[13]。"
        "可靠的路径规划是机器人顺利完成作业的基础。即使机器人具备较强的硬件性能和复杂地形适应能力，如果"
        "规划器没有正确处理障碍物、机身尺寸和运动约束，仍可能因路线选择不当发生碰撞、侧擦、局部停滞或反复"
        "转向，最终无法到达目标。因此，如何使规划结果既具有环境可达性，又符合四足机器人步态执行特点，已经"
        "成为四足机器人自主导航研究中的重要问题。"
    ),
    27: (
        "然而，面向灾后废墟和野外探索的四足机器人路径规划仍面临较多困难。这类环境通常无法预先获得完整、"
        "准确的地图，树木、岩石、草丛、低矮碎石和可跨越台阶在几何形态及通行风险上也存在明显差异，机器人"
        "需要依靠机载传感器在运动过程中持续感知局部环境并实时调整路线。与此同时，四足机器人具有不同于"
        "轮式平台的机身尺寸、转弯扫掠范围和动态步态，直接沿用传统地面机器人规划方法容易出现中心轨迹可行而"
        "机身侧面碰撞、固定安全距离封闭窄区域以及频繁重规划引起转向振荡等问题。针对上述需求，本文面向无先验"
        "地图的野外复杂地形，研究以强化学习运动控制为执行基础、融合视觉局部感知和在线轨迹优化的分层运动规划"
        "方法，使机器人能够在随机障碍和地形起伏条件下生成安全、平滑且可实时执行的运动指令，并为后续灾后搜救、"
        "野外勘察和危险区域自主探索系统提供技术基础。因此，该课题具有较明确的理论研究价值和工程应用意义。"
    ),
}


ctx, desktop = connect()
doc = desktop.loadComponentFromURL(
    uno.systemPathToFileUrl(str(SOURCE)),
    "_blank",
    0,
    (prop("Hidden", True), prop("ReadOnly", False)),
)

paragraphs = []
enum = doc.Text.createEnumeration()
while enum.hasMoreElements():
    item = enum.nextElement()
    if item.supportsService("com.sun.star.text.Paragraph") and item.String.strip():
        paragraphs.append(item)

for number, value in replacements.items():
    paragraphs[number - 1].String = value

# The legacy .doc contains tracked revisions that LibreOffice otherwise exports
# as malformed nested OOXML. Accept them in the disposable working copy first.
dispatch = ctx.ServiceManager.createInstanceWithContext(
    "com.sun.star.frame.DispatchHelper", ctx
)
dispatch.executeDispatch(
    doc.CurrentController.Frame, ".uno:AcceptAllTrackedChanges", "", 0, ()
)

doc.storeAsURL(
    uno.systemPathToFileUrl(str(OUTPUT)),
    (prop("FilterName", "Office Open XML Text"), prop("Overwrite", True)),
)
doc.close(True)
print(OUTPUT)
