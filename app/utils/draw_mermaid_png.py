import uuid
from app.core.logger import logger
from app.utils.path_util import PROJECT_ROOT

def draw_mermaid_png(app, save_file_name="main_graph"):
    print("kb_import_app agent graph: ")
    print(app.get_graph().print_ascii())
    print("=" * 50)

    # 打印图的可视化结构，生成更加美观的Mermaid 代码，通过processon 编辑器查看
    print(app.get_graph().draw_mermaid())



    logger.info(f"Resolved project root: {PROJECT_ROOT}")

    # 3. 生成 PNG并写入文件
    png_bytes = app.get_graph().draw_mermaid_png()
    output_path = f"{PROJECT_ROOT}/output/{save_file_name}_" + str(uuid.uuid4())[:8] + ".png"
    with open(output_path, "wb") as f:
        f.write(png_bytes)
    print(f"\nGraph is generated：{output_path}")
