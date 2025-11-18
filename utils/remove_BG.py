from rembg import remove, new_session
from PIL import Image
import io, sys, pathlib
import os

inp = r"D:\BASIC_FILE\PICTURE\GeoMVD_924\GSO-compare\Black_Decker_Stainless_Steel_Toaster_4_Slice\selected\MV-Adapter_res512_stp50.png"

name = os.path.basename(inp).split(".")[0]
out = os.path.join(os.path.dirname(inp), name + "_rmbg.png")
model = "isnet-anime"  # 或 "u2net_human_seg", "isnet-anime"
session = new_session(model_name=model)

with open(inp, "rb") as f:
    result = remove(f.read(), session=session)

# result 已是带透明通道的 PNG 字节
with open(out, "wb") as g:
    g.write(result)

print(f"Saved: {out}")