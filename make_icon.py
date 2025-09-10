from PIL import Image, ImageOps
src = Image.open("icon.png").convert("RGBA")
if src.width != src.height:
    side = max(src.width, src.height)
    canvas = Image.new("RGBA", (side, side), (0,0,0,0))
    canvas.paste(src, ((side - src.width)//2, (side - src.height)//2))
    src = canvas
src = ImageOps.contain(src, (256,256))
src.save("icon.ico", sizes=[(256,256),(128,128),(64,64),(48,48),(32,32),(16,16)])
print("icon.ico erzeugt.")
