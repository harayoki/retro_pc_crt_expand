from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import numpy as np
import colorsys
import argparse


def preprocess(image, mode, background_color=(0, 0, 0)):
    """
    スケーリングの前処理
    画像サイズを指定されたゲーム機・PCの仕様にあわせて調整する。
    パラメータ:
    - image: 入力画像 (PIL Image)
    - mode: "MSX1", "MSX2", "FAMICOM" のいずれか 大概のゲーム機はFAMICOM相当でまかなえる。
    - background_color: 追加する背景色 (デフォルトは黒 (0, 0, 0))

    処理内容:
    1. 画像を一度横256ピクセルにリサイズ（縦幅はアスペクト比維持で適切にスケールされる）
        320ドットや512ドットの解像度を持つドット絵は今回考慮していないが
        PCENGONE_320 や MSX2_512 などモード引数を増やすして対応したい。
    2. 縦のドット数で多い部分はカット、少ない部分は背景色で補填する。
       """

    # 1. 横幅を256ピクセルに変更（アスペクト比を維持）
    width, height = image.size
    new_height = int(height * (256 / width))
    image = image.resize((256, new_height), Image.Resampling.NEAREST)

    # 2. カット＆トリム
    if mode == "MSX1":
        target_height = 192
    elif mode == "MSX2":
        target_height = 212
    else:
        target_height = 224
    current_height = image.height
    if current_height > target_height:
        # 上下カット
        crop_start = (current_height - target_height) // 2
        image = image.crop((0, crop_start, 256, crop_start + target_height))
    else:
        # 背景色追加
        pad_top = (target_height - current_height) // 2
        new_image = Image.new("RGB", (256, target_height), background_color)
        new_image.paste(image, (0, pad_top))
        image = new_image

    return image


def create_scan_line_bg(scanline_scale, width, height, color, scan_line_ratio):
    """
    スキャンラインエフェクト入りの背景を作成する
    """
    # 一度2倍サイズで作ってから
    base_height = int(height / scanline_scale)
    img = Image.new("RGB", (1, base_height), (color[0], color[1], color[2]))
    ratio = 1.0 - scan_line_ratio
    scan_line_color = \
        (int(color[0] * ratio), int(color[1] * ratio), int(color[2] * ratio))
    if scan_line_ratio > 0.0:
        for y in range(1, base_height, 2):
            img.putpixel((0, y), scan_line_color)
    # 最終的な大きさにリサイズ
    img = img.resize((width, height), Image.Resampling.NEAREST)
    return img


def horizontal_blur(rgb_image, blur_ratio_rgb=(1.0, 1.0, 0.0)):
    """
    画像の横方向にブラーをかける
    """
    rgb_image = rgb_image.copy()
    for y in range(rgb_image.shape[0]):
        # 一行ごとにブラーをかける
        row = rgb_image[y]
        for x in range(1, rgb_image.shape[1] - 1):
            row[x, 0] = (row[x - 1, 0] * blur_ratio_rgb[0] + row[x, 0] * (1.0 - blur_ratio_rgb[0])).astype(np.uint8)
            row[x, 1] = (row[x - 1, 1] * blur_ratio_rgb[1] + row[x, 1] * (1.0 - blur_ratio_rgb[1])).astype(np.uint8)
            row[x, 2] = (row[x - 1, 2] * blur_ratio_rgb[2] + row[x, 2] * (1.0 - blur_ratio_rgb[2])).astype(np.uint8)
    return rgb_image


def upscale_with_crt_effect(image_obj, width, crt_width_ratio, xblur_ratio, scan_line_ratio):
    """
    CRTモニタの効果をシミュレートして画像を拡大する
    """

    # 対応解像度の確認 320以上は通しているものの動作確認はしていない
    assert image_obj.width == 256 or image_obj.width == 320 or image_obj.width == 512, \
        f"Width must be 256, 320, 512, but got {image_obj.width}"
    assert image_obj.height == 192 or image_obj.height == 212 or image_obj.height == 224, \
        "Height must be 192, 212 or 224"

    # 内部計算上の横幅を指定 小さい値になると左右方向の絵が劣化するので古いCRTのシミュレートになる。
    inner_crt_width = int(width * crt_width_ratio)
    print(f"inner_crt_width: {inner_crt_width}")

    final_size = (width, int( width * 2 * image_obj.height / 586))
    print (f"input {image_obj.width} x {image_obj.height} -> output {final_size[0]} x {final_size[1]}")
    # CRTの処理場の横サイズは性能によって可変 にじみを 330～440 オーバーにするなら 200ぐらいでいい
    image_obj = image_obj.resize((inner_crt_width, image_obj.height), Image.Resampling.BICUBIC)

    # スキャンラインエフェクトは最終引き伸ばしより先に行う
    scanline_scale = final_size[1] / (image_obj.height * 2)
    if scan_line_ratio > 0.0:
        # 偶数行と奇数行で輝度差を出す
        image_obj = image_obj.resize((image_obj.width, image_obj.height * 2), Image.Resampling.NEAREST)
        rgb_image = np.array(image_obj, dtype=np.uint8)
        for y in range(image_obj.height):
            if y % 2 == 0:
                for x in range(1, image_obj.width - 1):
                    rgb_image[y, x] = (
                        np.clip(rgb_image[y, x] * (1.0 - scan_line_ratio), 0, 255))
        image_obj = Image.fromarray(rgb_image)

    # 横幅をCRTサイズに拡大 縦幅は224ドットを480にする比率で拡大
    image_obj = image_obj.resize(final_size, Image.Resampling.BICUBIC)
    rgb_image = np.array(image_obj, dtype=np.uint8)
    height, width, channels = rgb_image.shape

    # # Rチャンネルだけ左に1ドットずらす
    # rgb_image[:, 1:, 0] = rgb_image[:, :-1, 0]
    # # # Gチャンネルだけ右に1ドットずらす
    # rgb_image[:, :-1, 1] = rgb_image[:, 1:, 1]

    # X方向だけのブラー
    xblur_rgb = [0.8 * xblur_ratio, 0.6 * xblur_ratio, 0.0 * xblur_ratio]
    boke_rgb = horizontal_blur(rgb_image, xblur_rgb)
    # 画像の合成 明るいほうを採用
    rgb_image = np.maximum(rgb_image, boke_rgb)

    rgb_image_obj = Image.fromarray(rgb_image)
    return rgb_image_obj, scanline_scale


def generate_subpixel_decomposition(image, subpixel_effect_ratio):
    """
    画像をRGBのサブピクセル構造に分解し、
    横長の三角形パターンで並べるシミュレーションを行う。

    - 入力: 1280x960 の画像を想定
    - 出力: RGBサブピクセルを適用した画像
    """
    width, height = image.size
    subpixel_img = Image.new("RGB", (width, height), "black")
    pixels = subpixel_img.load()
    original_pixels = image.load()

    for y in range(height):
        for x in range(0, width, 3):  # 3サブピクセルごとにRGBを配置
            if x + 2 < width:
                ratio = 1.0 - subpixel_effect_ratio
                if y % 8 < 4:
                    r, g, b = original_pixels[x + 0, y]
                    pixels[x + 0, y] = (r, int(g * ratio) , int(b * ratio))
                    r, g, b = original_pixels[x + 1, y]
                    pixels[x + 1, y] = (int(r * ratio), g, int(b * ratio))
                    r, g, b = original_pixels[x + 2, y]
                    pixels[x + 2, y] = (int(r * ratio), int(g * ratio), b)
                else:
                    r, g, b = original_pixels[x + 0, y]
                    pixels[x + 0, y] = (int(r * ratio), g, int(b * ratio))
                    r, g, b = original_pixels[x + 1, y]
                    pixels[x + 1, y] = (int(r * ratio), int(g * ratio), b)
                    r, g, b = original_pixels[x + 2, y]
                    pixels[x + 2, y] = (r, int(g * ratio) , int(b * ratio))

    # # 少しGaussianBlur
    # subpixel_img2 = subpixel_img.filter(ImageFilter.GaussianBlur(1.0))
    # # 比較して明るい方で合成
    # subpixel_img = np.maximum(subpixel_img2, subpixel_img)

    # 明るく
    subpixel_img = ImageEnhance.Brightness(subpixel_img).enhance(1.0 + 0.4 * subpixel_effect_ratio)
    # # 彩度を上げる
    subpixel_img = ImageEnhance.Color(subpixel_img).enhance(1.0 + 0.4 * subpixel_effect_ratio)

    return subpixel_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input image file")
    parser.add_argument("-o", "--output", help="output image file", default="_.png")
    parser.add_argument(
        "-s", "--scale",
        help="output image scale / ex 2.8125 192 -> 1080 / 5.0943 212 -> 1080 / 4.82143 224 -> 1080",
        type=float, default=1.0)
    parser.add_argument("-m", "--mode", help="MSX1, MSX2, FAMICOM", default="MSX1")
    parser.add_argument("-cw", "--crt_width_ratio", help="CRT monitor width ratio",
                type=float, default=0.625)  # 0.625 = 400 / 640
    parser.add_argument("-bg", "--background_color", nargs=3, type=int,
                help="background color (R,G,B) 0 - 255", default=[0, 0, 0])
    parser.add_argument("-hm", "--hmargin", action="store_true",
                help="add horizontal margin width(597 base) to the image")
    parser.add_argument("-vm", "--vmargin", action="store_true",
                help="add margin height(448 base) to the image")
    parser.add_argument("-xb", "--xblur_rgb", type=float, default=0.0,
                        help="horizontal blur ratio for RGB")
    parser.add_argument("-sp", "--subpixel_effect_ratio", type=float, default=0.0,
                help="generate subpixel decomposition")
    parser.add_argument("-sl", "--scan_line_ratio", type = float, default=0.0,
                        help="scan line effect  0.0 - 1.0")
    args = parser.parse_args()
    args.xblur_rgb = max(0.0, min(1.0, args.xblur_rgb))
    args.subpixel_effect_ratio = np.clip(args.subpixel_effect_ratio, 0.0, 1.0)
    args.scan_line_ratio = np.clip(args.scan_line_ratio, 0.0, 1.0)

    image = Image.open(args.input)
    image.convert("RGB")
    image = preprocess(image, args.mode, tuple(args.background_color))
    image, scanline_scale = upscale_with_crt_effect(
        image, 586, args.crt_width_ratio, args.xblur_rgb, args.scan_line_ratio)
    if args.subpixel_effect_ratio > 0.0:
        # サブpixelエフェクト: 色分解すると暗くなるので適切に作るのが難しい
        image = generate_subpixel_decomposition(image, args.subpixel_effect_ratio)
    size_with_margin = [image.width, image.height]
    if args.hmargin:
        size_with_margin[0] = int(image.width / 586 * 597)
        if size_with_margin[0] % 2 == 1:
            size_with_margin[0] += 1
    if args.vmargin:
        size_with_margin[1] = int(image.width / 586 * 448)
        if size_with_margin[1] % 2 == 1:
            size_with_margin[1] += 1
    if image.width != size_with_margin[0] or image.height != size_with_margin[1]:
        print(f"size with  margin: {size_with_margin}")
        pad_top = (size_with_margin[1] - image.height) // 2
        pad_left = (size_with_margin[0] - image.width) // 2
        # 背景にスキャンライン伊賀のエフェクトはつけていない
        new_image = create_scan_line_bg(
            scanline_scale, size_with_margin[0], size_with_margin[1], args.background_color, args.scan_line_ratio)
        new_image.paste(image, (pad_left, pad_top))
        image = new_image
    if args.scale != 0:
        scaled_size = (int(image.width * args.scale), int(image.height * args.scale))
        print(f"scaling: {size_with_margin} -> {scaled_size}")
        image = image.resize(scaled_size, Image.Resampling.BICUBIC)
    image.save(args.output)


if __name__ == "__main__":
    main()
