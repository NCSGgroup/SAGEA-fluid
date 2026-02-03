from PIL import Image
import os
from pdf2image import convert_from_path
from PyPDF2 import PdfMerger

class GeoFileKit:
    def __init__(self):
        pass

    @staticmethod
    def png_to_pdf(input_png, output_pdf, dpi=300):
        """
        PNG to PDF
        parameters:
            input_png: input path and file of PNG
            output_pdf: input path and file of PDF
            dpi: default is 300dpi
        """
        # open PNG
        image = Image.open(input_png)


        if image.mode == 'RGBA':
            image = image.convert('RGB')

        # save PDF
        image.save(output_pdf, "PDF", resolution=dpi)
        print(f"Successfully {input_png} transfer to {output_pdf}")

    @staticmethod
    def pdf_to_png_pillow(pdf_path, output_folder, dpi=300):
        """
        using Pillow PDF to PNG(only for one page of PDF)

        parameters:
            pdf_path: input path and file of PDF
            output_folder: the path of PNG
            dpi: default is 300dpi
        """
        # Create outputpath
        os.makedirs(output_folder, exist_ok=True)

        # Open PDF and transfer it
        image = Image.open(pdf_path)
        image.load()  # loading file

        # Set DPI
        image.info["dpi"] = (dpi, dpi)

        # Save PNG
        output_path = os.path.join(output_folder, "output.png")
        image.save(output_path, "PNG", dpi=(dpi, dpi))

        print(f"Successfully, PNG saved in: {output_path}")
    @staticmethod
    def pdf_to_png(pdf_path, output_folder, dpi=300):
        """
        将PDF文件转换为PNG图像

        参数:
            pdf_path: PDF文件路径
            output_folder: 输出文件夹路径
            dpi: 图像分辨率(默认为300)
        """
        # 确保输出文件夹存在
        os.makedirs(output_folder, exist_ok=True)

        # 转换PDF为图像列表
        images = convert_from_path(pdf_path, dpi=dpi)

        # 保存每页为PNG
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        for i, image in enumerate(images):
            output_path = os.path.join(output_folder, f"{base_name}_page_{i+1}.png")
            image.save(output_path, 'PNG')
            print(f"已保存: {output_path}")

    @staticmethod
    def simple_delete_files_by_extension(directory, extension):
        """Recursively delete files with the specified suffix"""
        if not extension.startswith('.'):
            extension = '.' + extension

        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(extension):
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
                    except Exception as e:
                        print(f"Deletion failed {file_path}: {e}")

    @staticmethod
    def merge_pdfs_advanced(path, output_filename="merged.pdf", sort_by_numbers=True):
        """
        高级版PDF合并函数
        Args:
            path: 文件路径
            output_filename: 输出文件名
            sort_by_numbers: 是否按数字排序（如果为False则按文件名排序）
        """
        pdf_files = [f for f in os.listdir(path) if f.lower().endswith('.pdf')]

        if sort_by_numbers:
            try:
                pdf_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
            except:
                pdf_files.sort()  # 如果数字提取失败，按默认排序
        else:
            pdf_files.sort()

        print("将按以下顺序合并PDF文件:")
        for i, pdf in enumerate(pdf_files, 1):
            print(f"{i}. {pdf}")

        merger = PdfMerger()

        for pdf_file in pdf_files:
            file_path = os.path.join(path, pdf_file)
            merger.append(file_path)

        output_path = os.path.join(path, output_filename)
        with open(output_path, 'wb') as f:
            merger.write(f)

        print(f"\n成功合并 {len(pdf_files)} 个PDF文件!")
        print(f"输出文件: {output_path}")



def demo_pdf_png():
    a = GeoFileKit()
    a.pdf_to_png(
        pdf_path="D:\Reference/1_Manuscripts\HUST-CRA, submit to Advances in Atmospheric Sciences\HUST-CRA\Figures/Figure3.pdf",
        output_folder="D:\Reference/1_Manuscripts\HUST-CRA, submit to Advances in Atmospheric Sciences\HUST-CRA\Figures/Fig3.png")
def demo_png_pdf():
    a = GeoFileKit()
    a.png_to_pdf(
        input_png="I:\Eassy\SaGEA_fluid/figure/Figure1.png",
        output_pdf="I:/Eassy/SaGEA_fluid/Figure1_1.pdf",
        dpi=500,
    )
def demo_delete_file():
    a = GeoFileKit()
    a.simple_delete_files_by_extension('/path/to/directory', '.tmp')

def demo_merge_pdfs():
    a = GeoFileKit().merge_pdfs_advanced(path="D:\HUST\sets/",output_filename="merge.pdf",sort_by_numbers=True)

if __name__ == "__main__":
    # demo_pdf_png()
    # demo_png_pdf()
    demo_merge_pdfs()