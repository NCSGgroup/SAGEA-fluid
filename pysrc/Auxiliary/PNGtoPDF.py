from PIL import Image
import os

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


# demo
png_to_pdf("I:\CRALICOM/result\Figure\Version2/Figure2_Appendix.png", "I:\CRALICOM/result\Figure\Version2/Figure2_Appendix.pdf", dpi=100)
pdf_to_png_pillow("single_page.pdf", "output_png_pillow", dpi=300)