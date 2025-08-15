import requests
import base64
import fitz
import os
import cv2
import docx
from bs4 import BeautifulSoup
from layoutparser.models import Detectron2LayoutModel
from PIL import Image

def convert_pdf_to_images(pdf_path, output_folder):
    doc = fitz.open(pdf_path)    
    images = []

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img_filename = f"page_{page_num + 1}.png"
        img_filepath = os.path.join(output_folder, img_filename)
        pix.save(img_filepath)

        image = cv2.imread(img_filepath)
        image_rgb = image[..., ::-1]  # Convert BGR to RGB
        images.append(image_rgb)
        print(f"Page {page_num + 1} saved and added to images[]")
    doc.close()

    return images
def convert_word_to_images(docx_path, output_folder):
    doc = docx.Document(docx_path)
    os.makedirs(output_folder, exist_ok=True)
    images = []
    img_count = 0

    for rel in doc.part.rels.values():
        if "image" in rel.target_ref:
            img_count += 1
            img_data = rel.target_part.blob
            img_filename = f"word_image_{img_count}.png"
            img_filepath = os.path.join(output_folder, img_filename)

            with open(img_filepath, "wb") as img_file:
                img_file.write(img_data)

            image = cv2.imread(img_filepath)
            image_rgb = image[..., ::-1]  # Convert BGR to RGB
            images.append(image_rgb)
            print(f"Extracted image {img_count} from Word document")

    return images

def convert_html_to_images(html_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    images = []
    img_count = 0

    with open(html_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "lxml")

    for img_tag in soup.find_all("img"):
        img_url = img_tag.get("src")
        img_count += 1
        img_filename = f"html_image_{img_count}.png"
        img_filepath = os.path.join(output_folder, img_filename)

        # If image is a base64-encoded data URL
        if img_url.startswith("data:image/"):
            header, img_data = img_url.split(",", 1)
            img_data = base64.b64decode(img_data)
            with open(img_filepath, "wb") as img_file:
                img_file.write(img_data)
        elif img_url.startswith("http"):  # If it's a regular URL, download it
            img_data = requests.get(img_url).content
            with open(img_filepath, "wb") as img_file:
                img_file.write(img_data)
        else:  # If image is a local file, just copy it
            img_data = open(img_url, "rb").read()
            with open(img_filepath, "wb") as img_file:
                img_file.write(img_data)

        image = cv2.imread(img_filepath)
        image_rgb = image[..., ::-1]  # Convert BGR to RGB
        images.append(image_rgb)
        print(f"Extracted image {img_count} from HTML document")

    return images

def load_images(image_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    images = []
    image = cv2.imread(image_path)
    if image is not None:
        image_rgb = image[..., ::-1]  # Convert BGR to RGB
        images.append(image_rgb)
        print(f"Loaded image from {image_path} and added to images[]")
    else:
        print(f"Failed to load image from {image_path}")
    return images

def convert_file_to_images(file_path, output_folder):
    if file_path.endswith(".pdf"):
        return convert_pdf_to_images(file_path, output_folder)
    elif file_path.endswith(".docx"):
        return convert_word_to_images(file_path, output_folder)
    elif file_path.endswith(".html"):
        return convert_html_to_images(file_path, output_folder)
    elif file_path.endswith((".png", ".jpg", ".jpeg")):  # Accept images
        return load_images(file_path, output_folder)
    else:
        raise ValueError("Unsupported file format. Please provide a PDF, Word, HTML, or image file.")

def process_images_with_layoutparser(images_array, output_folder):
    model = Detectron2LayoutModel(
        'lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure", 5: "Image", 6: "Chart"}  # Added Chart
    )

    element_dirs = {
        "Text": os.path.join(output_folder, "extracted_text"),
        "Title": os.path.join(output_folder, "extracted_text"),
        "List": os.path.join(output_folder, "extracted_text"),
        "Table": os.path.join(output_folder, "extracted_table"),
        "Figure": os.path.join(output_folder, "extracted_image"),
        "Chart": os.path.join(output_folder, "extracted_chart")
    }

    imglist = []
    chartlist = []
    textlist = []
    tablelist = []

    imagepath = output_folder + 'extracted_image/'
    tablepath = output_folder + 'extracted_table/'
    textpath = output_folder + 'extracted_text/'
    chartpath = output_folder + 'extracted_chart/'

    result = {}   

    for dir_name in element_dirs.values():
        os.makedirs(dir_name, exist_ok=True)

    for img_index, image_rgb in enumerate(images_array):
        print(f"Processing image {img_index + 1}")
        layout = model.detect(image_rgb)

        for i, block in enumerate(layout):
            element_type = block.type            

            if element_type in element_dirs:
                x1, y1, x2, y2 = map(int, block.coordinates)                

                if element_type == 'Text':
                    cropped_image = image_rgb[y1:y2, x1:x2]
                    cropped_image_pil = Image.fromarray(cropped_image)
                    file_path = os.path.join(element_dirs[element_type], f"{element_type}_img{img_index + 1}_block{i + 1}.png")
                    cropped_image_pil.save(file_path)
                    textlist.append(f"{element_type}_img{img_index + 1}_block{i + 1}.png")
                elif element_type == 'Title':
                    cropped_image = image_rgb[y1:y2, x1:x2]
                    cropped_image_pil = Image.fromarray(cropped_image)
                    file_path = os.path.join(element_dirs[element_type], f"{element_type}_img{img_index + 1}_block{i + 1}.png")
                    cropped_image_pil.save(file_path)
                    textlist.append(f"{element_type}_img{img_index + 1}_block{i + 1}.png")
                elif element_type == 'List':
                    cropped_image = image_rgb[y1:y2, x1:x2]
                    cropped_image_pil = Image.fromarray(cropped_image)
                    file_path = os.path.join(element_dirs[element_type], f"{element_type}_img{img_index + 1}_block{i + 1}.png")
                    cropped_image_pil.save(file_path)
                    textlist.append(f"{element_type}_img{img_index + 1}_block{i + 1}.png")
                elif element_type == 'Table':
                    cropped_image = image_rgb[y1:y2+30, x1:x2]
                    cropped_image_pil = Image.fromarray(cropped_image)
                    file_path = os.path.join(element_dirs[element_type], f"{element_type}_img{img_index + 1}_block{i + 1}.png")
                    cropped_image_pil.save(file_path)
                    tablelist.append(f"{element_type}_img{img_index + 1}_block{i + 1}.png")
                elif element_type == 'Figure':
                    cropped_image = image_rgb[y1:y2+30, x1:x2]
                    cropped_image_pil = Image.fromarray(cropped_image)
                    file_path = os.path.join(element_dirs[element_type], f"{element_type}_img{img_index + 1}_block{i + 1}.png")
                    cropped_image_pil.save(file_path)
                    imglist.append(f"{element_type}_img{img_index + 1}_block{i + 1}.png")
                else:
                    cropped_image = image_rgb[y1:y2+30, x1:x2]
                    cropped_image_pil = Image.fromarray(cropped_image)
                    file_path = os.path.join(element_dirs[element_type], f"{element_type}_img{img_index + 1}_block{i + 1}.png")
                    cropped_image_pil.save(file_path)
                    chartlist.append(f"{element_type}_img{img_index + 1}_block{i + 1}.png")                
                
                print(f"Saved {element_type} element from image {img_index + 1} to {file_path}")
            else:
                print(f"Unknown element type: {element_type}, not saved.")
    
    imgdict = {}
    imgdict['folderpath'] = imagepath
    imgdict['imglist'] = imglist

    tabledict = {}
    tabledict['folderpath'] = tablepath
    tabledict['imglist'] = tablelist

    textdict = {}
    textdict['folderpath'] = textpath
    textdict['imglist'] = textlist

    chartdict = {}
    chartdict['folderpath'] = chartpath
    chartdict['imglist'] = chartlist
    
    result['image'] = imgdict
    result['text'] = textdict
    result['table'] = tabledict
    result['chart'] = chartdict

    return result
