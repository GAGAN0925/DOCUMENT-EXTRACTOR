import fitz
import os
import cv2
import docx
from bs4 import BeautifulSoup
import base64
import requests
from layoutparser.models import Detectron2LayoutModel
from PIL import Image
import pytesseract
from pytesseract import Output
import pandas as pd
import json
from py_client.utils import load_config
from detectron2.utils.logger import setup_logger
import logging
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
import numpy as np

config = load_config('py_client/config/config.yaml')

# Set the environment variable for torch cache
os.environ['TORCH_HOME'] = r'C:\Users\Public\pdfproject\torch_cache'

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Setup logger
setup_logger()

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Define function to set up configuration and load the trained model
def setup_predictor():
    # Configure the model for inference
    try:
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))  # Adjust if your model config is different
        cfg.MODEL.WEIGHTS = config['model_weights_path']  # Path to your trained model
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config["inference_threshold"]  # Set threshold for inference
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(config['classes']['class_names'])  # Update based on your dataset's number of classes (7 in your case)
        cfg.MODEL.DEVICE = "cpu"  # Force using CPU
        logging.info("Model setup successful.")        
        return DefaultPredictor(cfg)
    except Exception as e:
        return logging.error(f"An error occurred during model setup: {e}")

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def convert_pdf_to_images(pdf_path, output_folder, title):
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
        print(f"Title [{title}], Page {page_num + 1} saved of {doc.page_count}")
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

def convert_file_to_images(file_path,output_folder):    
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
    
def process_images_with_layoutparser(images_array, output_folder, sourcefile,title):
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

    imagepath = output_folder + 'extracted_image/'
    tablepath = output_folder + 'extracted_table/'
    textpath = output_folder + 'extracted_text/'
    chartpath = output_folder + 'extracted_chart/'    

    resultdf = pd.DataFrame(columns=['source','title','page_number','ocr_text','class','link'])
    sourcelst = []
    titlelst= []
    page_numberlst = [] 
    ocr_textlst = []
    classlst= []
    linklst = []                

    for dir_name in element_dirs.values():
        os.makedirs(dir_name, exist_ok=True)

    for img_index, image_rgb in enumerate(images_array):
        print(f"Processing image {img_index + 1}")
        layout = model.detect(image_rgb)

        for i, block in enumerate(layout):
            element_type = block.type
            ocr_text = ""            

            if element_type in element_dirs:
                x1, y1, x2, y2 = map(int, block.coordinates)                                

                if element_type in ['Text','Title','List']:
                    cropped_image = image_rgb[y1:y2, x1:x2]
                    cropped_image_pil = Image.fromarray(cropped_image)
                    file_path = os.path.join(element_dirs[element_type], f"{element_type}_img{img_index + 1}_block{i + 1}.png")
                    cropped_image_pil.save(file_path)
                    link = f"{element_type}_img{img_index + 1}_block{i + 1}.png"  
                    ocr_text = pytesseract.image_to_string(cropped_image, config='--psm 6')                              
                elif element_type == 'Table':
                    cropped_image = image_rgb[y1:y2+30, x1:x2]
                    cropped_image_pil = Image.fromarray(cropped_image)
                    file_path = os.path.join(element_dirs[element_type], f"{element_type}_img{img_index + 1}_block{i + 1}.png")
                    cropped_image_pil.save(file_path)
                    link = f"{element_type}_img{img_index + 1}_block{i + 1}.png"
                elif element_type == 'Figure':
                    cropped_image = image_rgb[y1:y2+30, x1:x2]
                    cropped_image_pil = Image.fromarray(cropped_image)
                    file_path = os.path.join(element_dirs[element_type], f"{element_type}_img{img_index + 1}_block{i + 1}.png")
                    cropped_image_pil.save(file_path)
                    link = f"{element_type}_img{img_index + 1}_block{i + 1}.png"
                else:
                    cropped_image = image_rgb[y1:y2+30, x1:x2]
                    cropped_image_pil = Image.fromarray(cropped_image)
                    file_path = os.path.join(element_dirs[element_type], f"{element_type}_img{img_index + 1}_block{i + 1}.png")
                    cropped_image_pil.save(file_path)
                    link = f"{element_type}_img{img_index + 1}_block{i + 1}.png"              
                
                print(f"Saved {element_type} element from image {img_index + 1} to {file_path}")
                sourcelst.append(sourcefile)
                titlelst.append(title)
                page_numberlst.append(img_index + 1)
                ocr_textlst.append(ocr_text)
                classlst.append(element_type)
                linklst.append(link)                            

            else:
                print(f"Unknown element type: {element_type}, not saved.")
    result = {
        "source":sourcelst,
        "title":titlelst,
        "page_number":page_numberlst,
        "ocr_text":ocr_textlst,
        "class":classlst,
        "link":linklst
        }
                
    df = pd.DataFrame(data=result)
    #resultdf = pd.concat([resultdf, df])  
    return df

def get_text_with_layoutparser(image):
    model = Detectron2LayoutModel(
        'lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure", 5: "Image", 6: "Chart"}  # Added Chart
    )
    layout = model.detect(image)   
    
    ocr_text = ""
    for block in layout:
        element_type = block.type                    

        if element_type in ['Text','Title','List']:
            x1, y1, x2, y2 = map(int, block.coordinates) 
            cropped_image = image[y1:y2, x1:x2]
            ocr_text += pytesseract.image_to_string(cropped_image, config='--psm 6') + " "
    return ocr_text


def process_files(df,outputfolder):
    Processed_df = pd.DataFrame(columns=['source','title','page_number','ocr_text','class','link'])    

    for i in range(len(df)):
        sourcefile = df.loc[i,"Source"]
        title = df.loc[i,"Title"]
        file_path = df.loc[i,"FileName"]   

        output_folder = f"{outputfolder}{file_path}/"
        os.makedirs(output_folder, exist_ok=True)

        file_path = 'data\\' + file_path + '.pdf'

        if os.path.exists(file_path):
            images_array = convert_file_to_images(file_path, output_folder)
            returndf = process_images_with_layoutparser(images_array,output_folder,sourcefile,title)
            Processed_df = pd.concat([Processed_df,returndf], ignore_index=True)
        else:
            print(f"File path [{file_path}] does not exist ")
    
    return Processed_df

def process_pdf_annie(df,outputfolder):
    Processed_df = pd.DataFrame(columns=['source','title','page_number','ocr_text','class','link'])    

    for i in range(len(df)):
        sourcefile = df.loc[i,"Source"]
        title = df.loc[i,"Title"]
        file_path = df.loc[i,"FileName"]   

        output_folder = f"{outputfolder}{file_path}/"
        os.makedirs(output_folder, exist_ok=True)

        file_path = 'data\\' + file_path + '.pdf'

        if os.path.exists(file_path):            
            returndf = process_with_annie(file_path,output_folder,sourcefile,title)
            Processed_df = pd.concat([Processed_df,returndf], ignore_index=True)
        else:
            print(f"File path [{file_path}] does not exist ")
    
    return Processed_df

def process_with_annie(pdf_path, output_dir,source, title):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load trained model
    predictor = setup_predictor()

    # Manually define class names as per your dataset
    class_names = config['classes']['class_names']
    text_classes = config['classes']['text_classes']
    image_classes = config['classes']['image_classes']

    # Create a folder for each class in the output directory
    for class_name in class_names:
        class_output_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

    resultdf = pd.DataFrame(columns=['source','title','page_number','ocr_text','class','link'])
    sourcelst = []
    titlelst= []
    page_numberlst = [] 
    ocr_textlst = []
    classlst= []
    linklst = []

    images = convert_pdf_to_images(pdf_path, output_dir, title)

    # Continue processing...
    print(f"Processing PDF: {pdf_path}")

    # Initialize text file and JSON structure
    ocr_text_file = os.path.join(output_dir, 'main.txt')
    json_output = []

    # Process each page in the PDF
    with open(ocr_text_file, 'w', encoding='utf-8') as ocr_file:
        for page_num, im in enumerate(images):
            # Convert PIL image to OpenCV format
            im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

            # Make predictions on the image
            outputs = predictor(im)
            instances = outputs["instances"].to("cpu")

            # JSON entry for the current page
            page_entry = {"page_number": page_num + 1, "instances": []}

            # Process each detected instance
            for i, box in enumerate(instances.pred_boxes):
                # Get the predicted class and corresponding label
                predicted_class = instances.pred_classes[i].item()
                class_name = class_names[predicted_class]

                # Get the bounding box for the instance
                bbox = box.numpy().astype(int).tolist()
                x1, y1, x2, y2 = bbox  # Unpack the bounding box coordinates

                # Extract the sub-image corresponding to the bounding box
                cropped_img = im[y1:y2, x1:x2]

                # Prepare instance entry for the JSON file
                instance_entry = {
                    "instance_id": i,
                    "class": class_name,
                    "bbox": bbox
                }

                ocr_text = ""
                output_image_path =""
                if class_name in text_classes:
                    # Perform OCR on the cropped image
                    ocr_text = pytesseract.image_to_string(cropped_img, config='--psm 6')

                    # Write to the text file
                    ocr_file.write(f"Page {page_num + 1}, Instance {i}, Class: {class_name}\n")
                    ocr_file.write(ocr_text + "\n\n")

                    # Add OCR text to JSON
                    instance_entry["ocr_text"] = ocr_text
                    image_output_dir = os.path.join(output_dir, class_name)
                    output_image_path = os.path.join(image_output_dir, f"page_{page_num + 1}instance{i}.jpg")
                    cv2.imwrite(output_image_path, cropped_img)
                    
                elif class_name in image_classes:
                    # Save the cropped image
                    image_output_dir = os.path.join(output_dir, class_name)
                    output_image_path = os.path.join(image_output_dir, f"page_{page_num + 1}instance{i}.jpg")
                    cv2.imwrite(output_image_path, cropped_img)

                    ocr_text = get_text_with_layoutparser(cropped_img)

                    # Write the image path to the text file
                    ocr_file.write(f"[Image saved at: {output_image_path}]\n\n")

                    # Add image path to JSON
                    instance_entry["image_path"] = output_image_path

                # Append the instance information to the page entry
                page_entry["instances"].append(instance_entry)

                sourcelst.append(source)
                titlelst.append(title)
                page_numberlst.append(page_num + 1)
                ocr_textlst.append(ocr_text)
                classlst.append(class_name)
                linklst.append(output_image_path)                 

            # Append the page entry to the JSON output
            json_output.append(page_entry)


            print(f"Processed page {page_num + 1} of {len(images)}.")
    
    result = {
        "source":sourcelst,
        "title":titlelst,
        "page_number":page_numberlst,
        "ocr_text":ocr_textlst,
        "class":classlst,
        "link":linklst
        }
                
    df = pd.DataFrame(data=result)

    # Save the JSON output to a file
    json_file_path = os.path.join(output_dir, 'main.json')
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(json_output, json_file, indent=4)
    
    return df
