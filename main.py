from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, request, jsonify
import torch
from diffusers import StableDiffusionPipeline
import base64
from io import BytesIO
import os
from werkzeug.utils import secure_filename
import easyocr
import cv2
import numpy as np
import urllib.request
import requests
import glob
from moviepy.editor import *
from concurrent.futures import ThreadPoolExecutor
import subprocess

# Load model
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", revision="fp16", torch_dtype=torch.float16)
pipe.to("cuda")

# Start flask app and set to ngrok
app = Flask(__name__)
run_with_ngrok(app)

app.config['IMAGES_PATH'] = '/content/drive/MyDrive/image/1688'
app.config['VIDEO_PATH'] = '/content/drive/MyDrive/video/1688'
app.config['UPLOAD_FOLDER'] = '/content/drive/MyDrive/upload'
app.config['FONTS_PATH'] = '/content/drive/MyDrive/fonts'

image_orginal_path = os.path.join(app.config['IMAGES_PATH'], f"original")
rm_text_path = os.path.join(app.config['IMAGES_PATH'], f"rm-text")
resized_path = os.path.join(app.config['IMAGES_PATH'], f"resized")
resized_info_path = os.path.join(resized_path, f"info")
resized_noninfo_path = os.path.join(resized_path, f"noninfo")
info_path = os.path.join(app.config['IMAGES_PATH'], f"info")

video_resized_path = os.path.join(app.config['VIDEO_PATH'], f"resized/info")
video_info_path =  os.path.join(app.config['VIDEO_PATH'], f"info")
video_item_resized_path = os.path.join(app.config['VIDEO_PATH'], f"item-resized") 
video_item_noninfo_path = os.path.join(app.config['VIDEO_PATH'], f"noninfo") 
video_mixed_path = os.path.join(app.config['VIDEO_PATH'], f"mixed")
video_mixed_musix_path = os.path.join(app.config['VIDEO_PATH'], f"mixed-music")

if not os.path.exists(image_orginal_path):
    os.makedirs(image_orginal_path)
if not os.path.exists(rm_text_path):
    os.makedirs(rm_text_path)
if not os.path.exists(resized_path):
    os.makedirs(resized_path)
if not os.path.exists(resized_info_path):
    os.makedirs(resized_info_path)
if not os.path.exists(resized_noninfo_path):
    os.makedirs(resized_noninfo_path)
if not os.path.exists(info_path):
    os.makedirs(info_path)   
if not os.path.exists(video_resized_path):
    os.makedirs(video_resized_path)
if not os.path.exists(video_info_path):
    os.makedirs(video_info_path)
if not os.path.exists(video_item_resized_path):
    os.makedirs(video_item_resized_path)
if not os.path.exists(video_item_noninfo_path):
    os.makedirs(video_item_noninfo_path)
if not os.path.exists(video_mixed_path):
    os.makedirs(video_mixed_path)
if not os.path.exists(video_mixed_musix_path):
    os.makedirs(video_mixed_musix_path)

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
    
# @app.route('/')
# def initial():
#   return render_template('index.html')


# @app.route('/submit-prompt', methods=['POST'])
# def generate_image():
#   prompt = request.form['prompt-input']
#   print(f"Generating an image of {prompt}")

#   image = pipe(prompt).images[0]
#   print("Image generated! Converting image ...")
  
#   buffered = BytesIO()
#   image.save(buffered, format="PNG")
#   img_str = base64.b64encode(buffered.getvalue())
#   img_str = "data:image/png;base64," + str(img_str)[2:-1]

#   print("Sending image ...")
#   return render_template('index.html', generated_image=img_str)

# Initialize EasyOCR for text recognition (both Chinese and English)
reader = easyocr.Reader(['ch_sim'])

def remove_text(input_image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Extract text and its location from the image
    result = reader.readtext(gray, paragraph=False)

    # Create a mask for inpainting (initially all zeros)
    mask = np.zeros_like(gray, dtype=np.uint8)

    # Iterate over the extracted text and create the inpainting mask
    for bbox, _, _ in result:
        poly = np.array(bbox).astype(np.int32)
        cv2.fillPoly(mask, [poly], 255)

    # Inpaint the image using the Navier-Stokes based method
    inpaint_radius = 5  # Increase this value to blend the removed text area better with the background
    img_inpainted = cv2.inpaint(input_image, mask, inpaint_radius, cv2.INPAINT_NS)

    return img_inpainted

@app.route('/remove-text', methods=['POST'])
def process_images():
    data = request.json
    item_id = data['item_id']
    images_type = data['images_type'] 
    images_url = data['images_url']
    rm_text_path = os.path.join(app.config['IMAGES_PATH'], f"rm-text/{images_type}")
    rm_text_path_folder = os.path.join(rm_text_path, str(item_id))
    if not os.path.exists(os.path.join(rm_text_path_folder)):
        os.makedirs(rm_text_path_folder)
     # Download all the images to rm_text_path_folder
    image_files = download_images_concurrently(images_url, rm_text_path_folder)
    output_files = []
    for i, filepath in enumerate(image_files):
        try:
            img = cv2.imread(filepath)  # Read the image using cv2.imread()
             # Extract text and its location from the image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            result = reader.readtext(gray, paragraph=False)
             # Skip image if no text is found
            if not result:
                output_files.append(os.path.join(rm_text_path_folder, os.path.basename(filepath)))
                continue
            img_inpainted = remove_text(img)
            output_path = os.path.join(rm_text_path_folder, os.path.basename(filepath))
            cv2.imwrite(output_path, img_inpainted)
            output_files.append(output_path)
        except Exception as e:
            print(f"Error processing image {filepath}: {e}")
            output_files.append(None)
     # Print the output files before returning the JSON response
    return jsonify({"output_files": output_files})
def download_image(url, index, folder):
    response = requests.get(url)
    image_file = os.path.join(folder, f'{index}.jpg')
    with open(image_file, 'wb') as f:
        f.write(response.content)
    return image_file
def download_images_concurrently(urls, folder):
    image_files = []
    with ThreadPoolExecutor() as executor:
        image_files = list(executor.map(download_image, urls, range(len(urls)), [folder]*len(urls)))
    return image_files

@app.route('/generate-video', methods=['POST'])
def generate_video():
    
    data = request.json
    
    item_id = data['item_id']
    w = data['w']
    h = data['h']
    
    # images_url = data['images_url']
    captions_titles = data['captions_titles']
    captions_options_titles = data['captions_options_titles']
    captions_options = data['captions_options']
    captions_price = data['captions_price']
    
    # image_files = download_images_concurrently(images_url)
    
    # base_folder = f"resized/info/{w}-{h}"

    image_path = os.path.join(app.config['IMAGES_PATH'], f"resized/info/{w}-{h}/{item_id}")

    # image_folder = os.path.join(image_path, f"{item_id}/")

    image_files = glob.glob(os.path.join(image_path, "*.jpg"))

    num_images = len(image_files)
    
    total_duration = 30
    
    duration_per_image = total_duration / num_images
    
    durations = [duration_per_image] * num_images

    image_clips = [create_image_clip(image_file, duration) for image_file, duration in zip(image_files, durations)]

    screen_size = (image_clips[0].size)

    # moving_text_clips = [create_moving_text(captions_titles[i], font_size, durations[i], screen_size, text_color='white', bg_color=(0, 0, 0), speed=0.6, padding=8) for i in range(len(captions_titles))]
    moving_text_clips = [create_moving_text(captions_titles[i], captions_options_titles[i], captions_options[i], captions_price[i], durations[i], screen_size, text_color='white', bg_color=(0, 0, 0), speed=0.6, padding=5) for i in range(len(captions_titles))]
   
    composite_clips = []

    for i in range(len(image_files)):
        img_clip = image_clips[i].subclip(0, durations[i])
        composite_clip = CompositeVideoClip([img_clip, moving_text_clips[i]])
        composite_clips.append(composite_clip)
        
    final_video = concatenate_videoclips(composite_clips)
    
    save_path = os.path.join(app.config['VIDEO_PATH'], f"info/{w}-{h}")
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    output_path = os.path.join(save_path, f"{item_id}.mp4")
    
    final_video.write_videofile(output_path, fps=24)
    
    print("Output files:", output_path)
    
    return jsonify({"output_files": output_path}, 200)


def create_image_clip(image_file, duration):
    return ImageClip(image_file).set_duration(duration)

def create_static_text(text, font_size, duration, screen_size, position='top'):
    text_clip = TextClip(text, fontsize=font_size, color='white')
    text_clip = text_clip.set_duration(duration)
    
    if position == 'top':
        text_position = (screen_size[0] - text_clip.size[0]) // 8, screen_size[1] - text_clip.size[1] - 15
        
    elif position == 'top':
        text_position = (screen_size[0] - text_clip.size[0]) // 8, 8
    else:
        text_position = (screen_size[0] - text_clip.size[0]) // 8, (screen_size[1] - text_clip.size[1]) // 8

    text_clip = text_clip.set_position(text_position)
    return text_clip

def create_moving_text(title, option_title, option, price, duration, screen_size, text_color='white', bg_color=(0, 0, 0), speed=0.4, padding=1):
    font_path = os.path.join(app.config['FONTS_PATH'], f"th-th/NotoSansThai-Regular.ttf")  # Change this to the path of the SimHei font on your system
    font_sizes = [25, 28, 28, 33] # font sizes for title, option_title, option, and price

    lines = [
        TextClip(title, fontsize=font_sizes[0], color=text_color, font=font_path),
        TextClip(option_title, fontsize=font_sizes[1], color=text_color, font=font_path),
        TextClip(option, fontsize=font_sizes[2], color=text_color, font=font_path),
        TextClip(price, fontsize=font_sizes[3], color=text_color, font=font_path)
    ]

    # Calculate the total height of the text block
    
    total_height = sum([l.h for l in lines]) + 10 * padding

    # Position the lines vertically with padding and centered horizontally
    line_positions = [((screen_size[0] - line.w) // 2, i * (h + padding)) for i, (line, h) in enumerate(zip(lines, [l.h for l in lines]))]
    positioned_lines = [line.set_position(pos) for line, pos in zip(lines, line_positions)]

    # Create a composite clip with the positioned lines
    lines_composite = CompositeVideoClip(positioned_lines, size=(screen_size[0], total_height))

    # Create a colored background for the text block
    txt_bg = lines_composite.on_color(size=(screen_size[0] + padding * 1, total_height + padding * 1),
                                      color=bg_color, pos=('center', 'center'), col_opacity=0.5)

    # Set the moving position for the text block
    txt_mov = txt_bg.set_position(lambda t: ((screen_size[0] - txt_bg.w) // 4,
                                             min(screen_size[1] // 8, int(screen_size[1] * t / 10))))
    txt_mov = txt_mov.set_duration(duration)
    return txt_mov

# def download_image(url, index):
#     response = requests.get(url)
#     image_file = f'image{index}.jpg'
#     with open(image_file, 'wb') as f:
#         f.write(response.content)
#     return image_file

# def download_images_concurrently(urls):
#     image_files = []
#     with ThreadPoolExecutor() as executor:
#         image_files = list(executor.map(download_image, urls, range(len(urls))))
#     return image_files
# generate video end
@app.route('/resize-1688-item-video', methods=['POST']) 
def resize_1688_item_video(): 
    data = request.json 
    item_id = data['item_id']
    w = data['w']
    h = data['h']
    
    item_video_url = data['video_url']
    
    output_size = os.path.join(app.config['VIDEO_PATH'], f"item-resized/{w}-{h}")
    
    output_path = os.path.join(output_size, f"{item_id}.mp4")

    if not os.path.exists(os.path.join(output_size)):
        os.makedirs(output_size)
        
    cmd = f'ffmpeg -i {item_video_url} -vf "scale=w={w}:h={h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black" -y {output_path}'
    
    subprocess.run(cmd, shell=True)

    return jsonify({'message': 'Resize Item Video successfully!'})
  
@app.route('/resize-1688-item-image', methods=['POST']) 
def resize_1688_item_image(): 
    data = request.json 
    item_id = data['item_id']
    w = data['w']
    h = data['h']
    images_type = data['images_type']
    
    size_path = os.path.join(app.config['IMAGES_PATH'], f"resized/{images_type}/{w}-{h}")
    
    if not os.path.exists(os.path.join(size_path)):
        os.makedirs(size_path)
        
    output_path = os.path.join(size_path, f"{item_id}")
    
    if not os.path.exists(os.path.join(output_path)):
        os.makedirs(output_path)
    
    item_images_data = glob.glob(os.path.join(app.config['IMAGES_PATH'], f"rm-text/{images_type}/{item_id}/*jpg"))
                                 
    for index, image in enumerate(item_images_data):
        output_path_file = f"{output_path}/{index}.jpg"

        cmd = f'ffmpeg -i {image} -vf "scale=w={w}:h={h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black" -y {output_path_file}' 
        subprocess.run(cmd, shell=True)
    return jsonify({'message': 'Resize Images successfully!'})

    
@app.route('/noninfo-images-to-video-slider', methods=['POST']) 
def noninfo_images_to_video_slider(): 
    data = request.json 
    item_id = data['item_id']
    w = data['w']
    h = data['h']
    
    video_size_path = os.path.join(app.config['VIDEO_PATH'], f"noninfo/{w}-{h}")
    
    video_name = os.path.join(video_size_path, f"{item_id}.mp4")

    if not os.path.exists(os.path.join(video_size_path)):
        os.makedirs(video_size_path)

    images_path = os.path.join(app.config['IMAGES_PATH'],f"resized/noninfo/{w}-{h}/{item_id}/")

    images = glob.glob(os.path.join(images_path, '*.jpg'))

    num_images = len(images)

    video_long = num_images * 2

    iframrate = num_images/video_long
    
    cmd = f'ffmpeg -framerate {iframrate} -i {images_path}/%d.jpg -c:v libx264 -r 25 -pix_fmt yuv420p -y -t {video_long} {video_name}'
    
    subprocess.run(cmd, shell=True)

    return jsonify({'message': 'Resize Item Video successfully!'})

@app.route('/item-images-info-resize', methods=['POST'])
def item_images_info_resize(): 
    data = request.json
    item_id = data['item_id']
    w = data['w']
    h = data['h']
    
    captions_titles = data['captions_titles']
    captions_options_titles = data['captions_options_titles']
    captions_options = data['captions_options']
    captions_price = data['captions_price']
    
    size_path = os.path.join(app.config['IMAGES_PATH'], f"info/resized/{w}-{h}")
        
    output_path = os.path.join(size_path, f"{item_id}")
    
    if not os.path.exists(os.path.join(size_path)):
        os.makedirs(size_path)
        
    if not os.path.exists(os.path.join(output_path)):
        os.makedirs(output_path)
    
    font_size_titles = 30
    font_size_options_titles = 20
    font_size_options = 24
    font_size_price = 40
    price_color = "white"

    text_color = "white"
    
    background_color    = "000000@0.4"
    
    images = glob(app.config['IMAGES_PATH'],F"resized/info/{w}-{h}/{item_id}/*jpg")
    
    font_path = "fonts/th-th/NotoSansThai-Regular.ttf"
   
    for index, image in enumerate(images):
      
        output_path_file = f"{output_path}/{index}.jpg"
        
        text_captions_titles = captions_titles[index]
        text_captions_options_titles = captions_options_titles[index]
        text_captions_options = captions_options[index]
        text_captions_price = captions_price[index]
        
        cmd = f'ffmpeg -i {image} -vf "drawtext=text=\'{text_captions_titles}\':fontfile={font_path}:fontsize={font_size_titles}:fontcolor={text_color}:x=(w-text_w)/2:y=30:box=1:boxcolor={background_color}:boxborderw=5,drawtext=text=\'{text_captions_options_titles}\':fontfile={font_path}:fontsize={font_size_options_titles}:fontcolor={text_color}:x=(w-text_w)/2:y=68:box=1:boxcolor={background_color}:boxborderw=5,drawtext=text=\'{text_captions_options}\':fontfile={font_path}:fontsize={font_size_options}:fontcolor={text_color}:x=(w-text_w)/2:y=98:box=1:boxcolor={background_color}:boxborderw=5,drawtext=text=\'Price: {text_captions_price}\':fontfile={font_path}:fontsize={font_size_price}:fontcolor={price_color}:x=(w-text_w)/2:y=140:box=1:boxcolor={background_color}:boxborderw=5" -y {output_path_file}'
        
        subprocess.run(cmd, shell=True)
        
    return jsonify({'message': 'Resize Images successfully!'})
@app.route('/ffmpeg-mixing-video', methods=['POST']) 
def mixing_videos(): 
    data = request.json 
    item_id = data['item_id'] 
    w = data['w'] 
    h = data['h'] 
    input_1 = os.path.join(app.config['VIDEO_PATH'], f"item-resized/{w}-{h}/{item_id}.mp4") 
    input_3 = os.path.join(app.config['VIDEO_PATH'], f"info/{w}-{h}/{item_id}.mp4") 
    input_2 = os.path.join(app.config['VIDEO_PATH'], f"noninfo/{w}-{h}/{item_id}.mp4") 
    output1_size = os.path.join(app.config['VIDEO_PATH'], f"mixed/{w}-{h}")
    output2_size = os.path.join(app.config['VIDEO_PATH'], f"mixed-music/{w}-{h}")
    output1 = os.path.join(output1_size, f"{item_id}.mp4")
    output2 = os.path.join(output2_size, f"{item_id}.mp4")
    input_audio = f""
    
    if not os.path.exists(os.path.join(output1_size)):
        os.makedirs(output1_size)
    if not os.path.exists(os.path.join(output2_size)):
        os.makedirs(output2_size)
        
    if os.path.exists(input_1):
        cmd = f'ffmpeg -i {input_1} -i {input_2} -i {input_3} -filter_complex "[0:v]setpts=\'(if(gt(TB,55),55/TB,1))*PTS\'[v0];[1:v]setpts=\'(if(gt(TB,55),55/TB,1))*PTS\'[v1];[2:v]setpts=\'(if(gt(TB,55),55/TB,1))*PTS\'[v2];[v0][v1][v2]concat=n=3:v=1:a=0" -an -t 55 -y {output1}'
        subprocess.run(cmd, shell=True)
    else:
        cmd = f'ffmpeg -i {input_2} -i {input_3} -filter_complex "[0:v]setpts=\'(if(gt(TB,55),55/TB,1))*PTS\'[v0];[1:v]setpts=\'(if(gt(TB,55),55/TB,1))*PTS\'[v1];[v0][v1]concat=n=2:v=1:a=0" -an -t 55 -y {output1}'
        subprocess.run(cmd, shell=True)
        
    if os.path.exists(output1):
        cmd = f'ffmpeg -y -i {output1} -i {input_audio} -c:v copy -map 0:v:0 -map 1:a:0 -c:a copy -shortest -f mp4 {output2}'
        subprocess.run(cmd, shell=True)
    else:
        return jsonify({'no-video': 'Have No Item-Resize Video!'})
    
    return jsonify({'message': 'Video mixing successfully!'})
    # cmd = f'ffmpeg -i {input_1} -i {input_2} -i {input_3} -i {input_4} -filter_complex "[0:v]setpts=\'(if(gt(TB,55),55/TB,1))*PTS\'[v0];[1:v]setpts=\'(if(gt(TB,55),55/TB,1))*PTS\'[v1];[2:v]setpts=\'(if(gt(TB,55),55/TB,1))*PTS\'[v2];[3:v]setpts=\'(if(gt(TB,55),55/TB,1))*PTS\'[v3];[v0][v1][v2][v3]concat=n=4:v=1:a=0" -an -t 55 -y {output}'sdf
    """
Certainly! This is a command-line command that uses the FFmpeg software to concatenate two video files and output the result as a single video file. Here's what each part of the command does:
 -  `ffmpeg` : This is the FFmpeg command-line tool that does the video processing.
-  `-i {input_2} -i {input_3}` : This specifies the two input video files that will be concatenated.  `{input_2}`  and  `{input_3}`  are variables that should be replaced with the actual file paths.
-  `-filter_complex` : This is an FFmpeg option that allows us to use complex filtergraphs to apply different filters and effects to the input streams. In this case, we're using it to concatenate the two inputs. 
-  `"[0:v]setpts=\'(if(gt(TB,55),55/TB,1))*PTS\'[v0];[1:v]setpts=\'(if(gt(TB,55),55/TB,1))*PTS\'[v1];[v0][v1]concat=n=2:v=1:a=0"` : This complex filtergraph has three parts, separated by semicolons. The first two parts set the PTS (presentation timestamps) of the video frames to speed up or slow down the video as needed to make it match a maximum duration of 55 seconds ( `-t 55` ). The third part concatenates the two video streams together, using  `concat=n=2:v=1:a=0`  to specify that we want to concatenate two video streams ( `n=2:v=1:a=0`  means "2 videos, 1 video stream per input, 0 audio streams per input").  `[v0]`  and  `[v1]`  are temporary labels used to refer to the output of the first and second parts of the filtergraph.
-  `-an` : This option tells FFmpeg to remove any audio streams from the output video file.
-  `-t 55` : This option sets a maximum duration of 55 seconds for the output video. If the concatenated video is longer than 55 seconds, it will be trimmed.
-  `-y {output}` : This specifies the output file path and tells FFmpeg to overwrite it if it already exists.  `{output}`  is a variable that should be replaced with the actual file path.
 Overall, this command takes two input video files, speeds them up or slows them down as needed to make them fit into a 55-second duration, concatenates them into a single video file, and removes any audio streams from the output. The resulting video file will be saved at the specified output path.
    """
@app.route('/ffmpeg-mixing-end', methods=['POST']) 
def mixing_end(): 
    data = request.json 
    item_id = data['item_id']
    w = data['w']
    h = data['h']
    input_1 = os.path.join(app.config['VIDEO_PATH'], f"mixed/{w}-{h}/{item_id}.mp4")
    input_2 = os.path.join(app.config['VIDEO_PATH'], f"end/th-th/{w}-{h}.mp4")
    output_size = os.path.join(app.config['VIDEO_PATH'], f"mixed-end/{w}-{h}")
    output = os.path.join(output_size, f"{item_id}.mp4")
   
    if not os.path.exists(os.path.join(output_size)):
        os.makedirs(output_size)
    
    if os.path.exists(input_1):
        cmd = f'ffmpeg -i {input_1} -i {input_2} -filter_complex "[0:v] [0:a] [1:v] [1:a] concat=n=2:v=1:a=1" {output}' 
        subprocess.run(cmd, shell=True)
    else:
        return jsonify({'no-video': 'Have No Mixed Video!'})
    
    return jsonify({'message': 'End Video mixed successfully!'})
  

if __name__ == '__main__':
    app.run()
    
    
