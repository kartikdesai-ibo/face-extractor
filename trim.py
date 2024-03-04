from moviepy.editor import *
import os

os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/Cellar/ffmpeg/5.1/bin/ffmpeg"
     
# loading video dsa gfg intro video  
clip = VideoFileClip("sjr19.mp4")  
      
# getting only first 5 seconds  
clip = clip.subclip(3000, 3300)  


clip.write_videofile("trimmed.mp4")