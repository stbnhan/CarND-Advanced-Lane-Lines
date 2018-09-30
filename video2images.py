import cv2

fname_out = []
n_frame = 0

def video_pipeline(img):
	global n_frame
	n_frame += 1
	fname_out = "./video/" + str(n_frame) + '.jpg'
	cv2.imwrite(fname_out, img)
	return img

from moviepy.editor import VideoFileClip

video_output = 'P4_video_final.mp4'
clip = VideoFileClip('project_video.mp4')
output_clip = clip.fl_image(video_pipeline)
output_clip.write_videofile(video_output, audio=False)
