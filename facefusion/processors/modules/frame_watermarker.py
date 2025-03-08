import math
from argparse import ArgumentParser
from typing import List

import cv2
import numpy as np

import facefusion.jobs.job_manager
import facefusion.jobs.job_store
import facefusion.processors.core as processors
from facefusion import process_manager, wording, config, state_manager
from facefusion.processors.typing import FrameWatermarkerInputs
from facefusion.typing import ApplyStateItem, Args, Face, ProcessMode, QueuePayload, UpdateProgress, VisionFrame
from facefusion.vision import read_image, read_static_image, write_image
from facefusion.processors import choices as processors_choices
from facefusion.program_helper import find_argument_group


def get_inference_pool() -> None:
	pass


def clear_inference_pool() -> None:
	pass


def register_args(program: ArgumentParser) -> None:
	group_processors = find_argument_group(program, 'processors')
	if group_processors:
		group_processors.add_argument('--frame-watermarker-model', help = wording.get('help.frame_watermarker_model'), default = config.get_str_value('processors.frame_watermarker_model', 'default'), choices = processors_choices.frame_watermarker_models)
		facefusion.jobs.job_store.register_step_keys([ 'frame_watermarker_model' ])


def apply_args(args: Args, apply_state_item: ApplyStateItem) -> None:
	apply_state_item('frame_watermarker_model', args.get('frame_watermarker_model'))


def pre_check() -> bool:
	return True


def pre_process(mode: ProcessMode) -> bool:
	if mode == 'preview':
		return False

	return True


def post_process() -> None:
	pass


def get_reference_frame(source_face: Face, target_face: Face, temp_vision_frame: VisionFrame) -> VisionFrame:
	pass


### WATERMARK MODULE HELPERS ###
def extrapolate_thikness(diag: float) -> int:
	a = 5 / 202
	b = -2.0792
	thickness = round(a * diag + b)

	# Reduce thickness for hard model by a factor of 3
	model = state_manager.get_item('frame_watermarker_model')
	if model != 'default':
		thickness = max(1, round(thickness / 10))

	return thickness


def can_fit_diagonally(inner_width: float, inner_height: float, outer_width: float, outer_height: float) -> bool:
	def fits(w: float, h: float, W: float, H: float) -> bool:
		return (w <= W and h <= H) or (w <= H and h <= W)

	inner_diag = math.sqrt(inner_width**2 + inner_height**2)
	outer_diag = math.sqrt(outer_width**2 + outer_height**2)

	if inner_diag > outer_diag:
		return False

	angle = math.atan2(inner_height, inner_width)
	cos_angle = math.cos(angle)
	sin_angle = math.sin(angle)

	width_rotated = inner_width * cos_angle + inner_height * sin_angle
	height_rotated = inner_width * sin_angle + inner_height * cos_angle

	return fits(width_rotated, height_rotated, outer_width, outer_height)


def get_optimal_font_scale(text: str, width: int, height: int) -> float:
	font_scale = 100.0
	font = cv2.FONT_HERSHEY_DUPLEX
	diag = np.sqrt(width**2 + height**2)
	thickness = extrapolate_thikness(diag)  # Text thickness

	while True:
		text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

		if can_fit_diagonally(text_size[0], text_size[1], width, height):
			return font_scale
		else:
			font_scale -= 1
### WATERMARK MODULE HELPERS ###


def prepare_text_frame(temp_vision_frame: VisionFrame) -> VisionFrame:
	height, width = temp_vision_frame.shape[:2]
	model = state_manager.get_item('frame_watermarker_model')

	text = 'DEEPFAKE' if model == 'default' else 'Purchase premium for just $1 to hide watermarks'
	font = cv2.FONT_HERSHEY_DUPLEX
	color = (255, 255, 255)

	font_scale = get_optimal_font_scale(text, width, height)
	angle = np.degrees(np.arctan2(height, width))
	diag_length = int(np.sqrt(width**2 + height**2))
	thickness = extrapolate_thikness(diag_length)
	text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
	# Create an empty layer for text
	text_img: VisionFrame = np.zeros((diag_length, diag_length, 3), dtype=np.uint8)
	# Text starting coordinates
	x_start = (diag_length - text_size[0]) // 2
	y_start = (diag_length + text_size[1]) // 2
	# Add text to the empty layer
	text_img = cv2.putText(text_img, text, (x_start, y_start), font, font_scale, color, thickness, cv2.LINE_AA)
	# Rotate the text layer by the calculated angle
	M: VisionFrame = cv2.getRotationMatrix2D((diag_length // 2, diag_length // 2), angle, 1)
	rotated_text: VisionFrame = cv2.warpAffine(text_img, M, (diag_length, diag_length))
	# Crop the text to the original frame size
	x_offset = (diag_length - width) // 2
	y_offset = (diag_length - height) // 2
	cropped_text:VisionFrame = rotated_text[y_offset:y_offset+height, x_offset:x_offset+width]

	return cropped_text


def watermark_frame(temp_vision_frame: VisionFrame, source_watermark_frame: VisionFrame) -> VisionFrame:
	temp_vision_frame = temp_vision_frame.copy()

	model = state_manager.get_item('frame_watermarker_model')
	opacity = 0.03 if model == 'default' else 0.6
	temp_vision_frame = cv2.addWeighted(temp_vision_frame, 1, source_watermark_frame, opacity, 0)

	return temp_vision_frame


def process_frame(inputs: FrameWatermarkerInputs) -> VisionFrame:
	target_vision_frame: VisionFrame = inputs['target_vision_frame']
	source_watermark_frame: VisionFrame = inputs.get('source_watermark_frame')

	return watermark_frame(target_vision_frame, source_watermark_frame)


def process_frames(source_paths: List[str], queue_payloads: List[QueuePayload], update_progress: UpdateProgress) -> None:
	_ = source_paths
	watermark_frame = None

	for queue_payload in process_manager.manage(queue_payloads):
		target_vision_path = queue_payload['frame_path']
		target_vision_frame = read_image(target_vision_path)
		if watermark_frame is None:
			watermark_frame = prepare_text_frame(target_vision_frame)
		output_vision_frame = process_frame(
		{
			'target_vision_frame': target_vision_frame,
			'source_watermark_frame': watermark_frame
		})
		write_image(target_vision_path, output_vision_frame)
		update_progress(1)


def process_image(source_paths: List[str], target_path: str, output_path: str) -> None:
	_ = source_paths

	target_vision_frame = read_static_image(target_path)
	watermark_frame = prepare_text_frame(target_vision_frame)
	output_vision_frame = process_frame(
	{
		'target_vision_frame': target_vision_frame,
		'source_watermark_frame': watermark_frame
	})
	write_image(output_path, output_vision_frame)


def process_video(source_paths: List[str], temp_frame_paths: List[str]) -> None:
	_ = source_paths

	processors.multi_process_frames(None, temp_frame_paths, process_frames)
