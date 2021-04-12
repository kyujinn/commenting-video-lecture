from scenedetect import VideoManager, SceneManager, StatsManager
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import save_images, write_scene_list_html

# !!!!!! Set Video Path before run this code !!!!!!
video_path = "./data/video_datamining.mp4"
#video_path = "./data/video3.mp4"
stats_path = 'result1.csv'

video_manager = VideoManager([video_path])
stats_manager = StatsManager()
scene_manager = SceneManager(stats_manager)

scene_manager.add_detector(ContentDetector(threshold=1))

video_manager.set_downscale_factor()

video_manager.start()
scene_manager.detect_scenes(frame_source=video_manager)

# result
with open(stats_path, 'w') as f:
    stats_manager.save_to_csv(f, video_manager.get_base_timecode())

scene_list = scene_manager.get_scene_list()
print(f'{len(scene_list)} scenes detected!')

save_images(
    scene_list,
    video_manager,
    num_images=1,
    image_name_template='$SCENE_NUMBER',
    output_dir='scenes1')

write_scene_list_html('result1.html', scene_list)

for scene in scene_list:
    start, end = scene

    # your code
    print(f'{start.get_seconds()} - {end.get_seconds()}')
