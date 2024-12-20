
import os
import gradio as gr
import argparse
# Custom
from app_core import GlobalData
# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=3000, help="port")
parser.add_argument("--video_dir", type=str, default="test", help="video directory")
parser.add_argument("--anno_dir", type=str, default="test", help="annotation directory")
parser.add_argument("--vis_dir", type=str, default="test", help="visualization directory")
args = parser.parse_args()
# check arguments
assert args.port != 3000, "Please specify the port."
assert args.video_dir != "test", "Please specify the video directory."
assert args.anno_dir != "test", "Please specify the save directory."
assert args.vis_dir != "test", "Please specify the visualization directory."
# Global data
global_data = GlobalData(
    video_dir=args.video_dir,
    anno_dir=args.anno_dir,
    vis_dir=args.vis_dir,
)
# Main
def main(args):
    video_names = os.listdir(args.video_dir)
    video_names.sort()
    color_map = global_data.get_hex_pallete()
    with gr.Blocks() as demo:
        # with gr.Column():
        #     with gr.Row():
        #         info_label = gr.Label(value={
        #             "Frame": 0.0,
        #             "Mask": 0.0
        #         }, show_label=False, num_top_classes=2)
        #         info_text = gr.HighlightedText(label="Info", combine_adjacent=False, show_legend=False, show_label=False)
        #     with gr.Row():
        #         gr.Button("Current Frame")
        #         gr.Button("Previous Frame")
        #         gr.Button("First Frame")
        #     with gr.Row():
        #         segmented_curr_frame = gr.Image(type="numpy", label="current frame", show_label=False, show_download_button=False)
        #         segmented_prev_frame = gr.Image(type="numpy", label="previous frame", show_label=False, show_download_button=False)
        #         segmented_first_frame = gr.Image(type="numpy", label="first frame", show_label=False, show_download_button=False)
        #     with gr.Row():
        #         accept_mask_button = gr.Button("Accept mask")
        #         reject_mask_button = gr.Button("Reject mask")
        #         previous_mask_button = gr.Button("Previous mask")
        #     with gr.Row():
        #         with gr.Column():
        #             undo_point_button = gr.Button("Undo point")
        #             previous_frame_button = gr.Button("Previous frame")
        #         point_type = gr.Radio(["foreground point", "background point"], label="Select point type")
        #         alpha = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, label="Alpha")
        #     with gr.Row():
        #         video_name = gr.Dropdown(
        #             choices=video_names,
        #             label="select video name",
        #         )
        #         selected_mask_idx = gr.Number(value=-1, label="Mask index")
        with gr.Column():
            # with gr.Row():
            #     info_label = gr.Label(value={
            #         "Frame": 0.0,
            #         "Mask": 0.0
            #     }, show_label=False, num_top_classes=2)
            #     info_text = gr.HighlightedText(label="Info", combine_adjacent=False, show_legend=False, show_label=False)
            with gr.Row():
                gr.Button(
                    value="Previous Frame",
                    icon="/home/cvlab14/project/seongchan/automatic-annotator/code/figs/frame.png",
                )
                gr.Button(
                    value="First Frame",
                    icon="/home/cvlab14/project/seongchan/automatic-annotator/code/figs/frame.png",
                )
            with gr.Row():
                segmented_prev_frame = gr.AnnotatedImage(
                    color_map=color_map,
                    show_label=False,
                    show_legend=False,
                )
                segmented_first_frame = gr.AnnotatedImage(
                    color_map=color_map,
                    show_label=False,
                    show_legend=False,
                )
            with gr.Row():
                with gr.Column():
                    gr.Button(
                        value="Current Frame",
                        icon="/home/cvlab14/project/seongchan/automatic-annotator/code/figs/frame.png",
                    )
                    segmented_curr_frame = gr.Image(type="numpy", label="current frame", show_label=False, show_download_button=False)
                    color_map = gr.Gallery(
                        show_download_button=False,
                        show_label=False,
                        show_share_button=False,
                        columns=[5],
                        rows=[1],
                        allow_preview=False,
                        preview=False,
                        interactive=False,
                        height=150,
                    )
                with gr.Column():
                    gr.Button(
                        value="Control",
                        icon="/home/cvlab14/project/seongchan/automatic-annotator/code/figs/setting.png",
                    )
                    info_label = gr.Label(value={
                        "Frame": 0.0,
                        "Mask": 0.0
                    }, show_label=False, num_top_classes=2)
                    with gr.Row():
                        undo_point_button = gr.Button(
                            value="Undo point",
                            icon="/home/cvlab14/project/seongchan/automatic-annotator/code/figs/undo.png",
                        )
                        reject_mask_button = gr.Button(
                            value="Reject mask",
                            icon="/home/cvlab14/project/seongchan/automatic-annotator/code/figs/reject.png",
                        )
                    with gr.Row():
                        with gr.Column():
                            accept_mask_button = gr.Button(
                                value="Accept mask", 
                                icon="/home/cvlab14/project/seongchan/automatic-annotator/code/figs/accept.png",
                            )
                            previous_mask_button = gr.Button(
                                value="Previous mask",
                                icon="/home/cvlab14/project/seongchan/automatic-annotator/code/figs/prev.png",
                            )
                            previous_frame_button = gr.Button(
                                value="Previous frame",
                                icon="/home/cvlab14/project/seongchan/automatic-annotator/code/figs/prev.png",
                            )
                        point_type = gr.Radio(["foreground point", "background point"], label="Select point type")
                        sam_mask_size = gr.Radio(["small", "medium", "large"], label="Select mask size")
                    info_text = gr.HighlightedText(label="Info", combine_adjacent=False, show_legend=False, show_label=False)
                    with gr.Row():
                        video_name = gr.Dropdown(
                            choices=video_names,
                            label="select video name",
                        )
                        selected_mask_idx = gr.Number(value=-1, label="Mask index")
                        alpha = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, label="Alpha")
        # 1. Select video to annotate.
        video_name.change(
            fn=start_video,
            inputs=[video_name, alpha, selected_mask_idx],
            outputs=[segmented_curr_frame, segmented_prev_frame, segmented_first_frame, info_text, info_label, color_map],
        )
        selected_mask_idx.change(
            fn=start_video,
            inputs=[video_name, alpha, selected_mask_idx],
            outputs=[segmented_curr_frame, segmented_prev_frame, segmented_first_frame, info_text, info_label, color_map],
        )
        # 2. Accept or reject mask at current frame.
        # 2-1. Accept mask.
        accept_mask_button.click(
            fn=accept_mask,
            inputs=[alpha],
            outputs=[segmented_curr_frame, segmented_prev_frame, segmented_first_frame, info_text, info_label],
        )
        # 2-2. Reject mask.
        reject_mask_button.click(
            fn=reject_mask,
            inputs=[alpha],
            outputs=[segmented_curr_frame, segmented_prev_frame, segmented_first_frame, info_text, info_label],
        )
        previous_mask_button.click(
            fn=previous_mask,
            inputs=[alpha],
            outputs=[segmented_curr_frame, segmented_prev_frame, segmented_first_frame, info_text, info_label],
        )
        # 3. If you reject mask at current frame,
        #    you should prompt points to generate masks.
        segmented_curr_frame.select(
            fn=get_point,
            inputs=[point_type, alpha],
            outputs=[segmented_curr_frame, segmented_prev_frame, segmented_first_frame, info_text, info_label],
        )
        # 3-1. Undo point.
        undo_point_button.click(
            fn=undo_point,
            inputs=[alpha],
            outputs=[segmented_curr_frame, segmented_prev_frame, segmented_first_frame, info_text, info_label],
        )
        # 4. If you want to go back to the previous frame.
        previous_frame_button.click(
            fn=previous_frame,
            inputs=[alpha],
            outputs=[segmented_curr_frame, segmented_prev_frame, segmented_first_frame, info_text, info_label],
        )
        # 5. Select Color Map.
        color_map.select(
            fn=select_color_map,
            inputs=[video_name, alpha],
            outputs=[segmented_curr_frame, segmented_prev_frame, segmented_first_frame],
        )
        alpha.change(
            fn=change_alpha,
            inputs=[video_name, alpha],
            outputs=[segmented_curr_frame, segmented_prev_frame, segmented_first_frame],
        )
        # 6. Change SAM mask size.
        sam_mask_size.change(
            fn=change_sam_mask_size,
            inputs=[sam_mask_size, alpha],
            outputs=[segmented_curr_frame, segmented_prev_frame, segmented_first_frame],
        )
    # Launch the demo.
    demo.queue().launch(debug=True, server_port=args.port)

def start_video(video_name, alpha, selected_mask_idx):
    global global_data
    if video_name is None:
        info_text, info_label = global_data.get_info()
        color_map = global_data.get_random_color_map()
        return None, None, None, info_text, info_label, color_map
    global_data.init(video_name, alpha, selected_mask_idx)
    global_data.get_flow()
    global_data.postprocess_prev_frame_masks_torch()
    global_data.automatic_mask_generate()
    curr_frame, prev_frame, first_frame = global_data.segment_frames(alpha)
    info_text, info_label = global_data.get_info()
    print(info_text)
    color_map = global_data.get_random_color_map()
    return curr_frame, prev_frame, first_frame, info_text, info_label, color_map

def accept_mask(alpha):
    global global_data
    global_data.curr_mask_idx += 1
    if global_data.is_last_mask(): # annotation done at current frame
        global_data.save_masks(alpha)
        if global_data.is_last_frame(): # annotation done at last frame
            global_data.make_segmented_video(alpha)
            info_text = [("Annotation done!", "Segmented video saved!" if global_data.make_video else "Segmented video not saved!")]
            info_label = {
                "Frame": 1.0,
                "Mask": 1.0,
            }
            print(info_text)
            return None, None, None, info_text, info_label
        global_data.get_flow()
        global_data.postprocess_prev_frame_masks_torch()
    global_data.selected_points = []
    global_data.automatic_mask_generate()
    curr_frame, prev_frame, first_frame = global_data.segment_frames(alpha)
    info_text, info_label = global_data.get_info()
    print(info_text)
    return curr_frame, prev_frame, first_frame, info_text, info_label

def reject_mask(alpha):
    global global_data
    global_data.reject_mask()
    curr_frame, prev_frame, first_frame = global_data.segment_frames(alpha)
    info_text, info_label = global_data.get_info()
    print(info_text)
    return curr_frame, prev_frame, first_frame, info_text, info_label

def previous_mask(alpha): # TODO
    global global_data
    global_data.previous_mask()
    curr_frame, prev_frame, first_frame = global_data.segment_frames(alpha)
    info_text, info_label = global_data.get_info()
    print(info_text)
    return curr_frame, prev_frame, first_frame, info_text, info_label

def get_point(point_type, alpha, evt: gr.SelectData):
    global global_data
    global_data.get_point(evt.index, point_type)
    global_data.prompt_mask_generate()
    curr_frame, prev_frame, first_frame = global_data.segment_frames(alpha)
    info_text, info_label = global_data.get_info()
    print(info_text)
    return curr_frame, prev_frame, first_frame, info_text, info_label

def undo_point(alpha):
    global global_data
    global_data.undo_point()
    global_data.prompt_mask_generate()
    curr_frame, prev_frame, first_frame = global_data.segment_frames(alpha)
    info_text, info_label = global_data.get_info()
    print(info_text)
    return curr_frame, prev_frame, first_frame, info_text, info_label

def change_sam_mask_size(sam_mask_size, alpha):
    global global_data
    global_data.change_sam_mask_size(sam_mask_size)
    global_data.prompt_mask_generate()
    curr_frame, prev_frame, first_frame = global_data.segment_frames(alpha)
    return curr_frame, prev_frame, first_frame

def previous_frame(alpha):
    global global_data
    global_data.previous_frame()
    curr_frame, prev_frame, first_frame = global_data.segment_frames(alpha)
    info_text, info_label = global_data.get_info()
    print(info_text)
    return curr_frame, prev_frame, first_frame, info_text, info_label

def select_color_map(video_name, alpha, evt: gr.SelectData):
    global global_data
    if video_name is None:
        return None, None, None
    color_idx = evt.index
    global_data.change_color(color_idx)
    curr_frame, prev_frame, first_frame = global_data.segment_frames(alpha)
    return curr_frame, prev_frame, first_frame

def change_alpha(video_name, alpha):
    global global_data
    if video_name is None:
        return None, None, None
    curr_frame, prev_frame, first_frame = global_data.segment_frames(alpha)
    return curr_frame, prev_frame, first_frame

if __name__ == "__main__":
    main(args)