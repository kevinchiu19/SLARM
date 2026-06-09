import os
import gradio as gr
from glob import glob
from PIL import Image

# Supported image formats
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')


class ImageViewer:
    def __init__(self):
        self.data_root_path = 'xxx/SLARM_data/datasets/waymo/training'
        self.scene_paths = sorted(glob(self.data_root_path + '/*'))
        self.scene_id = 0
        self.image_paths = []
        self.current_index = 0
        self.folder_path = self.scene_paths[self.scene_id] + "/images"  # Default image directory
        self.load_images()

    def load_images(self):
        """Load all images in the specified folder"""
        if not os.path.exists(self.folder_path):
            self.image_paths = []
            return False

        # Find all supported image files
        self.image_paths = []
        for ext in IMAGE_EXTENSIONS:
            self.image_paths.extend(glob(os.path.join(self.folder_path, f'*_0{ext}')))  # *_0.jpg indicates front camera
            self.image_paths.extend(glob(os.path.join(self.folder_path, f'*_0{ext.upper()}')))  # *_0.jpg indicates front camera

        # Deduplicate and sort
        self.image_paths = sorted(list(set(self.image_paths)))
        self.current_index = 0 if self.image_paths else 0
        return len(self.image_paths) > 0

    def set_folder(self, folder_path):
        """Set image folder path"""
        if folder_path and os.path.exists(folder_path):
            self.folder_path = folder_path
            self.load_images()
            return f"Loaded folder: {folder_path}, total {len(self.image_paths)} images"
        return f"Folder does not exist: {folder_path}"

    def update_scene(self):
        self.folder_path = self.scene_paths[self.scene_id] + "/images"
        self.load_images()

    def get_current_image(self):
        """Get current image"""
        if not self.image_paths:
            return None, "No images found, please check the folder path"

        try:
            img = Image.open(self.image_paths[self.current_index])
            info = (f"Image {self.current_index + 1}/{len(self.image_paths)}\n"
                #    f"Filename: {os.path.basename(self.image_paths[self.current_index])}")  # Show filename only
                   f"Filename: {self.image_paths[self.current_index]}")  # Show full file path
            return img, info
        except Exception as e:
            return None, f"Unable to load image: {str(e)}"

    def next_image(self):
        """Next image"""
        if self.image_paths and len(self.image_paths) > 1:
            self.current_index = (self.current_index + 1) % len(self.image_paths)
        return self.get_current_image()

    def prev_image(self):
        """Previous image"""
        if self.image_paths and len(self.image_paths) > 1:
            self.current_index = (self.current_index - 1) % len(self.image_paths)
        return self.get_current_image()

    def next_10_image(self):
        """Next 10 images"""
        if self.image_paths and len(self.image_paths) > 10:
            self.current_index = (self.current_index + 10) % len(self.image_paths)
        return self.get_current_image()

    def prev_10_image(self):
        """Previous 10 images"""
        if self.image_paths and len(self.image_paths) > 10:
            self.current_index = (self.current_index - 10) % len(self.image_paths)
        return self.get_current_image()

    def next_scene(self):
        """Next scene"""
        if self.scene_paths and len(self.scene_paths) > 1:
            self.scene_id = (self.scene_id + 1) % len(self.scene_paths)
            self.update_scene()
        return self.get_current_image()

    def prev_scene(self):
        """Previous scene"""
        if self.scene_paths and len(self.scene_paths) > 1:
            self.scene_id = (self.scene_id - 1) % len(self.scene_paths)
            self.update_scene()
        return self.get_current_image()

# Create image viewer instance
viewer = ImageViewer()

# Define Gradio interface
with gr.Blocks(title="Image Viewer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Image Viewer")
    gr.Markdown("For viewing dataset images and filtering samples")

    with gr.Row():
        with gr.Column(scale=1):
            folder_input = gr.Textbox(
                label="Image Folder Path",
                value=viewer.folder_path,
                placeholder="Enter the folder path containing images"
            )
            load_btn = gr.Button("Load Images", variant="primary")
            status_text = gr.Textbox(label="Status", interactive=False)

            gr.Markdown("### Control Buttons")
            with gr.Row():
                prev_btn = gr.Button("Previous")
                next_btn = gr.Button("Next")
            with gr.Row():
                prev_10_btn = gr.Button("Previous 10")
                next_10_btn = gr.Button("Next 10")
            with gr.Row():
                prev_scene_btn = gr.Button("Previous Scene")
                next_scene_btn = gr.Button("Next Scene")

            gr.Markdown("### Supported Formats")
            gr.Textbox(
                value=", ".join(IMAGE_EXTENSIONS),
                interactive=False,
                label="Image Formats"
            )

        with gr.Column(scale=3):
            image_output = gr.Image(label="Image Display")
            info_output = gr.Textbox(label="Image Info", interactive=False)

    # Setup event handlers
    def load_images_from_folder(folder_path):
        status = viewer.set_folder(folder_path)
        img, info = viewer.get_current_image()
        return img, info, status

    # Load images button
    load_btn.click(
        fn=load_images_from_folder,
        inputs=[folder_input],
        outputs=[image_output, info_output, status_text]
    )

    # Previous button
    prev_btn.click(
        fn=viewer.prev_image,
        outputs=[image_output, info_output]
    )

    # Next button
    next_btn.click(
        fn=viewer.next_image,
        outputs=[image_output, info_output]
    )

    # Previous 10 button
    prev_10_btn.click(
        fn=viewer.prev_10_image,
        outputs=[image_output, info_output]
    )

    # Next 10 button
    next_10_btn.click(
        fn=viewer.next_10_image,
        outputs=[image_output, info_output]
    )

    # Previous scene button
    prev_scene_btn.click(
        fn=viewer.prev_scene,
        outputs=[image_output, info_output]
    )

    # Next scene button
    next_scene_btn.click(
        fn=viewer.next_scene,
        outputs=[image_output, info_output]
    )

    # Initial load
    demo.load(
        fn=lambda: (*viewer.get_current_image(), f"Initial folder: {viewer.folder_path}"),
        outputs=[image_output, info_output, status_text]
    )

# Launch application
if __name__ == "__main__":
    demo.launch(debug=True)
