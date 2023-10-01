"""Converting Youtube videos to sheet music."""
import os
import shutil
import math
import yaml
import cv2
import imagehash
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from pytube import YouTube
from PIL import Image as PILImage
from matplotlib import pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Image

logging.getLogger().setLevel(logging.INFO)


class YoutubeVideosToSheetMusicPDF:
    """
    Class for converting Youtube videos to sheet music.

    Steps:
    1. Download the youtube video.
    2. Each 'x' amount of seconds, take a screenshot of the video, and save it as .png file.
    3. Remove duplicate .png files.
    4. Merge .png files into a pdf.
    """

    def __init__(
        self,
        root_dir: str,
        youtube_url: str,
        song: str,
        composer: str,
        performer: str,
        transcriber: str,
        screenshot_time_interval: int,
        start_seconds: int,
        end_seconds: int,
        crop_dimensions: list,
        n_images: int,
    ) -> None:
        """
        Initialize a YoutubeVideosToSheetMusicPDF instance with the provided parameters.

        Args:
            root_dir (str):

        Returns:
            None
        """

        self.root_dir = root_dir
        self.youtube_url = youtube_url
        self.song = song
        self.composer = composer
        self.performer = performer
        self.transcriber = transcriber
        self.screenshot_time_interval = screenshot_time_interval
        self.start_seconds = start_seconds
        self.end_seconds = end_seconds
        self.crop_dimensions = crop_dimensions
        self.n_images = n_images

        # Create directories in the root directory if they do not exist yet.
        self.video_dir = f"{self.root_dir}/videos"
        self.screenshots_dir = f"{self.root_dir}/screenshots"
        self.pdf_dir = f"{self.root_dir}/pdfs"

        for directory in [self.video_dir, self.screenshots_dir, self.pdf_dir]:
            if not os.path.exists(directory):
                os.mkdir(directory)

        # Setting up filename.
        self.file_name = f"{song} - {performer} (composition by {composer}) (transcription by '{transcriber}')"
        self.video_file_path = f"{self.video_dir}/{self.file_name}.mp4"
        self.screenshots_file_path_prefix = f"{self.screenshots_dir}/{self.file_name}"
        self.pdf_file_path = f"{self.pdf_dir}/{self.file_name}.pdf"

        # For aesthetic purposes.
        print()

    def youtube_videos_downloader(self) -> None:
        """
        Downloads a YouTube video in MP4 format from the provided URL and
        saves it to the specified directory.

        This method creates a YouTube object using the provided YouTube URL,
        filters for the first available video-only stream in MP4 format,
        and downloads the video to the designated output directory.

        Args:
            None

        Returns:
            None
        """

        # Logging task.
        logging.info(" ### Downloading Youtube video as .mp4 format.")

        # Create a Youtube object.
        yt = YouTube(self.youtube_url)
        logging.info(f"Video found: {yt.title}")
        logging.info(f"Downloading...")

        # Download the video
        yt.streams.filter(only_video=True, file_extension="mp4").first().download(
            output_path=self.video_dir, filename=f"{self.file_name}.mp4"
        )

        logging.info(f"File saved as '{self.file_name}.mp4'.\n")

    def capture_screenshots(self) -> None:
        """
        Capture screenshots at specified time intervals from a video file and
        save them as PNG images. This method opens a video file, reads frames at
        regular intervals, and saves the frames as PNG images. Screenshots are
        saved in a directory named after the video's base name within the
        'screenshots' directory.

        Args:
            interval_seconds (int, optional): The time interval, in seconds,
                at which to capture screenshots (default=screenshot_time_interval). This value
                should always be small enough such that every part of the
                sheet music gets capture. If is too large, some part of the
                sheet music may be skipped. So, it is better to have this value
                a bit too low instead of too large.

        Returns:
            None
        """

        # Logging task.
        logging.info(
            f" ### Capturing screenshots each {self.screenshot_time_interval} seconds."
        )

        # Create specific directory if it does not exist yet.
        if not os.path.exists(self.screenshots_file_path_prefix):
            os.mkdir(self.screenshots_file_path_prefix)
        else:
            shutil.rmtree(self.screenshots_file_path_prefix)
            os.mkdir(self.screenshots_file_path_prefix)

        # Open the video file.
        cap = cv2.VideoCapture(self.video_file_path)

        # Initialise frame count and obtain frame rate.
        frame_number = 0
        fr = cap.get(cv2.CAP_PROP_FPS)
        total_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Loop throught the video.
        logging.info("Looping through all frames... ")
        progress_bar = iter(tqdm(range(total_num_frames)))

        while True:
            # Read a frame from the video
            ret, frame = cap.read()

            if not ret:
                break

            # Capture a screenshot every 'self.screenshot_time_interval' seconds
            if (
                frame_number > 0 + (self.start_seconds * fr)
                and frame_number < total_num_frames - (self.end_seconds * fr)
                and math.floor(frame_number % (self.screenshot_time_interval * fr)) == 0
            ):
                # Set up screenshot filename.
                screenshot_file_name = f"screenshot_{int(frame_number//(self.screenshot_time_interval*fr)):03d}.png"
                screenshot_file_path = (
                    f"{self.screenshots_file_path_prefix}/{screenshot_file_name}"
                )
                # Save the file.
                cv2.imwrite(screenshot_file_path, frame)

            frame_number += 1
            next(progress_bar)

        # Release the video capture object
        cap.release()
        logging.info("Done.\n")

    def crop_images(self) -> None:
        """ """

        # Check if cropping is necessary.
        if self.crop_dimensions == [0.0, 0.0, 1.0, 1.0]:
            return None

        # Logging task
        logging.info(" ### Cropping images.")

        for root, _, files in os.walk(self.screenshots_file_path_prefix):
            for file in tqdm(files):
                file_path = f"{root}/{file}"

                # Open the image and create average hash.
                with PILImage.open(file_path) as img:
                    # Obtain current width and height of the image.
                    width, height = img.size

                    # Obtain new cropped image.
                    cropped_image = img.crop(
                        (
                            self.crop_dimensions[0] * width,
                            self.crop_dimensions[1] * height,
                            self.crop_dimensions[2] * width,
                            self.crop_dimensions[3] * height,
                        )
                    )

                    cropped_image.save(f"{root}/{file}")
            logging.info("Done.\n")

    def convert_to_black_and_white(self, threshold: int=192, new_rgb_value: int=255) -> None:
        """
        """

        # Logging task.
        logging.info(" ### Converting images to perfect black and white.")

        # Loop through files.
        for root, _, files in os.walk(self.screenshots_file_path_prefix):
            for file in tqdm(files):
                # File path to the specific image.
                file_path = f"{root}/{file}"

                # Read image.
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

                # Create new purely black and white image.
                _, bw_image = cv2.threshold(image, threshold, new_rgb_value, cv2.THRESH_BINARY)

                # Overwrite image.
                cv2.imwrite(f"{root}/{file}", bw_image)

        logging.info("Done.\n")

    def grid_pixel_count(self, num_rows: int=25, num_columns: int=25) -> None:
        """
        """

        # Logging task.
        logging.info(" ### Counting black & white pixels.")

        black_pixel_list = []
        num_total_pixels_cell = None

        # Loop through files.
        for root, _, files in os.walk(self.screenshots_file_path_prefix):
            for file in tqdm(files):
                # File path to the specific image.
                file_path = f"{root}/{file}"

                # Read image.
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

                # Divide the image up in cells specified by num_rows and num_columns.
                height, width = image.shape
                cell_height, cell_width = height // num_rows, width // num_columns
                black_pixels_per_cell = []

                # Loop through rows and columns.
                for i in range(num_rows):
                    for j in range(num_columns):
                        # Define cell corner coordinates.
                        x_start = j*cell_width
                        x_end = (j+1)*cell_width
                        y_start = i*cell_height
                        y_end = (i+1)*cell_height

                        # Define cell.
                        cell = image[y_start:y_end, x_start:x_end]
                
                        # Count per cell.
                        num_total_pixels_cell = cell.size
                        num_white_pixels = cv2.countNonZero(cell)
                        num_black_pixels = num_total_pixels_cell - num_white_pixels
                        black_pixels_per_cell.append(num_black_pixels)

                black_pixel_list.append(black_pixels_per_cell)

        diff_list = []

        # Compute differences per image.
        for i in range(1, len(black_pixel_list)):
            # Compute the difference between two subsequent images by
            # averaging over all the cells.
            avg_diff = np.average(abs(np.array(black_pixel_list[i]) - np.array(black_pixel_list[i-1]))) / num_total_pixels_cell
            diff_list.append((files[i], avg_diff))
            #plt.scatter(i, avg_diff)

        #plt.savefig(f'figs/{self.song}_{num_rows}.png', dpi=200)
        #plt.close()

        diff_list = sorted(diff_list, key=lambda x: x[1])
        smallest_rsme = float('inf')
        boundary_index = None

        for i, (file, avg_diff) in enumerate(diff_list):
            if i > 0:
                y_middle = (avg_diff + diff_list[i-1][1])/2
                sum_diff = 0

                # Compute sum of squared differences in y-values.
                for _, (_, y) in enumerate(diff_list):
                    sum_diff += (y-y_middle)**2

                # Replace smalles difference value if difference is smaller.
                if sum_diff < smallest_rsme:
                    smallest_rsme = sum_diff
                    boundary_index = i

        duplicate_images = [tup[0] for tup in diff_list[:boundary_index]]
        logging.info("Done.\n")
        return duplicate_images

    def delete_duplicate_images(self, list_of_images: list) -> None:
        """
        """

        # Logging task.
        logging.info(" ### Deleting all duplicate images.")

        # Loop through screenshots/files.
        for root, _, files in os.walk(self.screenshots_file_path_prefix):
            for i, file in tqdm(enumerate(list_of_images)):
                file_path = f"{root}/{file}"

                # Delete image.
                os.remove(file_path)

        logging.info("Done.\n")

    def remove_duplicate_images(self, threshold: int = 1) -> None:
        """
        Remove duplicate PNG images from the specified directory
        using image hashing.

        This method scans the directory where PNG images are stored,
        computes an average hash for each image, and identifies and
        removes duplicate images. The function uses the `imagehash`
        library to efficiently compare images.

        Args:
            threshold (int): The threshold that determines if images
                are equal to each other. (default=2)

        Returns:
            None
        """

        # Logging task.
        logging.info(" ### Deleting all duplicate images.")

        # Empty list for the duplicate images.
        non_duplicate_images = []
        duplicate_images = []
        all_hashes = []

        # Loop through screenshots/files.
        for root, _, files in os.walk(self.screenshots_file_path_prefix):
            for i, file in tqdm(enumerate(files)):
                file_path = f"{root}/{file}"

                # Open the image and create average hash.
                with PILImage.open(file_path) as img:
                    all_hashes.append(imagehash.average_hash(img))

                if i == 0:
                    non_duplicate_images.append(file_path)
                else:
                    hamming_distance = all_hashes[-1] - all_hashes[-2]
                    if hamming_distance < threshold:
                        duplicate_images.append(file_path)
                    elif hamming_distance >= threshold:
                        non_duplicate_images.append(file_path)

        # Remove duplicate images.
        for duplicate_image in duplicate_images:
            os.remove(duplicate_image)

        logging.info("Done.\n")

    def create_pdf_from_pngs(self) -> None:
        """ """

        logging.info(" ### Creating .pdf file.")

        # Get a list of all PNG files in the input directory
        png_files = [
            f
            for f in os.listdir(self.screenshots_file_path_prefix)
            if f.lower().endswith(".png")
        ]
        png_files.sort()  # Sort the files alphabetically

        # Create a PDF document
        doc = SimpleDocTemplate(self.pdf_file_path, pagesize=letter)

        # Initialize a list of elements to be added to the PDF
        elements = []

        # Iterate through PNG files in pairs and add them to the PDF
        logging.info("Combining images... ")
        for i in tqdm(range(0, len(png_files), 2)):
            image_path1 = os.path.join(self.screenshots_file_path_prefix, png_files[i])
            image_path2 = (
                os.path.join(self.screenshots_file_path_prefix, png_files[i + 1])
                if i + 1 < len(png_files)
                else None
            )

            # Create PIL Image objects from PNG files
            img1 = PILImage.open(image_path1)
            img2 = PILImage.open(image_path2) if image_path2 else None

            # Resize images to fit half of the page width
            img1.thumbnail((letter[0] / 1.3, letter[1] / 1.3))
            if img2:
                img2.thumbnail((letter[0] / 1.3, letter[1] / 1.3))

            # Convert Pillow images to ReportLab Image objects
            reportlab_img1 = Image(image_path1, width=img1.width, height=img1.height)
            reportlab_img2 = (
                Image(image_path2, width=img2.width, height=img2.height)
                if img2
                else None
            )

            # Create a page for each pair of images
            page_elements = []
            page_elements.append(reportlab_img1)
            if reportlab_img2:
                page_elements.append(reportlab_img2)

            elements.extend(page_elements)

        # Build the PDF document
        doc.build(elements)
        logging.info("Done.\n")


# Set up config paths.
MAIN_CONFIG_PATH = "main_config.yaml"
PDF_CONFIG_PATH = "pdf_config.csv"


def main():
    """ """

    # Load main configuration file.
    with open(MAIN_CONFIG_PATH, "r", encoding="utf-8") as main_config_file:
        main_config = yaml.safe_load(main_config_file)

    # Load pdf config file.
    pdf_df = pd.read_csv(
        filepath_or_buffer=PDF_CONFIG_PATH, sep=",", header=0, index_col=False
    )

    # Only select rows where the 'generate' column equals 'True'.
    pdf_df = pdf_df[pdf_df["generate"]]

    # Import settings from the main config file.
    root_dir = main_config["ROOT_DIR"]

    for _, row in pdf_df.iterrows():
        # Music related variables.
        youtube_url = row["url"]
        song = row["song"]
        composer = row["composer"]
        performer = row["performer"]
        transcriber = row["transcriber"]

        # Time related variables.
        screenshot_time_interval = row["screenshot_time_interval"]
        start_seconds = row["start_seconds"]
        end_seconds = row["end_seconds"]

        # Space related variables.
        crop_dimensions = [
            float(row["crop_dimensions_left"]),
            float(row["crop_dimensions_top"]),
            float(row["crop_dimensions_right"]),
            float(row["crop_dimensions_bottom"]),
        ]

        # Verificational related variables.
        n_images = row["n_images"]

        YouTubeToPDFConverter = YoutubeVideosToSheetMusicPDF(
            root_dir,
            youtube_url,
            song,
            composer,
            performer,
            transcriber,
            screenshot_time_interval,
            start_seconds,
            end_seconds,
            crop_dimensions,
            n_images,
        )

        #YouTubeToPDFConverter.youtube_videos_downloader()
        YouTubeToPDFConverter.capture_screenshots()
        YouTubeToPDFConverter.crop_images()
        YouTubeToPDFConverter.convert_to_black_and_white()
        #for i, j in zip(range(1, 30), range(1, 30)):
        #    duplicate_images = YouTubeToPDFConverter.grid_pixel_count(i, j)
        duplicate_images = YouTubeToPDFConverter.grid_pixel_count()
        YouTubeToPDFConverter.delete_duplicate_images(duplicate_images)
        #YouTubeToPDFConverter.remove_duplicate_images()
        #YouTubeToPDFConverter.create_pdf_from_pngs()


if __name__ == "__main__":
    main()
