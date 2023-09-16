import os
import shutil
import math
import yaml
import cv2
import imagehash
import logging
from tqdm import tqdm
from pytube import YouTube
from PIL import Image as PILImage
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

    def __init__(self, config_path: str) -> None:
        """
        Initialize a YoutubeVideosToSheetMusicPDF instance with the provided parameters.

        Args:
            config_path (str): The path to the configuration file.
            
        Initializes an instance of the YoutubeVideosToSheetMusicPDF class with the 
        specified YouTube URL, download directory, screenshots directory, and start
        time in seconds. The class can be used to download the YouTube video
        and capture screenshots from the video.
        """

        # Load configuration file.
        with open(config_path, 'r') as config_file:
            self.config = yaml.safe_load(config_file)

        self.youtube_url = self.config['URL']
        self.root_dir = self.config['ROOT_DIR']
        self.start_seconds = self.config['START_SECONDS']
        self.end_seconds = self.config['END_SECONDS']
        self.crop_dimensions = self.config['CROP_DIMENSIONS']

        self.youtube_save_path = f'{self.root_dir}/videos'
        self.screenshots_dir = f'{self.root_dir}/screenshots'
        self.pdf_dir = f'{self.root_dir}/pdfs'

        self.file_name = YouTube(self.youtube_url).title
        self.file_name = " ".join(self.file_name.replace('|', ' ').replace('*', ' ').split())
        self.file_path = f'{self.youtube_save_path}/{self.file_name}.mp4'
        self.screenshots_save_path = f'{self.screenshots_dir}/{self.file_name}'
        self.pdf_path = f'{self.pdf_dir}/{self.file_name}.pdf'

        # Create directory if it does not exist yet.
        for directory in [self.youtube_save_path, self.screenshots_dir, self.pdf_dir]:
            if not os.path.exists(directory):
                os.mkdir(directory)

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

        # Printing task.
        logging.info('Downloading Youtube video as .mp4 format.')

        # Create a Youtube object.
        yt = YouTube(self.youtube_url)
        logging.info(f'Video found: {yt.title}')
        logging.info(f'Downloading...')
        
        # Download the video
        yt.streams \
            .filter(only_video=True, file_extension='mp4') \
            .first() \
            .download(
                output_path=self.youtube_save_path,
                filename=f'{self.file_name}.mp4'
            )

        logging.info('Done.\n')

    def capture_screenshots(self, interval_seconds: int=4) -> None:
        """
        Capture screenshots at specified time intervals from a video file and 
        save them as PNG images. This method opens a video file, reads frames at 
        regular intervals, and saves the frames as PNG images. Screenshots are 
        saved in a directory named after the video's base name within the 
        'screenshots' directory.

        Args:
            interval_seconds (int, optional): The time interval, in seconds, 
                at which to capture screenshots (default=5). This value 
                should always be small enough such that every part of the
                sheet music gets capture. If is too large, some part of the
                sheet music may be skipped. So, it is better to have this value
                a bit too low instead of too large.

        Returns:
            None
        """

        # Printing task.
        logging.info(f'Capturing screenshots each {interval_seconds} seconds.')

        # Create specific directory if it does not exist yet.
        if not os.path.exists(self.screenshots_save_path):
            os.mkdir(self.screenshots_save_path)
        else:
            shutil.rmtree(self.screenshots_save_path)
            os.mkdir(self.screenshots_save_path)

        # Open the video file.
        cap = cv2.VideoCapture(self.file_path)

        # Initialise frame count and obtain frame rate.
        frame_number = 0
        fr = cap.get(cv2.CAP_PROP_FPS)

        # Loop throught the video.
        logging.info('Saving images... ')
        while True:
            # Read a frame from the video
            ret, frame = cap.read()

            if not ret:
                break

            # Capture a screenshot every 'interval_seconds' seconds
            if frame_number > self.start_seconds * fr and math.floor(frame_number % (interval_seconds * fr)) == 0:
                screenshot_file = os.path.join(
                    self.screenshots_save_path,
                    f'screenshot_{int(frame_number // (interval_seconds * fr)) :03d}.png'
                )

                # Save the file.
                cv2.imwrite(screenshot_file, frame)

            frame_number += 1

        # Release the video capture object
        cap.release()
        logging.info('Done.\n')

    def crop_images(self) -> None:
        """
        """
        
        # Printing task
        logging.info('Cropping images...')


        for root, _, files in os.walk(self.screenshots_save_path):
            for file in tqdm(files):
                file_path = f'{root}/{file}'

                # Open the image and create average hash.
                with PILImage.open(file_path) as img:
                    # Obtain current width and height of the image.
                    width, height = img.size

                    # Obtain new cropped image.
                    cropped_image = img.crop((
                        self.crop_dimensions[0] * width,
                        self.crop_dimensions[1] * height,
                        self.crop_dimensions[2] * width,
                        self.crop_dimensions[3] * height
                    ))

                    cropped_image.save(f'{root}/{file}')
            logging.info('Done.\n')


    def remove_duplicate_images(self, threshold: int=1) -> None:
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

        # Printing task.
        logging.info('Deleting all duplicate images.')

        # Empty list for the duplicate images.
        non_duplicate_images = []
        duplicate_images = []
        all_hashes = []

        # Loop through screenshots/files.
        for root, _, files in os.walk(self.screenshots_save_path):
            for i, file in tqdm(enumerate(files)):
                file_path = f'{root}/{file}'

                # Open the image and create average hash.
                with PILImage.open(file_path) as img:
                    all_hashes.append(imagehash.average_hash(img))

                if i == 0:
                    non_duplicate_images.append(file_path)
                else:
                    hamming_distance = all_hashes[-1] - all_hashes[-2]
                    print(file, hamming_distance)
                    if hamming_distance < threshold:
                        duplicate_images.append(file_path)
                    elif hamming_distance >= threshold:
                        non_duplicate_images.append(file_path)

        # Remove duplicate images.
        for duplicate_image in duplicate_images:
            os.remove(duplicate_image)

        logging.info('Done.\n')

    def create_pdf_from_pngs(self) -> None:
        """
        """

        logging.info('Creating .pdf file.')

        # Get a list of all PNG files in the input directory
        png_files = [f for f in os.listdir(self.screenshots_save_path) if f.lower().endswith('.png')]
        png_files.sort() # Sort the files alphabetically

        # Create a PDF document
        doc = SimpleDocTemplate(self.pdf_path, pagesize=letter)

        # Initialize a list of elements to be added to the PDF
        elements = []

        # Iterate through PNG files in pairs and add them to the PDF
        logging.info('Combining images... ')
        for i in tqdm(range(0, len(png_files), 2)):
            image_path1 = os.path.join(self.screenshots_save_path, png_files[i])
            image_path2 = os.path.join(self.screenshots_save_path, png_files[i + 1]) if i + 1 < len(png_files) else None

            # Create PIL Image objects from PNG files
            img1 = PILImage.open(image_path1)
            img2 = PILImage.open(image_path2) if image_path2 else None

            # Resize images to fit half of the page width
            img1.thumbnail((letter[0] / 1.3, letter[1] / 1.3))
            if img2:
                img2.thumbnail((letter[0] / 1.3, letter[1] / 1.3))

            # Convert Pillow images to ReportLab Image objects
            reportlab_img1 = Image(image_path1, width=img1.width, height=img1.height)
            reportlab_img2 = Image(image_path2, width=img2.width, height=img2.height) if img2 else None

            # Create a page for each pair of images
            page_elements = []
            page_elements.append(reportlab_img1)
            if reportlab_img2:
                page_elements.append(reportlab_img2)

            elements.extend(page_elements)

        # Build the PDF document
        doc.build(elements)
        logging.info('Done.\n')  


CONFIG_PATH = 'config.yaml'
YouTubeToPDFConverter = YoutubeVideosToSheetMusicPDF(CONFIG_PATH)

#YouTubeToPDFConverter.youtube_videos_downloader()
#YouTubeToPDFConverter.capture_screenshots()
#YouTubeToPDFConverter.crop_images()
YouTubeToPDFConverter.remove_duplicate_images()
YouTubeToPDFConverter.create_pdf_from_pngs()
