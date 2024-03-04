# face-extractor

![Illustration of how face-extractor works](illustration.png)

Python script that detect faces on the image or video, extracts them and saves to the specified folder.

## Installation

Copy repository to your computer using one of the available methods. For example, this can be done using the `git clone` command:

```sh
git clone https://github.com/kartikdesai-ibo/face-extractor.git
```

Then you need to go to the project folder and install all the dependencies:

```sh
# change directory
cd face-extractor

# install dependencies
pip install -r requirements.txt
```

And you're done.

## Usage

To run the script you need to pass only the path to the video file that need to be processed, as well as the path to the folder where the extracted faces will be saved.

```sh
python extract.py --input v2.mp4 --output media/ --padding 1.5 --frames 30
```

By default, the files are saved in the `output` folder.

**Arguments:**

- `-h, --help`: show this help message and exit
- `-i, --input`: path to input directory or file
- `-o, --output`: path to output directory of faces
- `-p, --padding`: padding ratio around the face (default: 1.0)
- `-p, --frames`: frames to skip in video (default:30)
