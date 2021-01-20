# Mice Brain Mri Segmentation #
(actually, it can segment humans' scans as well. the repos' name is a bit stupid)

## Disclaimer ##
This repo contains code for our lab personal use.
Not only it lacks test code, but it was formed for a specific use, 
so adaptation for your use case may require some serious editing.

## Installation ##
1. clone the repo
1. install [Anaconda] package manager
1. [create virtual environment from the file] conda_env.yml
1. [install inkscape], [install ImageJ]
1. it is best to install [PyCharm] for code editing, [set up PyCharm to use Anaconda virtual environment]

[Anaconda]: https://www.anaconda.com/products/individual
[create virtual environment from the file]: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file
[PyCharm]: https://www.jetbrains.com/pycharm/download/
[set up PyCharm to use Anaconda virtual environment]: https://www.jetbrains.com/help/pycharm/conda-support-creating-conda-virtual-environment.html
[install inkscape]: https://inkscape.org/
[install ImageJ]: https://imagej.nih.gov/ij/download.html

---

## usage guidelines ##
* All the complex functions etc, are stored in `lib` folder, which is a python package.
* Simple scripts, like the examples from `scripts` folder import modules from lib and use them.
The intention is that these scripts act like shell scripts.
    >In my opinion, it is simpler to write a new script or edit an old one,
    than to put up a useful command line option parser. Also, shell syntax is irritating.

### Intended way of working with a project ###
1. Launch `scripts/new_project.py`, after editing project path in it
(in fact you should edit any script before executing it for the firs time)
1. put source raw images into `{project}/raw_img` folder renaming them
    > it is assumed that images are raw (like a C array, written to a file) and named 
    `{group}_{hour}_{animal}_{frame}` with no file suffix 
    > the naming convention is specified in `lib/default_config.yml` file under the key
    `global/file_naming_convention`. all the filenames are broken down based on the regex,
    formed from that string. the regex forms just fine if you change it, but any other effects were not tested                                                                                  >
1. To presegment images with ImageJ there exists `scripts/brain_segmentation_with_imagej.txt` ImageJ script.
it uses 2 simple text files in a corresponding project folder to store your segmentation progress.
to obtain such files execute `scripts/list_raw_images.py`
working with ImageJ script:
    1. Open `scripts/brain_segmentation_with_imagej.txt` as ImageJ script.
    1. press `P` (macro `select working directory`). Select any file in your project folder
    (for some reason you cannot select folder).
    1. Press `I` (macro `initiate sequence`). the first image should open 
    (it is hard-coded that images are 256*256, 32bit signed integers, but it can be easily
    changed by rewriting openImage() function of the script)
    1. Select a reference zone (it's mean intensity will be used as zero intensity), press `C` (`capture background`).
    1. Draw a line, that matches brain "vertical" direction, press `R` (`capture rotation`).
    the image will be automatically rotate, so that line you drew will become vertical.
    1. Select left/right hemisphere, press `Q`/`W` (`capture left hemisphere`/`capture right hemisphere`).
    1. Repeat for another hemisphere.
    1. Press `G` (`next in sequence`, analogous to `initiate sequence`, but increments integer in `ij_pointer.txt` file,
    if image can be opened).
    1. Repeat previous 5 steps.
    >You can close ImageJ at any image and continue later: `{project}/ij_img_list.txt` stores list of \n separated filenames,
    while `{project}/ij_pointer.txt` stores the position of the last opened file in the list
    >To proceed from where you left, do everything as the first time. after you press `I`, the last opened image will open.
    the segmentation results are dumped as text to `{project}/pre_meta` folder.
1. Execute `scripts/images_to_npy.py` (saves images in a `.npy` format in the `img` folder)
1. Execute `scripts/compose_metadata_crop_images.py` (ImageJ scripts are not an ideal instrument
for reading and writing files, but python contains a lot of out-of-the-box solutions,
so it is logical to use python to produce a single metadata file, from simple array dumps that ImageJ makes).
1. If folder for brain structures' masks is not created already, you need to [make](#Intended-way-of-mask-folder-creation) one.
1. Execute `scripts/segmentation.py` (may take 30-90 per group, depending on the number of comparisons.
also it outputs a huge sheet of text into stdout)
1. Execute `plot_segmentation_summary.py` if needed (not fast either)

### Intended way of mask folder creation ###
1. Execute script `scripts/new_masks_folder.py`. apart from bare folders it creates a simple text file
`{masks_folder}/download_info.yml`. This file requires manual editing. It stores data that allows to retrieve
data from the allen atlas website (URIs are stored in config, though). the download info file is an implicit argument for
the functions, which names start with `download_` and which accept an explicit `masks_folder` argument.
    > File `download_info.yml` is filled with example data. If you do not comment out the `download_` functions from script,
    they will retrieve the example slices from allen institute website. This may take 10-20 seconds.
1. Edit `download_info.yml`. the keys to the atlas slice_ids will be used for naming corresponding files.
They must coincide with `frame` part of the mri scan file names.
    >The function `project_ops.download_slice_ids()` called in `scripts/new_masks_folder.py` retrieves 
    slice ids (which i failed to find a pattern in) and assigns them to a linear range of floating point numbers (coordinates),
    then outputs this into  a csv table named `{masks_folder}/slice_id_table.txt`. this can help to find slices you need.
1. Comment out the call of `project_ops.new_masks_folder()` function in `scripts/new_masks_folder.py`
(raises exceptions if a folder exists). Execute scipt to retrieve needed atlas slices.
1. Each atlas slice is saved to an `.svg` file, which needs a bit of manual editing (with Inkscape).
In each file create a rectangle, that will delimit the part that will be rendered (it is easier to do this with rectangle,
than with metadata editing). Open Inkscape xml-editing tool and give the rectangle an new xml attribute `structure_id`,
with the `bbox` value.
    > You can add various visible elements to `.svg` slices. those can be raster images, lines, polylines, etc.
    On rendering, all the elements, but those without their own graphical representation and those, that have `structure_id`
    attribute, will be turned invisible (tags for elements with no representation are listed in config file).
    All colors will be set to black (background) and white (brain structure).
    Scince file is copied when rendering, the user appearance will not change after the procedure.
1. Edit the `lib/default_config.yml`: under key `ontology/inkscape_exe` insert actual path to inkscape executable
    > To ensure that the path is correct you can call inkscape shell interface, eg: 
    `"c:/users/user/my precious/inkscape" --help` (note that both bash and Windows shell, unlike most languages,
    discriminate against single qoutes for some reason).
1. Execute `scripts/prerender_masks.py`. This takes 0.5-10 min per slice, depending on the element count.
The resolution of the slices will be like one of the corresponding cropped images. 
---

## Core ideas ##
### Files ###
* OS filesystem is a good interface. It also allows for simple and fast manual editing, if needed.
So we use files as the major input and output even if it is not really needed.
* each lib function should preferably change/create only one file or folder inside project/masks folder,
and read only from one (different) file/folder. 
* Not using matching names means that config will contain a lot of unnecessary entries.
So names of folders and single files are usually savagely hard-coded into lib functions.
Also, metadata files have same names as images (except for suffixes).
File names should contain `frame` piece, which should match the name of an svg slice.
There are no randomly generated file names

### File types ###
* arrays are stored as png pictures (if int8/int16 and acceptable channel number) or files with [`.npy`, `.npz` suffixes].
they should not be stored as a raw array, because python is not c++ and hard drives are not 64 Mbytes anymore.
* tables are stored as csv files with `\t` (tabulation) as a delimiter.
* python pickles (serializations of python objects) are stored as .pickle or .pth files.
* dictionaries/lists should be stored as [.yml] or .xml text files.
.yml files are preferable as they are more legible.
[`.npy`, `.npz` suffixes]: https://numpy.org/devdocs/reference/generated/numpy.lib.format.html
[.yml]: https://en.wikipedia.org/wiki/YAML

### Configuration ###
* config is stored in a single .yml file. The outer-most keys coincide with lib modules' names + there is a `global` key.
* After importing a module `filename.py`, one should load config with function call `lib.load_config(filename)`.
* By default, the `lib/default_config.yml` file is loaded.
* Data under key `filename` is loaded into module "constant" `filename._LOC`,
the `filename._GLOB` will contain data from the `global` key.
> for now, config stores the dictionary with `frame` shapes (shapes of mri scan crops and mask renders).
it is far from being a brilliant solution, so this may change)

### Coding style ###
* Variable type declarations are used. They are supported by Pycharm hint system, which is very handy.
* Number of forks is minimized, because they can make debugging hard.
* Random file names can potentially be a problem in recreating bugs or
can be misleading if one tries to manipulate files via ipython console, etc.
* Functions do not throw exceptions for catching, only for information and debugging.
* many functions run some check-up code at the start, to ensure that files are present 
and the files with the same names as function outputs do not exist.
These are `FileExistsError`, `FileNotFoundError`, `AssertionError`, again, not intended for catching.
* writing classes with mutable instances is hard, so I prefer not to write them.
* Some of the logically connected functions are united into "static classes".
It does not seem "pythonic", but may enchance code understanding.
