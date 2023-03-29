# COMET 
Cost-Optimized Maximal Overlap Estimation Technique

## Introduction

COMET is a software package designed to correct drift in single molecule localization (SMLM)
datasets with a high spatial and temporal resolution. 

## Usage 

### COMET Website 

The easiest way to use COMET is to go to our [dedicated website](https://www.smlm.tools), upload your 
file and let it run on our machine.

#### Step 1

Bring your dataset in the Thunderstorm dataformat:

- .csv file, comma separated
- the following headers: "frame", "x [nm]", "y [nm]" ("z [nm]" for 3D datasets) 
  - -> the "" are part of the header! 
  - the file can have more columns/headers, but the above-mentioned **have to be there**
- check the [Thunderstorm page](https://zitmen.github.io/thunderstorm/) for reference 

#### Step 2

Go to [https://www.smlm.tools](https://www.smlm.tools), press upload and select your 
correctly formatted .csv file

![Image 1 Startup Page](res/comet_startup.png) 

#### Step 3
After the upload is finished, you have to specify the dimension of the dataset, 
the segmentation method and the segmentation parameter depending on the 
segmentation method (s. Segmentation Methods for more detailed explaination).
Press run.

- **Tip #1**: If you're unsure about the segmentation, check the "keep file for 
subsequent analysis" checkbox, to be able to run the analysis again 
with different parameters, without the need to upload the file again


- **Tip #2**: After the successful upload of the file, the server will check how 
many jobs are currently queued, it's that's alot, check out the other methods 
to use the COMET analysis described later. 

![Image 2 Startup Page](res/comet_dataset_load.png)


#### Step 4
After your job is done, a diagram showing the drift curve should appear, 
together with a download button. Press the download button to automatically
download a .csv file containing the drift estimates. 

### Google Colab Notebook
Go to [Google Colab](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwi9ysrgsoH-AhXfSPEDHRoeBzMQFnoECA4QAQ&url=https%3A%2F%2Fcolab.research.google.com%2F&usg=AOvVaw3A5aPK2kLFzKOzb6sOckVw)
and create a new notebook. Under edit->notebook-settings and activate GPU. 
Copy the content of 

    /python_code/pair_indices.py 
to the first line of the notebook.

Copy the content of 

    /python_code/segment_dataset.py
to the second line
of the notebook.
Copy the content of 

    /python_code/drift_optimization_functions_3d.py 

(or [...]_2d.py for two-dimensional datasets) to the third line in the notebook.
Now upload your dataset-file, write your own import function in the 
next line of the notebook and create a numpy array containing the localizations
in the following format:

    localizations.shape = (number_of_localizations, dimensions_of_the_dataset+1)

where

    localizations[:, 0] -> x coordinates

    localizations[:, 1] -> y coordinates

    localizations[:, 2] -> z coordinates (or -> frame/time if 2D dataset)

    localizations[:, 3] -> frame/time 

Now decide on a segmentation method (see Segmentation Methods for detailed
information) and call the corresponding segmentation method, e.g.

    localizations, n_segments  = segment_by_num_windows(localizations, n_segments, return_n_segments=True)

now simply run the drift estimate function, e.g. 

    drift_estimate_nm = optimize_3d_chunked(n_segments, localizations, display_steps=False) 
 
drift_estimate_nm will now be a numpy array with the following format:

    drift_estimate_nm.shape = (number_of_segments, dimension_of_the_dataset)
containing the results in nanometer.

Now you can simply write your own code to look at the result or save it 
however you like. 


## Additional Information
### Segmentation Methods 

3 options to segment the dataset:

1) **segment by number of time windows**: 

this divides the dataset in equal parts containing a fixed 
number of localizations, you can specify the number of parts. 

2) **segments by number of locs per window**

similar to option 1)
but here you have to specify the number of localizations per 
window and the resulting number of segments is calculated. 

3) **segment by number of frames per window**:

this will use the information provided by the dataset and split the dataset in unequal parts, where each part contains all the localizations witin a number of frames that you specify.


