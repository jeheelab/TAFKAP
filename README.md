# TAFKAP
(Matlab) code for the probabilistic brain decoding algorithms "PRINCE" and "The Algorithm Formerly Known As Prince"

    This code runs the PRINCE or TAFKAP decoding algorithm. The original
    PRINCE algorithm was first described here: 

    https://www.nature.com/articles/nn.4150

    The new TAFKAP algorithm is described here:

    https://www.biorxiv.org/content/10.1101/2021.03.04.433946v1

    The main input to the code is 'samples'. If you are using the code to
    decode fMRI data, the rows in this matrix will likely correspond to
    trials, while the columns will correspond to voxels. The variable 
    names used in the code reflect this usage. However, note that the different
    columns could equally be different EEG/MEG channels, or any other set
    of variables that you measured. Similarly, rows do not have to be
    trials, but could be any set of measurements that you took of your
    voxels or other variables. 

    The second input, 'p', is a struct that allows you to customize        
    certain settings that influence how the algorithm runs. These
    settings are explained in more detail in commments, next to the code that
    defines their default values. In this struct, you must also supply some labels
    for your data; namely a list of stimulus values ('stimval'), a list
    of "run numbers" ('runNs') and two lists of binary indicator
    variables ('train_trials' and 'test_trials'), that tell the code 
    which trials to use for training, and which to use for testing. The
    stimulus values in 'stimval' must be circular and in the range of 
    [0, 180]. For instance, it could be the orientation of a visual
    stimulus, measured in degrees. However, it could also be (for
    instance) a color value or direction of motion - as long as these
    values are rescaled to [0, 180] (just take care, in that case, to
    transform the decoder outputs back to their original scale, e.g. [0, 360]
    or [0, 2*pi]. Non-circular or discrete values aren't implemented
    here, but the code could be easily adapted for these cases, by 
    altering the basis functions (defined in fun_basis.m) that are used 
    to fit tuning functions to voxels (or other response variables). The
    indices in 'runNs' can correspond to the indices of the fMRI runs
    from which each trial was taken. More broadly, they serve as indices
    to set up an inner cross-validation loop within the training data, to
    find the best hyperparameters for the TAFKAP algorithm (for PRINCE, 
    these indices do not need to be specified. If your data are not
    divided into fMRI runs, you should choose another way to divide your
    data into independent partitions. 
    
    The main code to run is TAFKAP_decode.m. To see how the code works, 
    you can run it without any input, in which case some data will be 
    simulated for you. The function returns four outputs: 
    -'est': an array of stimulus estimates (one for each trial)
    -'unc': an array of uncertainty values (one for each trial)
    -'liks': a [trial x stimulus_value] matrix of normalized
    likelihoods/posteriors
    -'hypers': the best setting for the hyperparameters (lambda_var and
    lambda) that was found on the training data (not applicable to PRINCE) 

    If your use of this code leads to some form of publication, please
    cite one of the following papers for attribution:

    For PRINCE:
    van Bergen, R. S., Ma, W. J., Pratte, M. S., & Jehee, J. F. M. (2015). 
    Sensory uncertainty decoded from visual cortex predicts behavior. 
    Nature neuroscience, 18(12), 1728-1730. https://doi.org/10.1038/nn.4150

    For TAFKAP:
    van Bergen, R.S., & Jehee, J. F. M. (2021). TAFKAP: An improved 
    method for probabilistic decoding of cortical activity. 
    https://www.biorxiv.org/content/10.1101/2021.03.04.433946v1
    
    PYTHON CODE:
    This repository now also includes a Python port of our original 
    Matlab code. Please note that this version has not undergone as
    much testing as the Matlab version, as we have to date only used
    the latter in our own analyses. 
